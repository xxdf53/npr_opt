import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from networks.attention_pool import AttentionPool
from networks.tkp import TopKPooling, SNet

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 use_attn_pool=False, multi_scale=False, use_sobel=False,
                 use_tkp=False, tkp_k=5):
        super(ResNet, self).__init__()

        self.multi_scale = multi_scale
        self.use_sobel = use_sobel
        self.use_tkp = use_tkp

        # ── Sobel edge kernels (frozen) ──
        if use_sobel:
            sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=torch.float32) / 8.0
            sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=torch.float32) / 8.0
            self.register_buffer('sobel_x', sx.view(1, 1, 3, 3))
            self.register_buffer('sobel_y', sy.view(1, 1, 3, 3))

        # ── conv1 input channels ──
        in_ch = 9 if multi_scale else 3    # 3 scales × 3 rgb = 9 channels

        self.unfoldSize = 2
        self.unfoldIndex = 0
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # ── pooling ──
        self.use_attn_pool = use_attn_pool
        if use_attn_pool:
            self.attn_pool = AttentionPool(512)
        elif use_tkp:
            self.tkp = TopKPooling(512, k=tkp_k)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── fc ──
        fc_in = 512 * tkp_k if use_tkp else 512
        self.fc1 = nn.Linear(fc_in, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def interpolate(self, img, factor):
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode='nearest',
                          recompute_scale_factor=True),
            scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)

    # ── forward ────────────────────────────────────────────────────
    def forward(self, x):
        # 1. Multi-scale NPR extraction ─────────────────────────
        NPR_05 = x - self.interpolate(x, 0.50)          # f≈0.5
        if self.multi_scale:
            NPR_025 = x - self.interpolate(x, 0.25)     # f≈0.25
            NPR_075 = x - self.interpolate(x, 0.75)     # f≈0.33 + aliasing
            npr = torch.cat([NPR_025, NPR_05, NPR_075], dim=1)  # [B,9,H,W]
            npr = npr * (2.0 / 3.0)
        else:
            npr = NPR_05 * (2.0 / 3.0)                 # [B,3,H,W]

        # 2. Sobel edge-guided weighting ──────────────────────
        if self.use_sobel:
            gray = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
            gx = F.conv2d(gray, self.sobel_x, padding=1)
            gy = F.conv2d(gray, self.sobel_y, padding=1)
            edge = (gx.pow(2) + gy.pow(2)).sqrt()               # [B,1,H,W]
            edge_norm = edge / (edge.max() + 1e-8)
            weight = torch.sigmoid(edge_norm * 5.0)             # sharpen
            # expand to match npr channels
            weight = weight.expand_as(npr)
            npr = npr * weight

        # 3. Backbone ─────────────────────────────────────────
        x = self.conv1(npr)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # 4. Pooling ──────────────────────────────────────────
        if self.use_attn_pool:
            x = self.attn_pool(x)
            x = x.view(x.size(0), -1)
        elif self.use_tkp:
            vec, vec_aux = self.tkp(x)     # [B, C*k], [B, C*k]
            self.tkp_vec_aux = vec_aux     # store for auxiliary loss
            x = vec
        else:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        # 5. Classification ────────────────────────────────────
        x = self.fc1(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, use_attn_pool=False,
             multi_scale=False, use_sobel=False, use_tkp=False, tkp_k=5,
             **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained  (bool): If True, load ImageNet pre-trained weights.
                            Not compatible with multi_scale/use_tkp.
        use_attn_pool (bool): Use attention pooling instead of GAP.
        multi_scale  (bool): Use 3-scale NPR (0.25/0.50/0.75).
        use_sobel    (bool): Enable Sobel edge-guided weighting.
        use_tkp      (bool): Replace GAP with Top-K Pooling.
        tkp_k        (int):  Number of top activations per channel.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   use_attn_pool=use_attn_pool,
                   multi_scale=multi_scale,
                   use_sobel=use_sobel,
                   use_tkp=use_tkp,
                   tkp_k=tkp_k,
                   **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
