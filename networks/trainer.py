import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        # ── model kwargs (backward compatible) ──
        model_kwargs = dict(
            num_classes=1,
            use_attn_pool=getattr(opt, 'use_attn_pool', False),
            multi_scale=getattr(opt, 'multi_scale', False),
            use_sobel=getattr(opt, 'use_sobel', False),
            use_tkp=getattr(opt, 'use_tkp', False),
            tkp_k=getattr(opt, 'tkp_k', 5),
            full_layers=getattr(opt, 'full_layers', True),
            use_post_bn=getattr(opt, 'use_post_bn', True),
        )

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=False, **model_kwargs)

        if not self.isTrain or opt.continue_train:
            self.model = resnet50(**model_kwargs)

        self.use_tkp = model_kwargs['use_tkp']

        if self.isTrain:
            pos_weight = getattr(opt, 'pos_weight', 1.5)
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]))
            if pos_weight != 1.0:
                print(f'BCEWithLogitsLoss pos_weight={pos_weight}')
            # initialize optimizers
            if opt.optim == 'adam':
                wd = getattr(opt, 'weight_decay', 1e-4)
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=opt.lr, betas=(opt.beta1, 0.999),
                    weight_decay=wd)
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])
        if self.isTrain and hasattr(self, 'loss_fn'):
            self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(self.device)

        # TKP: alpha weight for auxiliary loss
        self.tkp_alpha = getattr(opt, 'tkp_alpha', 0.1)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*' * 25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*' * 25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    # ── forward: return main + aux output for TKP ──
    def forward(self):
        out = self.model(self.input)
        if self.use_tkp:
            # During training, TKP returns vec, vec_aux (see tkp.py)
            # For simplicity we compute aux loss inside optimize_parameters.
            self.output = out
        else:
            self.output = out

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        loss_main = self.loss_fn(self.output.squeeze(1), self.label)

        # ── fc1 weight L2 penalty (prevents explosion from all-positive GAP features) ──
        fc1_reg = 0.0
        fc1_wd = getattr(self.opt, 'fc1_wd', 0.01)
        if fc1_wd > 0:
            fc1_reg = fc1_wd * (self.model.fc1.weight.pow(2).sum() +
                                self.model.fc1.bias.pow(2).sum())

        # ── TKP auxiliary loss ──
        if self.use_tkp and hasattr(self.model, 'tkp_vec_aux'):
            fc_aux = self.model.fc1(self.model.tkp_vec_aux)
            loss_aux = self.loss_fn(fc_aux.squeeze(1), self.label)
            self.loss = loss_main + self.tkp_alpha * loss_aux + fc1_reg
        else:
            self.loss = loss_main + fc1_reg

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

