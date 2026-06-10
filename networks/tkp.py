import torch
import torch.nn as nn


class TopKPooling(nn.Module):
    """Top-K Pooling: retain the K strongest activations per channel.

    Compared to GAP which averages all spatial positions (diluting sparse
    discriminative signals), TKP preserves the most informative local
    forgery patterns.  This is especially beneficial after JPEG or blur
    degradation, where only a few spatial locations still carry detectable
    NPR artifacts.

    Regularisation helpers (RBLD and RKS) are only active during training.
    """

    def __init__(self, channels, k=5):
        super().__init__()
        self.channels = channels
        self.k = k

    def forward(self, x, use_rbld=True, use_rks=True):
        """Args:
            x: [B, C, H, W]  feature maps
            use_rbld: Rank-Based Linear Dropout  (training only)
            use_rks:  Random-K Sampling            (training only)
        Returns:
            vec:       [B, C*k]  regular vector for classification
            vec_aux:   [B, C*k]  auxiliary vector (RKS), zero during eval
        """
        B, C, H, W = x.shape

        # ── Top-K selection ──
        x_flat = x.view(B, C, H * W)               # [B, C, N]
        topk_vals, _ = torch.topk(x_flat, self.k, dim=-1)  # [B, C, k]

        training = self.training and use_rbld
        if training:
            topk_vals = self._rbld(topk_vals)       # rank-based linear dropout

        vec = topk_vals.reshape(B, C * self.k)      # [B, C*k]

        # ── Random-K Sampling (auxiliary) ──
        training_rks = self.training and use_rks
        if training_rks:
            vec_aux = self._rks(x_flat)             # [B, C*k]
        else:
            vec_aux = torch.zeros_like(vec)

        return vec, vec_aux

    # ------------------------------------------------------------------
    #  Rank-Based Linear Dropout  (LFM, Algorithm 1)
    # ------------------------------------------------------------------
    def _rbld(self, topk_vals):
        """Apply channel-independent dropout with probability linear in rank.
        Higher-rank (less important) values have higher dropout probability.
        """
        B, C, k = topk_vals.shape
        device = topk_vals.device

        # linear prob from p_min (rank 1) to p_max (rank k)
        p_min, p_max = 0.1, 0.3
        probs = torch.linspace(p_min, p_max, k, device=device)  # [k]
        probs = probs.view(1, 1, k).expand(B, C, k)

        mask = torch.rand(B, C, k, device=device) >= probs
        return topk_vals * mask.float()

    # ------------------------------------------------------------------
    #  Random-K Sampling
    # ------------------------------------------------------------------
    def _rks(self, x_flat):
        """Randomly sample k positions per channel, then sort.
        Provides auxiliary gradient paths to non-dominant features.
        """
        B, C, N = x_flat.shape
        device = x_flat.device

        # sample k random indices per channel
        idx = torch.randint(0, N, (B, C, self.k), device=device)   # [B,C,k]
        rnd_vals = x_flat.gather(dim=-1, index=idx)                # [B,C,k]
        rnd_vals, _ = torch.sort(rnd_vals, dim=-1)
        return rnd_vals.reshape(B, C * self.k)                     # [B,C*k]


class SNet(nn.Module):
    """Lightweight Salience Network (LFM, Sec 3.2).

    Five convolutional layers interleaved with three 2×2 max-pooling
    layers.  2×2 kernels are used throughout to match the 2×2 sampling
    window of the NPR extraction module.
    """

    def __init__(self, in_ch=9):
        super().__init__()
        self.net = nn.Sequential(
            # block 1
            nn.Conv2d(in_ch, 32, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # block 2
            nn.Conv2d(32, 32, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # block 3
            nn.Conv2d(32, 32, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # block 4
            nn.Conv2d(32, 64, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # block 5
            nn.Conv2d(64, 64, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # projection
            nn.Conv2d(64, 64, 1),
        )

    def forward(self, x):
        return self.net(x)
