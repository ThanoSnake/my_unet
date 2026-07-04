#
# Soft-Binned ECE (SB-ECE) -- a differentiable calibration loss
#
# A trainable surrogate of the Expected Calibration Error (Karandikar et al.,
# "Soft Calibration Objectives for Neural Networks", NeurIPS 2021). Plain ECE is
# non-differentiable because of (a) hard confidence binning and (b) the argmax in
# accuracy. Here:
#   - the hard binning is replaced by a SOFT RBF membership to fixed bin centers
#     (differentiable in the confidence), and
#   - correctness is used as a detached target (its non-differentiable argmax is
#     treated as a constant), so gradients flow only through the confidences.
# Minimising it pushes the per-bin confidence toward the per-bin accuracy -> the
# network learns to state calibrated confidence, WITHOUT a post-hoc temperature.
#
# Unlike DCA (a single-bin ECE, blind to over/under-confidence that cancels on
# average), the multi-bin form can correct the S-shaped (crossing) miscalibration.
# Computed on the foreground ROI (gt>0 | pred>0) so the easy background does not
# dominate. Meant as a REGULARISER added to Dice + CE (weight it with `weight`).
#

import torch
import torch.nn as nn


class SoftBinnedECELoss(nn.Module):

    def __init__(self, n_bins=15, bandwidth=None, weight=1.0, foreground_only=True):
        super().__init__()
        self.n_bins = n_bins
        self.weight = weight
        self.foreground_only = foreground_only
        # fixed bin centers in (0, 1); bandwidth ~ one bin width by default
        centers = (torch.arange(n_bins).float() + 0.5) / n_bins
        self.register_buffer("centers", centers)
        self.bandwidth = bandwidth if bandwidth is not None else (1.0 / n_bins)

    def forward(self, logits, target):
        # logits: [B, C, H, W]; target: [B, H, W] or [B, 1, H, W]
        logits = logits.float()                                   # stable under autocast
        if target.dim() == logits.dim():
            target = target[:, 0]
        target = target.long()

        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)                             # [B, H, W]
        correct = (pred == target).float().detach()              # non-diff target

        mask = ((target > 0) | (pred > 0)) if self.foreground_only \
            else torch.ones_like(target, dtype=torch.bool)
        if mask.sum() == 0:
            return logits.sum() * 0.0                             # no fg: 0, keep graph

        c = conf[mask].reshape(-1)                                # [N] (differentiable)
        a = correct[mask].reshape(-1)                             # [N] (detached)
        n = c.numel()

        # soft membership u[i, j] to bin j (RBF), normalised per pixel
        d = c.unsqueeze(1) - self.centers.unsqueeze(0)           # [N, M]
        u = torch.exp(-(d / self.bandwidth) ** 2)                # [N, M]
        u = u / (u.sum(dim=1, keepdim=True) + 1e-12)

        w = u.sum(dim=0)                                          # [M] soft bin counts
        bin_conf = (u * c.unsqueeze(1)).sum(dim=0) / (w + 1e-12)
        bin_acc = (u * a.unsqueeze(1)).sum(dim=0) / (w + 1e-12)
        sb_ece = ((w / n) * (bin_acc - bin_conf).abs()).sum()
        return self.weight * sb_ece
