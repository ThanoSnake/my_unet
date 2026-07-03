#
# Morphological Loss
#
# penalises |p - opening(p)| (tophat) and |closing(p) - p| (bottomhat)
# clean segmentation is near idempotent under opening and closing
# a non zero value of either flags speckles (removed by opening) or pinholes (filled by closing)
# we approximate dilation and erosion via logsumexp as hard min/max operations cause gradients
# to become zero and backpropagation not to work (the dead gradient problem)
#

import torch
import torch.nn as nn
import torch.nn.functional as F

#
# |p - opening(p)| + |closing(p) - p|
#
class MorphConsistencyLoss(nn.Module):

    def __init__(self, weight=0.1, k=3, beta=10.0):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.weight = weight
        self.k = k
        self.pad = k // 2
        self.register_buffer("beta", torch.tensor(float(beta)))

    def _neigh(self, x):
        # x: (N,1,H,W) -> (N,1,k*k,H,W) (local neighbourhoods)
        N, C, H, W = x.shape
        cols = F.unfold(x, self.k, padding=self.pad)
        return cols.view(N, C, self.k * self.k, H, W)

    # soft dilation = (1/b) * logsumexp(b * (neigh + SE)), here SE = 0
    def soft_dilation(self, x):
        return torch.logsumexp(self.beta * self._neigh(x), dim=2) / self.beta

    # soft erosion = -(1/b) * logsumexp(-b * (neigh - SE)), here SE = 0
    def soft_erosion(self, x):
        return -torch.logsumexp(-self.beta * self._neigh(x), dim=2) / self.beta

    def forward(self, pred_softmax):
        fg = pred_softmax[:, 1:]    # (B, C, H, W), classes 1.. are foreground
        B, C, H, W = fg.shape
        x = fg.reshape(B * C, 1, H, W)
        opening = self.soft_dilation(self.soft_erosion(x))
        closing = self.soft_erosion(self.soft_dilation(x))
        consistency = (x - opening).abs().mean() + (x - closing).abs().mean()
        return self.weight * consistency
