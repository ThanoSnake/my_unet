#
# Boundary loss (Kervadec et al., MIDL 2019) -- a distance-weighted surface loss
#
# L_B = mean over pixels of  sum_c softmax_c(p) * phi_c(p),  where phi_c is the
# level-set (signed distance) map of ground-truth class c: phi_c < 0 INSIDE the
# object, > 0 OUTSIDE. Putting probability mass inside the object (phi<0) LOWERS
# the loss; mass far outside is penalised in proportion to its distance -> the
# term acts directly on the surface / ASSD, complementing region losses (Dice /
# Tversky) which are distance-blind. phi is PRECOMPUTED in preprocessing
# (datasets/preprocessing_boundary.py) and rides alongside the label, so at train
# time this is just a cheap elementwise product -- no distance transform on the fly.
#
# It is a SURFACE regulariser, never used alone: schedule it as
#   region + lam * boundary,   with lam ramped up over training
# (Kervadec's "increase" strategy). The training loop passes lam in. NOTE the
# value can be negative -- that is expected and correct for this loss.
#

import torch.nn as nn


class BoundaryLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, probs, phi):
        # probs: [B, C, H, W] softmax ; phi: [B, K, H, W] signed distance of the K=C-1 fg classes
        fg = probs[:, 1:].float()                       # drop background -> [B, K, H, W]
        if phi is None:
            raise ValueError("BoundaryLoss needs the precomputed distance maps (phi is None)")
        if fg.shape[1] != phi.shape[1]:
            raise ValueError(f"boundary: {fg.shape[1]} fg classes vs {phi.shape[1]} phi maps")
        return (fg * phi.float()).mean()
