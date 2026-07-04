#
# clDice / soft-skeleton loss (Shit et al., CVPR 2021)
#
# A TOPOLOGY-preserving loss designed for tubular structures (hepatic vessels!).
# It compares the SOFT SKELETONS (centerlines) of prediction and ground truth:
#   Tprec = |skel(pred)  & gt|   / |skel(pred)|      (skeleton of pred inside gt)
#   Tsens = |skel(gt)    & pred| / |skel(gt)|        (skeleton of gt inside pred)
#   clDice = 2*Tprec*Tsens / (Tprec+Tsens)
# A prediction is rewarded only if its centerline stays inside the GT and the GT
# centerline stays inside the prediction -> preserves CONNECTIVITY, discouraging
# broken / merged vessels that a per-pixel Dice tolerates. The soft skeleton is a
# few iterations of morphological soft-erode/open via min/max pooling: cheap on
# the GPU, fully differentiable, so NOTHING needs precomputing.
#
# clDice is defined for a binary tubular mask; here it is applied one-vs-rest per
# FOREGROUND class and averaged. Use it as  Dice + CE + w * clDice. Existing files
# are untouched.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    return F.max_pool2d(img, (3, 3), stride=1, padding=(1, 1))


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img, iters=10):
    """Differentiable soft skeleton of a soft mask in [0,1], shape [B, 1, H, W]."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):

    def __init__(self, iters=10, smooth=1.0, do_bg=False):
        super().__init__()
        self.iters = iters
        self.smooth = smooth
        self.do_bg = do_bg

    def forward(self, probs, target):
        # probs: [B, C, H, W] softmax ; target: [B, H, W] (or [B, 1, H, W]) label map
        probs = probs.float()
        num_classes = probs.shape[1]
        if target.dim() == probs.dim():
            target = target[:, 0]
        target = target.long()

        start = 0 if self.do_bg else 1
        terms = []
        for c in range(start, num_classes):
            vp = probs[:, c:c + 1]                               # soft predicted mask
            vl = (target == c).float().unsqueeze(1)             # hard GT mask
            sp = soft_skeletonize(vp, self.iters)
            sl = soft_skeletonize(vl, self.iters)
            tprec = ((sp * vl).sum(dim=(1, 2, 3)) + self.smooth) / (sp.sum(dim=(1, 2, 3)) + self.smooth)
            tsens = ((sl * vp).sum(dim=(1, 2, 3)) + self.smooth) / (sl.sum(dim=(1, 2, 3)) + self.smooth)
            cl = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
            terms.append((1.0 - cl).mean())
        return torch.stack(terms).mean() if terms else probs.sum() * 0.0
