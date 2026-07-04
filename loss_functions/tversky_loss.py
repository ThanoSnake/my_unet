#
# Tversky & Focal-Tversky losses (region-based, imbalance-aware)
#
# Tversky index generalises Dice with SEPARATE control of false positives and
# false negatives:  TI_c = TP / (TP + alpha*FP + beta*FN). With beta > alpha the
# loss penalises MISSES (FN) harder than false alarms -> boosts the recall of thin
# / small foreground structures (hepatic vessels, small tumours). Focal-Tversky
# (Abraham & Khan, 2019) additionally focuses on the hard, low-overlap classes via
# the exponent 1/gamma on (1 - TI): with gamma = 4/3 the exponent is 3/4 < 1, which
# inflates the loss of the classes that are still poorly segmented.
#
# Reuses the tp/fp/fn machinery from dice_loss so the numerics match the existing
# Dice. do_bg=False (default) averages over foreground classes only -- the right
# choice under heavy background imbalance. Existing files are untouched.
#

import torch
import torch.nn as nn

from loss_functions.dice_loss import get_tp_fp_fn


class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, do_bg=False, batch=True):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.smooth = smooth
        self.do_bg = do_bg
        self.batch = batch                      # aggregate tp/fp/fn over the batch (like batch_dice)

    def _index(self, probs, target):
        """Per-class Tversky index TI_c (foreground only unless do_bg)."""
        axes = (0, 2, 3) if self.batch else (2, 3)
        tp, fp, fn = get_tp_fp_fn(probs, target, axes=axes)
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if not self.do_bg:
            ti = ti[1:] if ti.dim() == 1 else ti[:, 1:]
        return ti

    def forward(self, probs, target):
        return (1.0 - self._index(probs, target)).mean()


class FocalTverskyLoss(TverskyLoss):

    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3333333, smooth=1.0, do_bg=False, batch=True):
        super().__init__(alpha, beta, smooth, do_bg, batch)
        self.gamma = gamma                      # exponent = 1/gamma (0.75 for 4/3): <1 focuses on hard classes

    def forward(self, probs, target):
        ti = self._index(probs, target)
        return ((1.0 - ti).clamp_min(0.0) ** (1.0 / self.gamma)).mean()
