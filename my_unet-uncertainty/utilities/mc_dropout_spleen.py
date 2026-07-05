#
# Spleen calibration: per-class reliability over PREDICTED-positive pixels
#
# The shared SegCalibration (utilities/mc_dropout.py, left untouched) computes the
# per-class one-vs-rest ECE over the foreground UNION ROI (gt==c) | (pred==c). For
# a (near-)binary task that ROI drops every MISSED-organ pixel (gt==c but p_c<0.5)
# into the low-p_c bins with event==1 -> a spurious "accuracy ~ 1 at low confidence"
# artifact that inflates/distorts the per-class curve and makes it disagree with the
# top-label foreground ECE (which, for K=2, it should essentially match).
#
# Fix (as agreed): bin the per-class curve over PREDICTED-c pixels only -- criterion
# p_c > 0.5 (i.e. pred==c), NOT "gt==c OR p_c>0.5". Confidence = p_c, event = (gt==c).
# This removes the false-negative artifact, keeps the foreground focus, and for K=2
# makes the per-class panel ~identical to the foreground one. foreground + global
# curves, _Bins, curves(), summary() and the figure are all reused unchanged, so
# BOTH panels (foreground + per class) are still drawn.
#

import torch

# re-export so callers can `from utilities.mc_dropout_spleen import ...` everything
from utilities.mc_dropout import SegCalibration, save_calibration_figure, fit_temperature  # noqa: F401


# ---- temperature-scaling safety rail (Guo et al. 2017 + a sane-range guard) --
_T_LO, _T_HI = 0.5, 10.0


def fit_temperature_safe(logits, targets, t_lo=_T_LO, t_hi=_T_HI):
    """Fit a scalar temperature, then REJECT a pathological optimum.

    A single temperature can only globally sharpen (T<1) or flatten (T>1). On a
    confidently-wrong / degenerate model the NLL is minimised by T -> inf, i.e.
    softmax -> uniform (every probability 0.5, entropy = ln2, ZERO resolution),
    which destroys the uncertainty output. So if the fitted T lands outside
    [t_lo, t_hi] we fall back to T=1.0 (a no-op) and warn, keeping the raw but
    INFORMATIVE uncertainty instead of collapsing it. Returns (temperature, accepted).
    """
    t = fit_temperature(logits, targets)
    if not (t_lo <= t <= t_hi):
        print(f"[calibrate] fitted T={t:.4g} outside [{t_lo}, {t_hi}] -> model is "
              f"confidently wrong / degenerate on the fit set; falling back to T=1.0 "
              f"(raw uncertainty kept, post-hoc calibration skipped).")
        return 1.0, False
    return t, True


class SegCalibrationSpleen(SegCalibration):
    """SegCalibration whose per-class reliability uses predicted-positive pixels."""

    @torch.no_grad()
    def update(self, mean_prob, gt):
        conf_top, pred = mean_prob.max(dim=1)                     # [B, H, W]
        self.glob.add(conf_top, (pred == gt).float())            # reference: all pixels

        roi = (gt > 0) | (pred > 0)                              # foreground union (top-label)
        if roi.any():
            self.fg.add(conf_top[roi], (pred[roi] == gt[roi]).float())

        # per-class: PREDICTED-c pixels only (pred==c <=> p_c>0.5) -> p_c vs (gt==c).
        # Excludes the missed-c (gt==c, p_c<0.5) pixels that caused the low-confidence
        # artifact; no gt-conditioning of the low-p_c bins -> unbiased reliability.
        for c, bins in self.cls.items():
            m = (pred == c)
            if m.any():
                bins.add(mean_prob[:, c][m], (gt[m] == c).float())
