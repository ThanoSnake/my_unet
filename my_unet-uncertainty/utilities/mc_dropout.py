#
# MC-Dropout helpers
#
# The whole technique in three moves:
#   1. keep dropout stochastic at inference          -> enable_dropout()
#   2. run T forward passes, collect softmax probs   -> mc_forward()
#   3. reduce the T samples to uncertainty maps       -> uncertainty_maps()
#      (predictive entropy = total, mutual information = epistemic,
#       foreground-probability variance = a simple epistemic proxy)
# plus streaming calibration (Expected Calibration Error) and plotting.
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


#
# 1. switch dropout back on (and nothing else)
#
def enable_dropout(model):
    """Put ONLY the dropout layers into train() so they stay stochastic while
    the model is otherwise in eval() (InstanceNorm etc. remain frozen).
    Returns how many dropout modules were re-enabled (useful as a sanity check)."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n += 1
    return n


#
# 2. T stochastic passes
#
@torch.no_grad()
def mc_forward(model, data, T, temperature=1.0):
    """T stochastic forward passes -> softmax probabilities stacked as
    [T, B, C, H, W]. Assumes enable_dropout(model) has already been called.
    Logits are divided by `temperature` before softmax (T>1 softens): post-hoc
    temperature-scaling recalibration (Guo et al. 2017); temperature=1.0 is a no-op."""
    samples = [F.softmax(model(data) / temperature, dim=1) for _ in range(T)]
    return torch.stack(samples, dim=0)


#
# 3. reduce the samples to per-pixel uncertainty maps
#
def _entropy(prob, dim):
    """-sum_c p log p along `dim` (natural log, so units are nats)."""
    return -(prob * torch.log(prob + _EPS)).sum(dim=dim)


def uncertainty_maps(probs):
    """probs: [T, B, C, H, W] -> dict of maps.

    mean_prob        [B, C, H, W]  averaged posterior (the prediction is its argmax)
    pred             [B, H, W]     argmax of mean_prob
    entropy          [B, H, W]     predictive entropy  H(mean)               (total)
    expected_entropy [B, H, W]     E_t[H(prob_t)]                            (aleatoric)
    mutual_info      [B, H, W]     H(mean) - E_t[H(prob_t)]                  (epistemic)
    fg_var           [B, H, W]     Var_t[ P(foreground) ]                    (epistemic proxy)
    """
    mean_prob = probs.mean(dim=0)                                  # [B, C, H, W]
    pred = mean_prob.argmax(dim=1)                                 # [B, H, W]

    predictive_entropy = _entropy(mean_prob, dim=1)               # total
    expected_entropy = _entropy(probs, dim=2).mean(dim=0)         # aleatoric
    mutual_info = (predictive_entropy - expected_entropy).clamp_min(0.0)  # epistemic

    fg_prob = 1.0 - probs[:, :, 0]                                 # P(not background) [T, B, H, W]
    fg_var = fg_prob.var(dim=0, unbiased=False)                    # [B, H, W]

    return {
        "mean_prob": mean_prob,
        "pred": pred,
        "entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_info": mutual_info,
        "fg_var": fg_var,
    }


#
# recalibration: fit a single temperature (Guo et al. 2017)
#
def fit_temperature(logits, targets, max_iter=200, lr=0.1):
    """Fit a scalar temperature T>0 minimising the NLL of softmax(logits / T) on
    the given pixels (pass FOREGROUND pixels only, else background dominates and
    T stays ~1). logits: [N, C], targets: [N] long. Returns float T.
    Does not change argmax -> segmentation / Dice are unaffected."""
    log_t = torch.zeros(1, requires_grad=True)          # optimise log T so T = exp(.) > 0
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    ce = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = ce(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


#
# calibration (foreground-focused, for segmentation)
#
# Plain pixel-wise ECE is dominated by the easy, high-confidence background, so
# it looks optimistic and hides the interesting region (tumour / structure). We
# therefore restrict to the foreground and report, all streamed incrementally:
#   - foreground ECE : top-label confidence vs correctness, over the foreground
#                      ROI only (gt>0 OR pred>0)
#   - class-wise ECE : per foreground class c, one-vs-rest -- how calibrated is
#                      p_c as the probability of "pixel is c", over the SAME
#                      foreground ROI (gt>0 OR pred>0); plus their macro-average
#   - global ECE     : all pixels, kept ONLY as a reference baseline
#
class _Bins:
    """One set of reliability bins over [0, 1] (accuracy vs confidence)."""

    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.conf = np.zeros(n_bins)
        self.acc = np.zeros(n_bins)
        self.cnt = np.zeros(n_bins)

    def add(self, conf, correct):
        conf = conf.reshape(-1).cpu().numpy()
        correct = correct.reshape(-1).cpu().numpy()
        idx = np.clip(np.digitize(conf, self.edges) - 1, 0, self.n_bins - 1)
        self.cnt += np.bincount(idx, minlength=self.n_bins)
        self.conf += np.bincount(idx, weights=conf, minlength=self.n_bins)
        self.acc += np.bincount(idx, weights=correct, minlength=self.n_bins)

    def curve(self):
        tot = self.cnt.sum()
        bin_conf = np.divide(self.conf, self.cnt, out=np.zeros(self.n_bins), where=self.cnt > 0)
        bin_acc = np.divide(self.acc, self.cnt, out=np.zeros(self.n_bins), where=self.cnt > 0)
        weights = np.divide(self.cnt, tot, out=np.zeros(self.n_bins), where=tot > 0)
        ece = float((weights * np.abs(bin_acc - bin_conf)).sum())
        return bin_acc, bin_conf, self.cnt.copy(), ece


class SegCalibration:
    """Foreground-focused calibration for segmentation (see block comment)."""

    def __init__(self, num_classes, n_bins=15):
        self.num_classes = num_classes
        self.glob = _Bins(n_bins)                                     # reference (all pixels)
        self.fg = _Bins(n_bins)                                       # foreground union
        self.cls = {c: _Bins(n_bins) for c in range(1, num_classes)}  # one per fg class

    @torch.no_grad()
    def update(self, mean_prob, gt):
        conf_top, pred = mean_prob.max(dim=1)                         # [B, H, W]
        self.glob.add(conf_top, (pred == gt).float())

        roi = (gt > 0) | (pred > 0)                                   # foreground union
        if roi.any():
            # foreground: top-label confidence vs correctness
            self.fg.add(conf_top[roi], (pred[roi] == gt[roi]).float())
            # per-class: one-vs-rest over the SAME foreground ROI. Using this union
            # region (not a class-specific gt==c|pred==c mask) avoids a selection
            # bias: with a class-specific mask the low-p_c bins would contain only
            # missed gt==c pixels (event trivially 1) -> a fake accuracy=1 artifact.
            for c, bins in self.cls.items():
                bins.add(mean_prob[:, c][roi], (gt[roi] == c).float())

    def curves(self, class_names=None):
        """Ordered [(name, bin_acc, bin_conf, count, ece), ...] for plotting:
        foreground union first, then each foreground class."""
        names = class_names or {}
        out = [("foreground", *self.fg.curve())]
        for c in sorted(self.cls):
            out.append((names.get(c, f"class {c}"), *self.cls[c].curve()))
        return out

    def summary(self, class_names=None):
        names = class_names or {}
        classwise, eces = {}, []
        for c in sorted(self.cls):
            _, _, cnt, ece = self.cls[c].curve()
            has = cnt.sum() > 0
            classwise[names.get(c, f"class {c}")] = ece if has else None
            if has:
                eces.append(ece)
        return {
            "global_ece": self.glob.curve()[3],
            "foreground_ece": self.fg.curve()[3],
            "classwise_ece": classwise,
            "macro_foreground_ece": float(np.mean(eces)) if eces else None,
        }


#
# plotting (matplotlib imported lazily -> module stays import-cheap for training)
#
def save_uncertainty_png(image, gt, pred, entropy, mi, var, path, title=""):
    """5-panel figure: image | pred vs GT contours | entropy | mutual info | fg variance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 5, figsize=(20, 4.3))
    ax[0].imshow(image, cmap="gray"); ax[0].set_title("image")

    ax[1].imshow(image, cmap="gray"); ax[1].set_title("pred (red) / GT (cyan)")
    if pred.max() > 0:
        ax[1].contour(pred, levels=np.arange(0.5, pred.max() + 1), colors="red", linewidths=0.7)
    if gt.max() > 0:
        ax[1].contour(gt, levels=np.arange(0.5, gt.max() + 1), colors="cyan", linewidths=0.7)

    ax[2].imshow(entropy, cmap="magma"); ax[2].set_title("predictive entropy (total)")
    ax[3].imshow(mi, cmap="magma"); ax[3].set_title("mutual info (epistemic)")
    ax[4].imshow(var, cmap="magma"); ax[4].set_title("fg prob variance")

    for a in ax:
        a.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_calibration_figure(calib, class_names, path):
    """Reliability diagrams side by side: foreground union + one per foreground
    class, each annotated with its ECE. Bars = accuracy per confidence bin,
    dashed diagonal = perfect calibration; the gap is the miscalibration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    streams = calib.curves(class_names)
    n = len(streams)
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 4.4), squeeze=False)
    for a, (name, bin_acc, bin_conf, cnt, ece) in zip(axes[0], streams):
        edges = np.linspace(0.0, 1.0, len(bin_acc) + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        width = 0.9 / len(bin_acc)
        a.bar(centers, bin_acc, width=width, edgecolor="black", label="accuracy")
        a.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.set_title(f"{name}\nECE = {ece:.4f}  (n={int(cnt.sum())})")
        a.set_xlabel("confidence"); a.set_ylabel("accuracy")
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
