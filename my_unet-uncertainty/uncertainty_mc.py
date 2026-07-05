#
# MC-Dropout uncertainty: quantify + visualise
#
# Loads a trained checkpoint, keeps dropout ACTIVE (enable_dropout), and runs
# T stochastic forward passes per slice. From the T softmax samples it derives:
#   - predictive entropy   (total uncertainty)
#   - mutual information    (epistemic uncertainty)
#   - foreground variance   (epistemic proxy)
# and, using the mean prediction vs the ground truth, foreground-focused model
# calibration (per-class ECE + foreground ECE, background excluded).
#
# Optional recalibration: --temperature T (or the fit saved by calibrate_mc.py,
# auto-loaded) rescales the softmax as softmax(logits/T). With T != 1 the outputs
# get a '_cal' suffix so raw and recalibrated results sit side by side.
#
# Outputs (under <out-dir>/uncertainty/, <stem> = <tag>_f<fold>[ _cal ]):
#   <stem>_<case>.png            5-panel per-case uncertainty figure (repr. slice)
#   <stem>_<case>_maps.npy       [3, Z, H, W] entropy/MI/variance stacks (--save-volumes)
#   <stem>_calibration.png       reliability diagrams: foreground + per class
#   <stem>_uncertainty.json      per-case scalars + foreground/per-class ECE
#
# NOTE: this is T x slower than test_mc.py (T forward passes). T=20-30 is plenty.
#

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import config
from train_eval import pick_device
from mc_common import build_plain_loaders
from networks.UNET_mc import MCDropoutUNet
from utilities.mc_dropout import (
    enable_dropout, mc_forward, uncertainty_maps,
    SegCalibration, save_uncertainty_png, save_calibration_figure,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="mcdropout")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 = single-process loading (avoids the CUDA-in-forked-worker abort on Kaggle)")
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--mc-samples", type=int, default=30, help="number of stochastic passes T")
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    p.add_argument("--out-dir", default=os.path.join(config.PROJECT_ROOT, "results"))
    p.add_argument("--ckpt", default=None,
                   help="checkpoint path (default <out-dir>/<tag>_f<fold>_best.pth)")
    p.add_argument("--save-volumes", action="store_true",
                   help="also dump per-case entropy/MI/variance .npy stacks")
    p.add_argument("--temperature", type=float, default=None,
                   help="recalibration T for softmax(logits/T). Default: auto-load "
                        "<out-dir>/<tag>_f<fold>_temperature.json if present, else 1.0. "
                        "T != 1 writes a separate '<stem>_cal' output set.")
    args = p.parse_args()

    device = pick_device()
    _, _, test_loader, in_channels = build_plain_loaders(args)
    model = MCDropoutUNet(num_classes=args.num_classes, in_channels=in_channels,
                          dropout_p=args.dropout_p).to(device)

    stem = f"{args.tag}_f{args.fold}"
    ckpt = args.ckpt or os.path.join(args.out_dir, f"{stem}_best.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    model.eval()                         # freeze InstanceNorm ...
    n_do = enable_dropout(model)         # ... but keep dropout stochastic (the MC switch)
    if n_do == 0:
        raise SystemExit("no dropout layers found -- MC dropout would be deterministic")

    # resolve recalibration temperature: explicit arg > saved fit > 1.0 (no-op)
    temperature = args.temperature
    if temperature is None:
        tpath = os.path.join(args.out_dir, f"{stem}_temperature.json")
        if os.path.exists(tpath):
            with open(tpath) as f:
                temperature = float(json.load(f)["temperature"])
        else:
            temperature = 1.0
    recalibrated = abs(temperature - 1.0) > 1e-6
    out_stem = stem + ("_cal" if recalibrated else "")   # keep raw vs calibrated side by side

    unc_dir = os.path.join(args.out_dir, "uncertainty")
    os.makedirs(unc_dir, exist_ok=True)
    T = args.mc_samples
    print(f"[{stem}] loaded {ckpt}  T={T}  dropout layers active={n_do}  "
          f"temperature={temperature:.4f} {'(recalibrated)' if recalibrated else '(raw)'}")

    # slices are streamed in, grouped per case (volume), then restacked
    store = defaultdict(lambda: defaultdict(list))
    calib = SegCalibration(args.num_classes, n_bins=15)

    with torch.no_grad():
        for batch in test_loader:
            data = batch["data"][0].float().to(device)     # [b, c, H, W]
            target = batch["seg"][0].long().to(device)     # [b, 1, H, W]

            probs = mc_forward(model, data, T, temperature=temperature)   # [T, b, C, H, W]
            maps = uncertainty_maps(probs)

            gt = target[:, 0]                              # [b, H, W]
            calib.update(maps["mean_prob"], gt)            # foreground calibration (streaming)

            img = data[:, 0].cpu().numpy()
            gt_np = gt.cpu().numpy()
            pred_np = maps["pred"].cpu().numpy()
            entropy = maps["entropy"].cpu().numpy()
            mi = maps["mutual_info"].cpu().numpy()
            var = maps["fg_var"].cpu().numpy()

            for i, fname in enumerate(batch["fnames"]):
                key = os.path.basename(fname[0]).split(".")[0]
                s = store[key]
                s["image"].append(img[i]);   s["gt"].append(gt_np[i]);   s["pred"].append(pred_np[i])
                s["entropy"].append(entropy[i]); s["mi"].append(mi[i]);  s["var"].append(var[i])

    # per-case scalars + a representative-slice panel each
    summary = {}
    for key, s in store.items():
        entropy_vol = np.stack(s["entropy"]); mi_vol = np.stack(s["mi"]); var_vol = np.stack(s["var"])
        gt_vol = np.stack(s["gt"]); img_vol = np.stack(s["image"]); pred_vol = np.stack(s["pred"])
        fg = gt_vol > 0

        summary[key] = {
            "n_slices": int(gt_vol.shape[0]),
            "mean_entropy": float(entropy_vol.mean()),
            "mean_mutual_info": float(mi_vol.mean()),
            "mean_fg_variance": float(var_vol.mean()),
            "mean_entropy_on_gt": float(entropy_vol[fg].mean()) if fg.any() else None,
            "mean_mi_on_gt": float(mi_vol[fg].mean()) if fg.any() else None,
        }

        # representative slice: largest GT area, else the highest-entropy slice
        areas = fg.reshape(fg.shape[0], -1).sum(1)
        if areas.max() > 0:
            z = int(areas.argmax())
        else:
            z = int(entropy_vol.reshape(entropy_vol.shape[0], -1).sum(1).argmax())
        save_uncertainty_png(img_vol[z], gt_vol[z], pred_vol[z],
                             entropy_vol[z], mi_vol[z], var_vol[z],
                             os.path.join(unc_dir, f"{out_stem}_{key}.png"),
                             title=f"{key}  slice {z}  (T={temperature:.2f})")
        if args.save_volumes:
            np.save(os.path.join(unc_dir, f"{out_stem}_{key}_maps.npy"),
                    np.stack([entropy_vol, mi_vol, var_vol]).astype(np.float32))

    # dataset-level, foreground-focused calibration (per class + foreground)
    calibration = calib.summary(config.LABELS)
    save_calibration_figure(calib, config.LABELS,
                            os.path.join(unc_dir, f"{out_stem}_calibration.png"))

    out = {
        "tag": args.tag, "fold": args.fold, "mc_samples": T, "dropout_p": args.dropout_p,
        "temperature": temperature,
        "calibration": calibration,
        "mean_entropy": float(np.mean([v["mean_entropy"] for v in summary.values()])) if summary else None,
        "mean_mutual_info": float(np.mean([v["mean_mutual_info"] for v in summary.values()])) if summary else None,
        "per_case": summary,
    }
    json_path = os.path.join(unc_dir, f"{out_stem}_uncertainty.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[{out_stem}] foreground ECE={calibration['foreground_ece']:.4f}  "
          f"macro per-class ECE={calibration['macro_foreground_ece']}  "
          f"(global ref={calibration['global_ece']:.4f})  cases={len(summary)}")
    print(f"[{out_stem}] panels + calibration + {os.path.basename(json_path)} -> {unc_dir}")


if __name__ == "__main__":
    main()
