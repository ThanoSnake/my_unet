#
# Fit a recalibration temperature (Guo et al. 2017) on the Spleen VAL fold
#
# Spleen counterpart of calibrate_mc.py. Loads a trained checkpoint, runs
# DETERMINISTIC logits (dropout OFF) on the validation fold, and fits a single
# scalar temperature T that minimises the NLL over the FOREGROUND pixels
# (gt>0 OR pred>0). Background is excluded on purpose: on an abdominal CT it
# utterly dominates the pixel count (spleen is a small fraction) and is already
# well calibrated, so fitting on it would leave T~1 and fix nothing.
#
# T does NOT change argmax -> Dice / segmentation unaffected; only the softmax
# confidence is rescaled. Saves <out-dir>/<tag>_f<fold>_temperature.json, which
# uncertainty_mc_spleen.py auto-loads.
#

import argparse
import json
import os

import torch

import config
from mc_common_spleen import build_plain_loaders_spleen, pick_device
from networks.UNET_mc import MCDropoutUNet
from utilities.mc_dropout import fit_temperature


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="mcdropout")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    p.add_argument("--out-dir", default=os.path.join(config.PROJECT_ROOT, "results"))
    p.add_argument("--ckpt", default=None,
                   help="checkpoint path (default <out-dir>/<tag>_f<fold>_best.pth)")
    args = p.parse_args()

    device = pick_device()
    _, val_loader, _, in_channels = build_plain_loaders_spleen(args)   # fit on VAL, never test
    model = MCDropoutUNet(num_classes=args.num_classes, in_channels=in_channels,
                          dropout_p=args.dropout_p).to(device)

    stem = f"{args.tag}_f{args.fold}"
    ckpt = args.ckpt or os.path.join(args.out_dir, f"{stem}_best.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()   # dropout OFF -> deterministic logits for a stable T fit

    # collect foreground logits + labels over the validation fold
    logits_fg, targets_fg = [], []
    with torch.no_grad():
        for batch in val_loader:
            data = batch["data"][0].float().to(device)
            gt = batch["seg"][0].long().to(device)[:, 0]        # [b, H, W]
            logits = model(data)                                # [b, C, H, W]
            pred = logits.argmax(dim=1)
            mask = (gt > 0) | (pred > 0)                        # foreground ROI
            if mask.any():
                logits_fg.append(logits.permute(0, 2, 3, 1)[mask].cpu())  # [N, C]
                targets_fg.append(gt[mask].cpu())               # [N]

    if not logits_fg:
        raise SystemExit("no foreground pixels found in the val fold")
    logits_fg = torch.cat(logits_fg).float()
    targets_fg = torch.cat(targets_fg).long()

    temperature = fit_temperature(logits_fg, targets_fg)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "tag": args.tag, "fold": args.fold,
        "temperature": temperature,
        "n_foreground_pixels": int(targets_fg.numel()),
        "fit_on": "val foreground (gt>0 | pred>0), deterministic logits",
    }
    path = os.path.join(args.out_dir, f"{stem}_temperature.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[{stem}] fitted temperature T={temperature:.4f} "
          f"on {targets_fg.numel()} foreground pixels -> {path}")
    if temperature > 1:
        print(f"[{stem}] T>1 -> model was over-confident; softmax will be softened")
    elif temperature < 1:
        print(f"[{stem}] T<1 -> model was under-confident; softmax will be sharpened")


if __name__ == "__main__":
    main()
