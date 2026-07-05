#
# Train the Spleen MC-Dropout U-Net with TRAIN-TIME calibration (Soft-Binned ECE)
#
# Spleen counterpart of train_mc_recalibrate.py. Loss is Dice + CE + lambda*SB-ECE
# (a differentiable ECE surrogate over the foreground ROI) so the net states
# calibrated confidence natively, without relying on a post-hoc temperature (you
# can still run calibrate_mc_spleen.py on top; they are complementary). Default
# tag 'mcdropout_cal' so the test / uncertainty / calibrate scripts pick it up via
#   --tag mcdropout_cal
#
# The custom epoch loop is needed because the SB-ECE term needs the target, which
# train_eval.run_epoch's dice+ce signature does not pass. Everything else is
# reused by import (set_seed / pick_device / build_plain_loaders_spleen).
#
# GCP L4 VM:
#   DATA_DIR=$PWD/data/Task09_Spleen TASK=Task09_Spleen \
#       python train_mc_recalibrate_spleen.py --fold 0 --out-dir results
#

import argparse
import contextlib
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from mc_common_spleen import build_plain_loaders_spleen, set_seed, pick_device
from networks.UNET_mc import MCDropoutUNet
from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.calibration_loss import SoftBinnedECELoss


#
# custom epoch: Dice + CE + lambda * SB-ECE  (returns mean total loss + mean SB-ECE)
#
def run_epoch(model, loader, device, dice_loss, ce_loss, cal_loss, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    losses, cals = [], []
    amp = device.type == "cuda"
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            data = batch["data"][0].float().to(device, non_blocking=True)   # [b, c, H, W]
            target = batch["seg"][0].long().to(device, non_blocking=True)   # [b, 1, H, W]
            if train_mode:
                optimizer.zero_grad()
            with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
                logits = model(data)
                probs = F.softmax(logits, dim=1)
                y = target.squeeze(1)                                       # [b, H, W] (safe for b=1)
                loss = dice_loss(probs, y) + ce_loss(logits, y)
                cal = cal_loss(logits, target)                             # SB-ECE (foreground)
                loss = loss + cal
            if train_mode:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            cals.append(float(cal.detach().item()))
    return (float(np.mean(losses)) if losses else float("nan"),
            float(np.mean(cals)) if cals else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="mcdropout_cal")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--fg-margin", type=int, default=3,
                   help="empty axial slices kept on each side of the organ (hard negatives)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="CPU workers for augmentation; 0 if you hit a fork/CUDA error")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--cal-weight", type=float, default=1.0,
                   help="lambda for the SB-ECE term (Dice+CE+lambda*SB-ECE); try 1, 5, 10")
    p.add_argument("--cal-bins", type=int, default=15)
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    p.add_argument("--out-dir", default=os.path.join(config.PROJECT_ROOT, "results"))
    args = p.parse_args()

    if args.batch_size < 2:
        raise SystemExit("use --batch-size >= 2")

    set_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, in_channels = build_plain_loaders_spleen(args)
    model = MCDropoutUNet(num_classes=args.num_classes, in_channels=in_channels,
                          dropout_p=args.dropout_p).to(device)

    stem = f"{args.tag}_f{args.fold}"
    print(f"[{stem}] device={device} dropout_p={args.dropout_p} in_ch={in_channels} "
          f"classes={args.num_classes} patch={args.patch_size} bs={args.batch_size} "
          f"loss=dice+ce+{args.cal_weight}*SB-ECE(bins={args.cal_bins})")

    dice_loss = SoftDiceLoss(batch_dice=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    cal_loss = SoftBinnedECELoss(n_bins=args.cal_bins, weight=args.cal_weight,
                                 foreground_only=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=6)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, f"{stem}_best.pth")
    last_path = os.path.join(args.out_dir, f"{stem}_last.pth")

    best_val = float("inf")
    since_improved = 0
    for epoch in range(1, args.epochs + 1):
        tr, tr_cal = run_epoch(model, train_loader, device, dice_loss, ce_loss, cal_loss, optimizer)
        vl, vl_cal = run_epoch(model, val_loader, device, dice_loss, ce_loss, cal_loss, None)
        scheduler.step(vl)
        print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f} (sbece={tr_cal:.4f})  "
              f"val={vl:.4f} (sbece={vl_cal:.4f})")
        if vl < best_val:
            best_val = vl
            since_improved = 0
            torch.save(model.state_dict(), best_path)
        else:
            since_improved += 1
            patience_str = f"/{args.patience}" if args.patience else ""
            print(f"  val did not improve from {best_val:.4f} (patience {since_improved}{patience_str})")
        torch.save(model.state_dict(), last_path)
        if args.patience and since_improved >= args.patience:
            print(f"early stop @ epoch {epoch} (best val={best_val:.4f} @ ep {epoch - since_improved})")
            break

    print(f"[{stem}] best weights -> {best_path}")


if __name__ == "__main__":
    main()
