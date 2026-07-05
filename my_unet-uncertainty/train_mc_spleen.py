#
# Train the MC-Dropout U-Net on Task09_Spleen (one fold)
#
# Spleen counterpart of train_mc.py. Identical training recipe (Dice + CE, early
# stopping, best checkpoint) and the SAME MCDropoutUNet -- only the data plumbing
# differs (foreground-filtered Spleen loaders) and the defaults are tuned for a
# 256x256 abdominal-CT slice on an L4 (24 GB): larger patch, moderate batch.
#
# Reused by IMPORT (no existing file modified; nothing morphological is pulled in):
#   run_epoch / set_seed / pick_device / build_plain_loaders_spleen  <- mc_common_spleen.py
#
# GCP L4 VM:
#   DATA_DIR=$PWD/data/Task09_Spleen TASK=Task09_Spleen \
#       python train_mc_spleen.py --fold 0 --out-dir results
#

import argparse
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from mc_common_spleen import build_plain_loaders_spleen, run_epoch, set_seed, pick_device
from networks.UNET_mc import MCDropoutUNet
from loss_functions.dice_loss import SoftDiceLoss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="mcdropout")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8,
                   help="8 fits 256x256 comfortably on an L4 (bf16 autocast); raise if memory allows")
    p.add_argument("--patch-size", type=int, default=256,
                   help="working slice size; must be <= the --size used in preprocessing")
    p.add_argument("--fg-margin", type=int, default=3,
                   help="empty axial slices kept on each side of the organ (hard negatives)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="CPU workers for the (elastic) augmentation; 0 if you hit a fork/CUDA error")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    p.add_argument("--out-dir", default=os.path.join(config.PROJECT_ROOT, "results"))
    args = p.parse_args()

    if args.batch_size < 2:
        # run_epoch does target.squeeze(); a batch of 1 also drops the batch axis
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
          f"fg_margin={args.fg_margin} seed={args.seed} fold={args.fold}")

    dice_loss = SoftDiceLoss(batch_dice=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=6)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, f"{stem}_best.pth")
    last_path = os.path.join(args.out_dir, f"{stem}_last.pth")

    best_val = float("inf")
    since_improved = 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, dice_loss, ce_loss, optimizer)
        vl = run_epoch(model, val_loader, device, dice_loss, ce_loss, None)
        scheduler.step(vl)
        print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  val={vl:.4f}")
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
    print(f"[{stem}] last weights -> {last_path}")


if __name__ == "__main__":
    main()
