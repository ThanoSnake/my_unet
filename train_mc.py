#
# Train the MC-Dropout U-Net (one fold)
#
# Standard supervised training with early stopping. Dropout is active as usual
# during training; the point of the MC-Dropout model only shows up later, at
# uncertainty time (uncertainty_mc.py). The best checkpoint (lowest val loss) is
# saved so test/uncertainty can reuse the weights without re-training.
#
# Reuses the baseline plumbing by IMPORT (no existing file is modified):
#   run_epoch / set_seed / pick_device        <- train_eval.py
#   build_plain_loaders (2-channel image+label) <- mc_common.py
#
# Kaggle: the preprocessed data + splits are located via config, which reads the
# DATA_DIR / TASK env vars. Point DATA_DIR at your (read-only) input dataset and
# keep --out-dir on a writable path, e.g.
#   !DATA_DIR=/kaggle/input/msd-prep/Task08_HepaticVessel TASK=Task08_HepaticVessel \
#        python train_mc.py --fold 0 --out-dir /kaggle/working/results
#

import argparse
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from train_eval import run_epoch, set_seed, pick_device
from mc_common import build_plain_loaders
from networks.UNET_mc import MCDropoutUNet
from loss_functions.dice_loss import SoftDiceLoss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="mcdropout")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 = single-process loading; avoids the CUDA-in-forked-worker "
                        "abort on Kaggle. Small 2D slices load fast enough with 0.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    p.add_argument("--out-dir", default=os.path.join(config.PROJECT_ROOT, "results"))
    args = p.parse_args()

    if args.batch_size < 2:
        # run_epoch does target.squeeze(); with a batch of 1 that also drops the
        # batch axis and the loss shapes break.
        raise SystemExit("use --batch-size >= 2")

    set_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, in_channels = build_plain_loaders(args)
    model = MCDropoutUNet(num_classes=args.num_classes, in_channels=in_channels,
                          dropout_p=args.dropout_p).to(device)

    stem = f"{args.tag}_f{args.fold}"
    print(f"[{stem}] device={device} dropout_p={args.dropout_p} in_ch={in_channels} "
          f"classes={args.num_classes} seed={args.seed} fold={args.fold}")

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
