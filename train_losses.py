#
# Train the BASELINE U-Net with alternative segmentation LOSSES (no architecture change)
#
# Goal: improve raw segmentation (Dice + ASSD) on an imbalanced task (Task08
# HepaticVessel) by swapping ONLY the loss. The network is the untouched baseline
# networks/UNET.py (no dropout, in_channels=1). Choose the loss with --loss:
#
#   dice_ce               Dice + CE                                       (reference)
#   ftversky_ce_boundary  Focal-Tversky + CE + lam*Boundary(scheduled)    [Combo 1]
#   cldice_dice_ce        clDice + Dice + CE                              [Combo 2]
#
# Combo 1 needs the precomputed distance maps -> run run_preprocessing_losses.py so
# the npy is (2+K)-channel; the maps ride in the loader's `seg`. Combo 2 needs no
# precompute (the SAME dataset works). Everything shared (set_seed / pick_device /
# the atomic losses) is imported; nothing existing is modified. Checkpoint / scores
# use tag = the loss name by default, so test_losses.py and train_eval --fold-mean
# line up.
#
# A custom epoch loop is used because these losses need the target (and, for the
# boundary term, the per-class distance maps) that train_eval.run_epoch does not
# pass. Early stopping / LR schedule track a SCHEDULE-INDEPENDENT monitor (the
# region+CE part) so a growing boundary lam cannot fake "improvement".
#
# Kaggle:
#   DATA_DIR=/kaggle/input/<prep>/Task08_HepaticVessel TASK=Task08_HepaticVessel \
#       python train_losses.py --loss ftversky_ce_boundary --fold 0 --num-workers 0
#

import argparse
import contextlib
import glob
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from train_eval import set_seed, pick_device
from networks.UNET import UNet
from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.tversky_loss import FocalTverskyLoss
from loss_functions.boundary_loss import BoundaryLoss
from loss_functions.cldice_loss import SoftClDiceLoss


#
# loaders: n_dist>0 also fetches the n_dist SDF channels into `seg` (tuple label_slice)
#
def build_loaders(args, n_dist):
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)
    tr = splits[args.fold]["train"]
    vl = splits[args.fold]["val"]
    ts = splits[args.fold]["test"]

    data_dir = str(config.PREPROCESSED_DIR)
    label_slice = 1 if n_dist == 0 else tuple(range(1, 2 + n_dist))     # 1 = label; 2.. = phi
    common = dict(target_size=args.patch_size, batch_size=args.batch_size,
                  input_slice=(0,), label_slice=label_slice, num_processes=args.num_workers)

    train = NumpyDataSet(data_dir, keys=tr, **common)
    val = NumpyDataSet(data_dir, keys=vl, mode="val", do_reshuffle=False, **common)
    test = NumpyDataSet(data_dir, keys=ts, mode="test", do_reshuffle=False, **common)
    return train, val, test


def _assert_dist_channels(args, n_dist):
    """Fail early (clear message) if the boundary combo is run on a plain npy."""
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)
    keys = splits[args.fold]["train"]
    matches = glob.glob(os.path.join(str(config.PREPROCESSED_DIR), keys[0] + ".npy")) if keys else []
    if not matches:
        return                                       # let the loader raise its own error
    ch = np.load(matches[0], mmap_mode="r").shape[0]
    need = 2 + n_dist
    if ch < need:
        raise SystemExit(
            f"--loss ftversky_ce_boundary needs a {need}-channel npy "
            f"(image, label, {n_dist} distance map(s)) but {os.path.basename(matches[0])} "
            f"has {ch}. Re-run run_preprocessing_losses.py.")


#
# loss combos -- each returns (total_to_optimise, monitor_for_early_stopping)
#
class DiceCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = SoftDiceLoss(batch_dice=True)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, probs, y, phi, lam):
        base = self.dice(probs, y) + self.ce(logits, y)
        return base, base


class FTvCEBoundary(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.ftv = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, do_bg=False, batch=True)
        self.ce = nn.CrossEntropyLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits, probs, y, phi, lam):
        region = self.ftv(probs, y) + self.ce(logits, y)
        total = region + lam * self.boundary(probs, phi)
        return total, region                          # monitor = schedule-independent region+CE


class ClDiceDiceCE(nn.Module):
    def __init__(self, weight, iters):
        super().__init__()
        self.dice = SoftDiceLoss(batch_dice=True)
        self.ce = nn.CrossEntropyLoss()
        self.cldice = SoftClDiceLoss(iters=iters, do_bg=False)
        self.weight = weight

    def forward(self, logits, probs, y, phi, lam):
        base = self.dice(probs, y) + self.ce(logits, y)
        total = base + self.weight * self.cldice(probs, y)
        return total, base                            # monitor = Dice+CE (comparable across combos)


def make_loss(args):
    if args.loss == "dice_ce":
        return DiceCE()
    if args.loss == "ftversky_ce_boundary":
        return FTvCEBoundary(args.tversky_alpha, args.tversky_beta, args.focal_gamma)
    return ClDiceDiceCE(args.cldice_weight, args.cldice_iters)


def boundary_lambda(epoch, args):
    """Ramp lam 0 -> boundary_max over the first `boundary_warmup` epochs, then hold."""
    if args.boundary_warmup <= 0:
        return args.boundary_max
    return args.boundary_max * min(1.0, epoch / float(args.boundary_warmup))


#
# one epoch -> (mean total loss, mean monitor loss)
#
def run_epoch(model, loader, device, loss_fn, optimizer=None, lam=0.0):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    totals, monitors = [], []
    amp = device.type == "cuda"
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            data = batch["data"][0].float().to(device, non_blocking=True)   # [b, 1, H, W]
            seg = batch["seg"][0].to(device, non_blocking=True)             # [b, S, H, W]
            y = seg[:, 0].long()                                            # [b, H, W]
            phi = seg[:, 1:].float() if seg.shape[1] > 1 else None          # [b, K, H, W] or None
            if train_mode:
                optimizer.zero_grad()
            with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
                logits = model(data)
                probs = F.softmax(logits, dim=1)
                total, monitor = loss_fn(logits, probs, y, phi, lam)
            if train_mode:
                total.backward()
                optimizer.step()
            totals.append(float(total.detach().item()))
            monitors.append(float(monitor.detach().item()))
    return (float(np.mean(totals)) if totals else float("nan"),
            float(np.mean(monitors)) if monitors else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loss", required=True,
                   choices=["dice_ce", "ftversky_ce_boundary", "cldice_dice_ce"])
    p.add_argument("--tag", default=None, help="checkpoint/scores tag (default = the --loss name)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 = single-process loading (avoids the CUDA-in-forked-worker abort on Kaggle)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    # Focal-Tversky (Combo 1)
    p.add_argument("--tversky-alpha", type=float, default=0.3, help="FP weight (< beta favours recall)")
    p.add_argument("--tversky-beta", type=float, default=0.7, help="FN weight (> alpha favours recall)")
    p.add_argument("--focal-gamma", type=float, default=1.3333333, help="focus; exponent = 1/gamma")
    # Boundary schedule (Combo 1)
    p.add_argument("--boundary-max", type=float, default=0.5, help="max lam for the boundary term")
    p.add_argument("--boundary-warmup", type=int, default=40,
                   help="epochs to ramp lam 0 -> boundary-max (0 = constant lam)")
    # clDice (Combo 2)
    p.add_argument("--cldice-weight", type=float, default=0.5)
    p.add_argument("--cldice-iters", type=int, default=10)
    p.add_argument("--out-dir", default=os.path.join(config.PROJECT_ROOT, "results"))
    args = p.parse_args()

    if args.batch_size < 2:
        raise SystemExit("use --batch-size >= 2")

    tag = args.tag or args.loss
    set_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    needs_boundary = args.loss == "ftversky_ce_boundary"
    n_dist = (args.num_classes - 1) if needs_boundary else 0
    if needs_boundary:
        _assert_dist_channels(args, n_dist)

    train_loader, val_loader, _ = build_loaders(args, n_dist)
    model = UNet(num_classes=args.num_classes, in_channels=1).to(device)
    loss_fn = make_loss(args).to(device)

    stem = f"{tag}_f{args.fold}"
    desc = f"loss={args.loss} classes={args.num_classes} in_ch=1 fold={args.fold}"
    if needs_boundary:
        desc += (f" ftv(a={args.tversky_alpha},b={args.tversky_beta},g={args.focal_gamma})"
                 f" boundary(max={args.boundary_max},warmup={args.boundary_warmup})")
    elif args.loss == "cldice_dice_ce":
        desc += f" cldice(w={args.cldice_weight},iters={args.cldice_iters})"
    print(f"[{stem}] device={device} {desc}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=6)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, f"{stem}_best.pth")
    last_path = os.path.join(args.out_dir, f"{stem}_last.pth")

    best_val = float("inf")
    since_improved = 0
    for epoch in range(1, args.epochs + 1):
        lam = boundary_lambda(epoch, args) if needs_boundary else 0.0
        tr_tot, tr_mon = run_epoch(model, train_loader, device, loss_fn, optimizer, lam)
        vl_tot, vl_mon = run_epoch(model, val_loader, device, loss_fn, None, lam)
        scheduler.step(vl_mon)                        # step on the schedule-independent monitor
        extra = f"  lam={lam:.3f}" if needs_boundary else ""
        print(f"epoch {epoch:3d}/{args.epochs}  train={tr_tot:.4f}(mon={tr_mon:.4f})  "
              f"val={vl_tot:.4f}(mon={vl_mon:.4f}){extra}")
        if vl_mon < best_val:
            best_val = vl_mon
            since_improved = 0
            torch.save(model.state_dict(), best_path)
        else:
            since_improved += 1
            patience_str = f"/{args.patience}" if args.patience else ""
            print(f"  val monitor did not improve from {best_val:.4f} (patience {since_improved}{patience_str})")
        torch.save(model.state_dict(), last_path)
        if args.patience and since_improved >= args.patience:
            print(f"early stop @ epoch {epoch} (best val monitor={best_val:.4f} "
                  f"@ ep {epoch - since_improved})")
            break

    print(f"[{stem}] best weights -> {best_path}")


if __name__ == "__main__":
    main()
