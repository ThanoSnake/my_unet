#
# Train the BASELINE U-Net with alternative segmentation LOSSES (no architecture change).
#
# Goal: improve raw segmentation (Dice + ASSD) on Task08 HepaticVessel (severe imbalance,
# thin vessels + small tumours) by swapping ONLY the loss. The network is the untouched
# baseline networks/UNET.py (no dropout, in_channels=1). The DATA pipeline is the correct
# thanasis-style one (native-resolution patches + class-balanced foreground oversampling +
# full-slice validation), provided by pipeline_loss.py / NumpyDataLoader_loss.py -- the
# existing loading/preprocessing files are left untouched.
#
# Loss (--loss):
#   dice_ce               Dice + CE                                       (reference)
#   ftversky_ce_boundary  Focal-Tversky + CE + lam*Boundary(scheduled)    [Combo 1]
#   cldice_dice_ce        clDice + Dice + CE                              [Combo 2]
# All Dice/Tversky terms are FOREGROUND-only (do_bg=False): the >99% background otherwise
# dilutes the gradient on the sparse target.
#
# Model selection = MAX full-slice foreground val-Dice (nnU-Net pseudo-Dice), computed every
# --val-every epochs. This is loss-agnostic, so the three loss combos are compared fairly; the
# auxiliary boundary/clDice terms shape training but never the selection criterion. Combo 1 needs
# the precomputed distance maps -> run run_preprocessing_losses.py (the maps ride in the TRAIN
# loader's `seg`); Combo 2 needs no precompute.
#
# Kaggle:
#   DATA_DIR=/kaggle/input/<prep>/Task08_HepaticVessel TASK=Task08_HepaticVessel \
#       python train_losses.py --loss ftversky_ce_boundary --fold 0
#

import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from pipeline_loss import set_seed, pick_device, build_loaders, run_epoch, run_val_dice
from networks.UNET import UNet
from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.tversky_loss import FocalTverskyLoss
from loss_functions.boundary_loss import BoundaryLoss
from loss_functions.cldice_loss import SoftClDiceLoss


#
# loss combos -- forward(logits, probs, y, phi, lam) -> scalar total loss
#
class DiceCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = SoftDiceLoss(batch_dice=True, do_bg=False)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, probs, y, phi, lam):
        return self.dice(probs, y) + self.ce(logits, y)


class FTvCEBoundary(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.ftv = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, do_bg=False, batch=True)
        self.ce = nn.CrossEntropyLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits, probs, y, phi, lam):
        return self.ftv(probs, y) + self.ce(logits, y) + lam * self.boundary(probs, phi)


class ClDiceDiceCE(nn.Module):
    def __init__(self, weight, iters):
        super().__init__()
        self.dice = SoftDiceLoss(batch_dice=True, do_bg=False)
        self.ce = nn.CrossEntropyLoss()
        self.cldice = SoftClDiceLoss(iters=iters, do_bg=False)
        self.weight = weight

    def forward(self, logits, probs, y, phi, lam):
        return self.dice(probs, y) + self.ce(logits, y) + self.weight * self.cldice(probs, y)


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


def _assert_dist_channels(args, n_dist):
    """Fail early (clear message) if the boundary combo is run on a plain (no-phi) npy."""
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)
    keys = splits[args.fold]["train"]
    matches = glob.glob(os.path.join(str(config.PREPROCESSED_DIR), keys[0] + ".npy")) if keys else []
    if not matches:
        return
    ch = np.load(matches[0], mmap_mode="r").shape[0]
    need = 2 + n_dist
    if ch < need:
        raise SystemExit(
            f"--loss ftversky_ce_boundary needs a {need}-channel npy (image, label, {n_dist} "
            f"distance map(s)) but {os.path.basename(matches[0])} has {ch}. "
            f"Re-run run_preprocessing_losses.py.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loss", required=True,
                   choices=["dice_ce", "ftversky_ce_boundary", "cldice_dice_ce"])
    p.add_argument("--tag", default=None, help="checkpoint/scores tag (default = the --loss name)")
    p.add_argument("--fold", type=int, default=0)
    # training budget (HepaticVessel-tuned defaults; epoch = iters_per_epoch batches, NOT a full pass)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=30, help="stop after this many EPOCHS with no val-Dice gain")
    p.add_argument("--iters-per-epoch", type=int, default=250, help="batches per epoch; 0 = full pass over all slices")
    p.add_argument("--val-every", type=int, default=3, help="run full-slice val-Dice every K epochs")
    p.add_argument("--val-cases", type=int, default=15, help="validate on this many fixed seeded volumes (0 = all)")
    p.add_argument("--val-batch", type=int, default=12, help="full slices per validation forward pass")
    p.add_argument("--fg-fraction", type=float, default=0.33, help="fraction of train crops centred on foreground")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8, help="raise if the GPU allows (thanasis used 24 @ patch 128)")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 = single-process (safe on Kaggle). Raise to 4-6 for much faster native-res loading; "
                        "if you hit 'Cannot re-initialize CUDA in forked subprocess', drop back to 0.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    # Focal-Tversky (Combo 1)
    p.add_argument("--tversky-alpha", type=float, default=0.3, help="FP weight (< beta favours recall)")
    p.add_argument("--tversky-beta", type=float, default=0.7, help="FN weight (> alpha favours recall)")
    p.add_argument("--focal-gamma", type=float, default=1.3333333, help="focus; exponent = 1/gamma")
    # Boundary schedule (Combo 1)
    p.add_argument("--boundary-max", type=float, default=0.5, help="max lam for the boundary term")
    p.add_argument("--boundary-warmup", type=int, default=40, help="epochs to ramp lam 0 -> boundary-max (0 = constant)")
    # clDice (Combo 2)
    p.add_argument("--cldice-weight", type=float, default=0.5)
    p.add_argument("--cldice-iters", type=int, default=10)
    p.add_argument("--out-dir", default=os.path.join(config.PROJECT_ROOT, "results"))
    p.add_argument("--resume", action="store_true", help="continue from <tag>_f<fold>_last.pth if present")
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

    train_loader, val_loader, _, _ = build_loaders(args, n_dist=n_dist, want_train=True)
    model = UNet(num_classes=args.num_classes, in_channels=1).to(device)
    loss_fn = make_loss(args).to(device)

    stem = f"{tag}_f{args.fold}"
    epoch_len = args.iters_per_epoch if args.iters_per_epoch > 0 else len(train_loader)
    desc = f"loss={args.loss} classes={args.num_classes} patch={args.patch_size} bs={args.batch_size} " \
           f"iters/epoch={epoch_len} fg={args.fg_fraction} val_every={args.val_every}"
    if needs_boundary:
        desc += f" ftv(a={args.tversky_alpha},b={args.tversky_beta},g={args.focal_gamma})" \
                f" boundary(max={args.boundary_max},warmup={args.boundary_warmup})"
    elif args.loss == "cldice_dice_ce":
        desc += f" cldice(w={args.cldice_weight},iters={args.cldice_iters})"
    print(f"[{stem}] device={device} {desc}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=6)   # fed -val_dice (maximise Dice)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, f"{stem}_best.pth")
    last_path = os.path.join(args.out_dir, f"{stem}_last.pth")

    start_epoch, best_dice, best_epoch = 1, -1.0, 0
    t0, val_curve = time.time(), []
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        if isinstance(ck, dict) and "model" in ck:
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"]); start_epoch = ck["epoch"] + 1
            best_dice, best_epoch = ck["best_val"], ck.get("best_epoch", 0)
            print(f"resumed from epoch {ck['epoch']} (best val-Dice={best_dice:.4f} @ ep {best_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        lam = boundary_lambda(epoch, args) if needs_boundary else 0.0
        tr = run_epoch(model, train_loader, device, loss_fn, optimizer, lam=lam)
        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val:
            vd, per_class = run_val_dice(model, val_loader, device, args.num_classes)
            scheduler.step(-vd)                        # scheduler minimises; we maximise Dice
            lam_str = f"  lam={lam:.3f}" if needs_boundary else ""
            print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  val_fgDice={vd:.4f}  "
                  f"[per-class {' '.join(f'{d:.3f}' for d in per_class)}]{lam_str}")
            val_curve.append([epoch, round(vd, 5)])
            if vd > best_dice:
                best_dice, best_epoch = vd, epoch
                torch.save(model.state_dict(), best_path)     # best: weights only (for eval)
            else:
                patience_str = f"/{args.patience}" if args.patience else ""
                print(f"  val fg-Dice did not improve from {best_dice:.4f} @ ep {best_epoch} "
                      f"(stale {epoch - best_epoch}{patience_str})")
        else:
            print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  (val every {args.val_every})")
        # last: full state so training can resume after an interruption (Kaggle spot / timeout)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_val": best_dice,
                    "best_epoch": best_epoch}, last_path)
        if args.patience and (epoch - best_epoch) >= args.patience:
            print(f"early stop: no val fg-Dice improvement for {epoch - best_epoch} epochs "
                  f"(best={best_dice:.4f} @ ep {best_epoch})")
            break

    elapsed = time.time() - t0
    thr = 0.9 * best_dice
    ep_to_thr = next((e for e, d in val_curve if d >= thr), best_epoch)
    summary = {"tag": tag, "fold": args.fold, "loss": args.loss, "best_fg_dice": round(best_dice, 5),
               "best_epoch": best_epoch, "epochs_to_90pct_best": ep_to_thr, "stopped_epoch": epoch,
               "max_epochs": args.epochs, "seconds": round(elapsed, 1),
               "sec_per_epoch": round(elapsed / max(epoch - start_epoch + 1, 1), 2),
               "iters_per_epoch": epoch_len, "val_curve": val_curve}
    with open(os.path.join(args.out_dir, f"{stem}_train.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{stem}] best val fg-Dice {best_dice:.4f} @ ep{best_epoch} (90% @ ep{ep_to_thr}) "
          f"in {elapsed / 60:.1f} min -> {best_path}")


if __name__ == "__main__":
    main()
