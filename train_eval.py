#
# Train and Test for U-Net Segmentation
#

import argparse
import contextlib
import glob
import json
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from networks.UNET import UNet
from networks.morph_block import MorphResidualUNet
from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.morph_loss import MorphConsistencyLoss
from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from evaluation.evaluator import aggregate_scores, Evaluator

LABELS = config.LABELS   # {class_id: name}, read from the task's dataset.json

#
# fixed seed so runs share initialisation
#
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

#
# build loaders
#
def build_loaders(args):
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)
    tr, vl, ts = (splits[args.fold]["train"], splits[args.fold]["val"], splits[args.fold]["test"])
    data_dir = str(config.PREPROCESSED_DIR)
    # npy is 4-channel (image, tophat, bottomhat, label)
    if args.morph_block:
        input_slice = (0,)
    else:
        input_slice = (0,)
        if args.tophat:
            input_slice += (1,)
        if args.bottomhat:
            input_slice += (2,)
    label_slice = 3
    common = dict(target_size=args.patch_size, batch_size=args.batch_size, input_slice=input_slice,
                  label_slice=label_slice, num_processes=args.num_workers)
    train = NumpyDataSet(data_dir, keys=tr, **common)
    val = NumpyDataSet(data_dir, keys=vl, mode="val",  do_reshuffle=False, **common)
    test = NumpyDataSet(data_dir, keys=ts, mode="test", do_reshuffle=False, **common)
    in_channels = len(input_slice)
    return train, val, test, in_channels

#
# run epoch
#
def run_epoch(model, loader, device, dice_loss, ce_loss, optimizer=None, morph_loss=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    losses = []
    amp = device.type == "cuda"   # bf16 autocast on the L4 tensor cores (no-op off CUDA)
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            data = batch["data"][0].float().to(device, non_blocking=True)   # [b, c, H, W]
            target = batch["seg"][0].long().to(device, non_blocking=True)   # [b, 1, H, W]
            if train_mode:
                optimizer.zero_grad()
            with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
                pred = model(data)
                pred_softmax = F.softmax(pred, dim=1)
                loss = dice_loss(pred_softmax, target.squeeze()) + ce_loss(pred, target.squeeze())
                if morph_loss is not None:
                    loss = loss + morph_loss(pred_softmax)
            if train_mode:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")

#
# test
#
def evaluate_test(model, loader, device, json_path):
    model.eval()
    pred_dict, gt_dict = defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            data = batch["data"][0].float().to(device)
            target = batch["seg"][0].float().to(device)
            pred = model(data)
            pred_argmax = torch.argmax(pred.data.cpu(), dim=1, keepdim=True)
            for i, fname in enumerate(batch["fnames"]):
                pred_dict[fname[0]].append(pred_argmax[i].numpy())
                gt_dict[fname[0]].append(target[i].detach().cpu().numpy())
    pairs = [(np.stack(pred_dict[k]), np.stack(gt_dict[k])) for k in pred_dict]
    scores = aggregate_scores(pairs, evaluator=Evaluator, labels=LABELS,
        json_output_file=json_path, json_author="cv-project",
        json_task=config.TASK, advanced=True,
    )
    return scores


def _run_name(path):
    base = os.path.basename(path)
    for suf in ("_scores.json", ".json"):
        if base.endswith(suf):
            return base[:-len(suf)]
    return base

#
# average each metric across all per-fold <tag>_f<fold>_scores.json
#
def fold_mean(tag):
    results_dir = os.path.join(config.PROJECT_ROOT, "results")
    paths = sorted(glob.glob(os.path.join(results_dir, f"{tag}_f*_scores.json")))
    if not paths:
        raise SystemExit(f"no files match {tag}_f*_scores.json in {results_dir}")
    per_fold = []
    for p in paths:
        with open(p) as f:
            per_fold.append(json.load(f)["results"]["mean"])
    print(f"{tag}: mean +/- std over {len(paths)} folds "
          f"({', '.join(_run_name(p) for p in paths)})")
    agg = {}
    for label in per_fold[0]:
        agg[label] = {}
        for metric in ("Dice", "Avg. Symmetric Surface Distance"):
            vals = [fold[label].get(metric) for fold in per_fold]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            m, s = float(np.mean(vals)), float(np.std(vals))
            agg[label][metric] = {"mean": m, "std": s, "n": len(vals)}
            print(f"  label {label:>10} | {metric:<32} "
                  f"{m:.4f} +/- {s:.4f}  (n={len(vals)})")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"{tag}_mean_scores.json")
    with open(out, "w") as f:
        json.dump({"tag": tag, "folds": paths, "mean": agg}, f, indent=2)
    print(f"written to {out}")
    return agg

#
# per-label Dice/ASSD deltas of each run vs the baseline
#
def compare_runs(paths):
    if len(paths) < 2:
        raise SystemExit("--compare needs at least two JSON files")
    baseline, others = paths[0], paths[1:]

    def load_mean(p):
        """Flat {label: {metric: float}}, accepting per-fold or fold-mean files."""
        actual_path = p
        if not os.path.exists(actual_path):
            results_dir = os.path.join(config.PROJECT_ROOT, "results")
            alt_path = os.path.join(results_dir, p)
            if os.path.exists(alt_path):
                actual_path = alt_path
        with open(actual_path) as f:
            d = json.load(f)
        kind = "fold-mean" if "mean" in d else "per-fold"
        raw = d["mean"] if kind == "fold-mean" else d["results"]["mean"]
        # fold-mean files nest {"mean", "std", "n"}; flatten to the mean
        flat = {label: {m: (v["mean"] if isinstance(v, dict) else v) for m, v in metrics.items()}
                for label, metrics in raw.items()}
        return kind, flat

    kinds = {p: load_mean(p)[0] for p in paths}
    if len(set(kinds.values())) > 1:
        print("WARNING: mixing per-fold and fold-mean files in one compare")
        for p, k in kinds.items():
            print(f"  {k:<10} {_run_name(p)}")

    base = load_mean(baseline)[1]
    bname = _run_name(baseline)
    print(f"baseline : {bname}")
    for other in others:
        o = load_mean(other)[1]
        oname = _run_name(other)
        print(f"\n{oname} vs {bname} (mean over test set)")
        for label in base:
            for metric in ("Dice", "Avg. Symmetric Surface Distance"):
                bv, ov = base[label].get(metric), o[label].get(metric)
                if bv is None or ov is None:
                    continue
                print(f"  label {label:>10} | {metric:<32} "
                      f"{bv:.4f} -> {ov:.4f}   delta={ov - bv:+.4f}")

#
# main
#
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag")
    p.add_argument("--tophat", action="store_true")
    p.add_argument("--bottomhat", action="store_true")
    p.add_argument("--morph-block", action="store_true")
    p.add_argument("--morph-k", type=int, default=5)
    p.add_argument("--morph-beta", type=float, default=10.0)
    p.add_argument("--morph-loss", action="store_true")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    p.add_argument("--fold-mean", metavar="TAG")
    p.add_argument("--compare", nargs="+", metavar="JSON")
    p.add_argument("--test-only", action="store_true")
    args = p.parse_args()

    if args.fold_mean:
        fold_mean(args.fold_mean)
        return
    if args.compare:
        compare_runs(args.compare)
        return
    if not args.tag:
        p.error("--tag is required")

    set_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        # let Ada use TF32 matmuls and autotune convs for the fixed patch size
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader, test_loader, in_channels = build_loaders(args)

    if args.morph_block:
        # under --morph-block the --tophat/--bottomhat flags select the
        # learnable residuals, default to top-hat if neither is given
        use_th = args.tophat or not args.bottomhat
        use_bh = args.bottomhat
        base = UNet(num_classes=args.num_classes, in_channels=1 + use_th + use_bh)
        model = MorphResidualUNet(base, k=args.morph_k, beta=args.morph_beta,
                                  use_tophat=use_th, use_bottomhat=use_bh).to(device)
    else:
        model = UNet(num_classes=args.num_classes, in_channels=in_channels).to(device)

    # (<tag>_f<fold>_{best.pth,last.pth,scores.json}
    stem = f"{args.tag}_f{args.fold}"
    if args.morph_block:
        res = "+".join((["tophat"] if use_th else []) + (["bottomhat"] if use_bh else []))
        mode = f"morph-block({res},k={args.morph_k},beta={args.morph_beta})"
    else:
        parts = (["tophat"] if args.tophat else []) + (["bottomhat"] if args.bottomhat else [])
        mode = "+".join(parts) if parts else "baseline"
    loss_desc = "dice+ce" + ("+morph" if args.morph_loss else "")
    print(f"[{stem}] device={device} mode={mode} loss={loss_desc} seed={args.seed} "
          f"fold={args.fold} loader_in_ch={in_channels}")

    dice_loss = SoftDiceLoss(batch_dice=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    morph_loss = MorphConsistencyLoss().to(device) if args.morph_loss else None
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=6)

    results_dir = os.path.join(config.PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    best_path = os.path.join(results_dir, f"{stem}_best.pth")
    last_path = os.path.join(results_dir, f"{stem}_last.pth")
    if not args.test_only:
        best_val = float("inf")
        since_improved = 0
        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, device, dice_loss, ce_loss, optimizer, morph_loss)
            vl = run_epoch(model, val_loader, device, dice_loss, ce_loss, None, morph_loss)
            scheduler.step(vl)
            print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  val={vl:.4f}")
            if vl < best_val:
                best_val = vl
                since_improved = 0
                torch.save(model.state_dict(), best_path)
            else:
                since_improved += 1
                patience_str = f"/{args.patience}" if args.patience else ""
                print(f"  val loss did not improve from {best_val:.4f} (patience: {since_improved}{patience_str})")
            torch.save(model.state_dict(), last_path)
            if args.patience and since_improved >= args.patience:
                print(f"early stop: no val improvement for {args.patience} "
                      f"epochs (best val={best_val:.4f} @ ep {epoch - since_improved})")
                break
    ckpt = best_path if os.path.exists(best_path) else last_path
    model.load_state_dict(torch.load(ckpt, map_location=device))
    json_path = os.path.join(results_dir, f"{stem}_scores.json")
    scores = evaluate_test(model, test_loader, device, json_path)
    print(f"[{stem}] mean scores written to {json_path}")
    for label, md in scores["mean"].items():
        print(f"  label {label}: Dice={md.get('Dice')} "
              f"ASSD={md.get('Avg. Symmetric Surface Distance')}")


if __name__ == "__main__":
    main()
