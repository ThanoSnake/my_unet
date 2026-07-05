#
# Core training / validation / test pipeline for the LOSS-EXPERIMENT track.
#
# Ports the correct data-handling logic from the thanasis project onto OUR loss
# experiments, WITHOUT touching the existing loading/preprocessing files:
#
#   build_loaders   native-resolution, foreground-oversampled TRAIN loader; full-slice
#                   VAL (packed) and full-slice TEST loaders (NumpyDataLoader_loss).
#   run_epoch       one training pass; grad-norm clipped (5.0) so a foreground-sparse
#                   batch with a sharp region gradient cannot spike the epoch.
#   run_val_dice    full-slice GLOBAL foreground Dice (nnU-Net pseudo-Dice): accumulate
#                   TP/FP/FN over the whole val subset -> one stable, low-variance number
#                   that matches the test metric. This is the model-selection signal.
#   evaluate_test   full-slice inference per case -> Dice + ASSD (+more) via the existing
#                   evaluator; uint8 accumulation keeps RAM bounded on large CT.
#
# The loss itself (Dice/CE/Focal-Tversky/Boundary/clDice) is chosen in train_losses.py
# and passed in as `loss_fn`; here we only run the loop. Selection is on Dice, so the
# loss is a training-time driver only (fair across different loss combos).
#

import contextlib
import json
import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import config
from datasets.two_dim.NumpyDataLoader_loss import NumpyDataSet
from evaluation.evaluator import Evaluator


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
# loaders: n_dist>0 -> the TRAIN loader also carries the n_dist boundary distance maps in `seg`
# (label at seg[:,0], phi at seg[:,1:]). VAL/TEST need only the label (Dice via argmax).
#
def build_loaders(args, n_dist=0, want_train=True):
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)
    tr = splits[args.fold]["train"]
    vl = splits[args.fold]["val"]
    ts = splits[args.fold]["test"]
    data_dir = str(config.PREPROCESSED_DIR)

    common = dict(target_size=args.patch_size, input_slice=(0,), num_processes=args.num_workers)

    train = None
    if want_train:
        train_label_slice = tuple(range(1, 2 + n_dist)) if n_dist > 0 else 1
        cap = dict(num_batches=args.iters_per_epoch) if getattr(args, "iters_per_epoch", 0) > 0 else {}
        train = NumpyDataSet(data_dir, keys=tr, mode="train", batch_size=args.batch_size,
                             label_slice=train_label_slice, fg_fraction=args.fg_fraction, **common, **cap)

    # val: full-slice (mode="test") so selection uses foreground Dice on the real distribution, not a
    # background-dominated centre patch. --val-cases caps to a fixed seeded subset of volumes.
    val = None
    if want_train:
        val_keys = list(vl)
        vc = getattr(args, "val_cases", 0)
        if vc and 0 < vc < len(val_keys):
            val_keys = sorted(random.Random(args.seed).sample(val_keys, vc))
        val = NumpyDataSet(data_dir, keys=val_keys, mode="test", do_reshuffle=False,
                           batch_size=getattr(args, "val_batch", 12), label_slice=1,
                           fg_fraction=0.0, **common)

    # test: full-slice inference, one at a time, all slices
    test = NumpyDataSet(data_dir, keys=ts, mode="test", do_reshuffle=False,
                        batch_size=1, label_slice=1, fg_fraction=0.0, **common)
    in_channels = 1
    return train, val, test, in_channels


#
# one TRAIN pass -> mean loss. loss_fn(logits, probs, y, phi, lam) -> scalar tensor.
#
def run_epoch(model, loader, device, loss_fn, optimizer, lam=0.0, grad_clip=5.0):
    model.train()
    losses = []
    amp = device.type == "cuda"
    with torch.enable_grad():
        for batch in loader:
            data = batch["data"][0].float().to(device, non_blocking=True)     # [b, 1, H, W]
            seg = batch["seg"][0].to(device, non_blocking=True)               # [b, S, H, W]
            y = seg[:, 0].long()                                              # [b, H, W]
            phi = seg[:, 1:].float() if seg.shape[1] > 1 else None            # [b, K, H, W] or None
            optimizer.zero_grad()
            with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
                logits = model(data)
                probs = F.softmax(logits, dim=1)
                loss = loss_fn(logits, probs, y, phi, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().item()))
    return float(np.mean(losses)) if losses else float("nan")


#
# full-slice validation -> global foreground Dice (TP/FP/FN accumulated over the whole val subset,
# nnU-Net pseudo-Dice). Faithful, low-variance model-selection signal. OOM-safe chunking.
#
def run_val_dice(model, loader, device, num_classes):
    model.eval()
    n_fg = max(num_classes - 1, 1)
    tp = torch.zeros(n_fg); fp = torch.zeros(n_fg); fn = torch.zeros(n_fg)
    amp = device.type == "cuda"
    chunk = [0]   # GPU sub-batch cap; 0 = whole loader-batch, halves after a CUDA OOM

    def _count(x, t):
        with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
            pred_lab = model(x).argmax(1)                                    # [b, H, W]
        for c in range(1, num_classes):
            p, tt, i = (pred_lab == c), (t == c), c - 1
            tp[i] += (p & tt).sum().item()
            fp[i] += (p & ~tt).sum().item()
            fn[i] += (~p & tt).sum().item()

    with torch.no_grad():
        for batch in loader:
            data = batch["data"][0].float()                                 # [b, c, H, W] (cpu)
            target = batch["seg"][0][:, 0].long()                           # [b, H, W]
            b, i = data.shape[0], 0
            while i < b:                                                     # forward in OOM-safe chunks
                step = min(chunk[0] or b, b - i)
                try:
                    _count(data[i:i + step].to(device, non_blocking=True),
                           target[i:i + step].to(device, non_blocking=True))
                    i += step
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower() or step == 1:
                        raise
                    torch.cuda.empty_cache()
                    chunk[0] = max(step // 2, 1)
                    print(f"[val] CUDA OOM at chunk {step} -> retry at {chunk[0]}", flush=True)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)         # per-class global Dice
    present = (tp + fn) > 0                              # classes actually present in the val subset
    mean = dice[present].mean().item() if present.any() else 0.0
    return mean, dice.tolist()


#
# test: full-slice inference per case -> Dice + ASSD.
#
# We compute ONLY Dice + ASSD (the two target metrics) and PARALLELISE the metric
# computation across cases. The default evaluator also computes exact Hausdorff /
# HD95 / ASD, and does it sequentially -- on full-resolution HepaticVessel (huge
# vessel surface) that took *hours* for a single fold. Dice + ASSD, spread over the
# CPU cores, brings it down to minutes. (Add more metrics here later if you want.)
#
_TEST_METRICS = ["Dice"]
_TEST_ADVANCED = ["Avg. Symmetric Surface Distance"]


def _eval_one(payload):
    """Dice + ASSD for ONE (pred, gt) case. Module-level so multiprocessing can pickle it."""
    pred, gt, labels = payload
    ev = Evaluator(labels=labels, metrics=list(_TEST_METRICS), advanced_metrics=list(_TEST_ADVANCED))
    ev.set_test(pred)
    ev.set_reference(gt)
    res = ev.evaluate(advanced=True)
    return {lab: {m: float(v) for m, v in md.items()} for lab, md in res.items()}


def evaluate_test(model, loader, device, json_path, metric_workers=None):
    model.eval()
    # accumulate per-case pred/GT as uint8 (labels are small ints) to keep RAM bounded on large CT
    pred_dict, gt_dict = defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            data = batch["data"][0].float().to(device)
            target = batch["seg"][0][:, 0].to(torch.uint8).numpy()           # [b, H, W]
            pred = model(data).argmax(1).to(torch.uint8).cpu().numpy()       # [b, H, W]
            for i, fname in enumerate(batch["fnames"]):
                pred_dict[fname[0]].append(pred[i])
                gt_dict[fname[0]].append(target[i])
    pairs = [(np.stack(pred_dict[k]), np.stack(gt_dict[k]), config.LABELS) for k in pred_dict]  # each [Z,H,W]

    # metric computation is the bottleneck on full-res vessels -> parallelise across cases
    nproc = min(len(pairs), metric_workers or (os.cpu_count() or 1)) if pairs else 1
    print(f"[eval] {len(pairs)} test cases -> Dice + ASSD on {nproc} process(es)", flush=True)
    if nproc > 1:
        with Pool(nproc) as pool:
            per_case = pool.map(_eval_one, pairs)
    else:
        per_case = [_eval_one(p) for p in pairs]

    # aggregate (nanmean over cases), matching the JSON the existing tools read
    acc = {}
    for case in per_case:
        for lab, md in case.items():
            acc.setdefault(lab, {})
            for m, v in md.items():
                acc[lab].setdefault(m, []).append(v)
    mean = {lab: {m: float(np.nanmean(vs)) for m, vs in md.items()} for lab, md in acc.items()}

    out = {"task": config.TASK, "results": {"all": per_case, "mean": mean}}
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    return {"mean": mean}
