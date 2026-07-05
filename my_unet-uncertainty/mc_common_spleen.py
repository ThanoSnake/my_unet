#
# Shared data + train/eval plumbing for the Spleen MC-Dropout scripts
#
# Spleen counterpart of mc_common.build_plain_loaders. Reads the same 2-channel
# (image, label) npy (label at slice index 1) produced by
# run_preprocessing_mc_spleen.py, but:
#   - TRAIN uses NumpyDataSetSpleen with foreground filtering (organ-bearing
#     slices + a margin of neighbours) so the net is not swamped by empty slices.
#   - VAL / TEST use the plain NumpyDataSet (every slice) so volume metrics,
#     temperature fitting and uncertainty maps see the whole scan.
#
# The tiny training/eval helpers (set_seed, pick_device, run_epoch, evaluate_test)
# are re-implemented HERE on purpose: the baseline train_eval.py imports the
# morphological modules at module load, and this uncertainty project must NOT pull
# those in. Everything here depends only on config, the loaders and (lazily) the
# non-morphological evaluation.evaluator.
#

import contextlib
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import config
from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from datasets.two_dim.NumpyDataLoader_spleen import NumpyDataSetSpleen


# ----------------------------- loaders ---------------------------------------
def build_plain_loaders_spleen(args):
    """Train/val/test loaders for the Spleen 2-channel npy.

    args must carry: fold, patch_size, batch_size, num_workers.
    Optional: fg_margin (empty-slice neighbours kept for training, default 3).
    """
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)
    tr = splits[args.fold]["train"]
    vl = splits[args.fold]["val"]
    ts = splits[args.fold]["test"]

    data_dir = str(config.PREPROCESSED_DIR)
    margin = getattr(args, "fg_margin", 3)
    common = dict(target_size=args.patch_size, batch_size=args.batch_size,
                  input_slice=(0,), label_slice=1)

    # Workers ONLY for train (the expensive elastic augmentation). val/test run
    # single-process on purpose: their augmentation is trivial (a resize), so 0 costs
    # almost nothing -- AND it avoids spinning up a SECOND pool of forked workers next
    # to the training pool once CUDA is live, which aborts the workers (SIGABRT) on a
    # DL VM. That second (val) pool is exactly what crashed with num_workers>0 while the
    # identical train pool had just run a full epoch fine.
    train = NumpyDataSetSpleen(data_dir, keys=tr, foreground_only=True, margin=margin,
                               num_processes=args.num_workers, **common)   # train: fg-filtered, parallel
    val = NumpyDataSet(data_dir, keys=vl, mode="val", do_reshuffle=False, num_processes=0, **common)
    test = NumpyDataSet(data_dir, keys=ts, mode="test", do_reshuffle=False, num_processes=0, **common)
    in_channels = 1
    return train, val, test, in_channels


# ----------------------------- train/eval engine -----------------------------
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


def run_epoch(model, loader, device, dice_loss, ce_loss, optimizer=None):
    """One Dice+CE epoch (bf16 autocast on CUDA). optimizer=None -> eval pass."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    losses = []
    amp = device.type == "cuda"
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
            if train_mode:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_test(model, loader, device, json_path):
    """Deterministic per-case metrics -> <json_path> (same schema as the baseline).
    aggregate_scores/Evaluator are imported lazily so training does not pay for
    pandas / SimpleITK, and so this module stays free of any morph import."""
    from evaluation.evaluator import aggregate_scores, Evaluator

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
    scores = aggregate_scores(pairs, evaluator=Evaluator, labels=config.LABELS,
                              json_output_file=json_path, json_author="cv-project",
                              json_task=config.TASK, advanced=True)
    return scores
