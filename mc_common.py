#
# Shared data plumbing for the MC-Dropout scripts
#
# The MC-Dropout track uses the 2-channel (image, label) npy produced by
# run_preprocessing_mc.py -> the label lives at slice index 1 (not 3 as in the
# morphological 4-channel format). That is the only reason train_mc / test_mc /
# uncertainty_mc cannot reuse train_eval.build_loaders (which hardcodes
# label_slice=3); everything else (run_epoch, evaluate_test) is reused by import.
#

import pickle

import config
from datasets.two_dim.NumpyDataLoader import NumpyDataSet


def build_plain_loaders(args):
    """Train/val/test loaders for the 2-channel MC-Dropout npy.

    input  = slice 0 (image)   -> in_channels = 1
    target = slice 1 (label)
    Reads the preprocessed dir and the k-fold splits from config.
    Expects args to carry: fold, patch_size, batch_size, num_workers.
    """
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)
    tr = splits[args.fold]["train"]
    vl = splits[args.fold]["val"]
    ts = splits[args.fold]["test"]

    data_dir = str(config.PREPROCESSED_DIR)
    common = dict(target_size=args.patch_size, batch_size=args.batch_size,
                  input_slice=(0,), label_slice=1, num_processes=args.num_workers)

    train = NumpyDataSet(data_dir, keys=tr, **common)
    val = NumpyDataSet(data_dir, keys=vl, mode="val", do_reshuffle=False, **common)
    test = NumpyDataSet(data_dir, keys=ts, mode="test", do_reshuffle=False, **common)
    in_channels = 1
    return train, val, test, in_channels
