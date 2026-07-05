#
# One-time preprocessing for the MC-Dropout track on Task09_Spleen
#
# Spleen counterpart of run_preprocessing_mc.py. Produces the 2-channel
# (image, label) npy at a common (Z, SIZE, SIZE) + the seeded k-fold splits.pkl.
# Unlike the Hippocampus version it orients axially, body-crops and resizes (see
# datasets/preprocessing_plain_spleen.py). create_splits is reused unchanged.
#
# GCP L4 VM (writes into DATA_DIR -> point it at a WRITABLE copy of the raw task):
#   DATA_DIR=$PWD/data/Task09_Spleen TASK=Task09_Spleen \
#       python run_preprocessing_mc_spleen.py --size 256
#

import argparse

import config
from datasets.preprocessing_plain_spleen import preprocess_data_spleen
from datasets.create_splits import create_splits

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256,
                   help="common in-plane size S; every case is resized to (Z, S, S). "
                        "Train/eval with --patch-size <= S (loader can downsize); "
                        "re-run this with a larger --size to go above S.")
    p.add_argument("--ct-center", type=float, default=40.0, help="HU window center (soft tissue)")
    p.add_argument("--ct-width", type=float, default=400.0, help="HU window width")
    p.add_argument("--body-thr", type=float, default=0.1,
                   help="normalised-intensity threshold for the body mask (air -> 0)")
    p.add_argument("--body-margin", type=int, default=8, help="pixels of margin around the body bbox")
    p.add_argument("--num-workers", type=int, default=None)
    args = p.parse_args()

    preprocess_data_spleen(root_dir=str(config.DATA_DIR), size=args.size,
                           ct_center=args.ct_center, ct_width=args.ct_width,
                           body_thr=args.body_thr, body_margin=args.body_margin,
                           num_workers=args.num_workers)
    create_splits(output_dir=str(config.DATA_DIR), image_dir=str(config.PREPROCESSED_DIR))
    print(f"Spleen (2-channel, {args.size}x{args.size}) preprocessing done.")
