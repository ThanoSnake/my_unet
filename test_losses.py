#
# Test a baseline U-Net trained with an alternative loss -- Dice + ASSD metrics.
#
# Loads a checkpoint from train_losses.py and runs FULL-SLICE inference at native
# resolution (pipeline_loss.evaluate_test), writing <tag>_f<fold>_scores.json (Dice,
# ASSD, ... per class). This is the faithful, literature-comparable evaluation -- NOT
# the old resize-to-64. The existing aggregation still works:
#   python train_eval.py --fold-mean ftversky_ce_boundary
#   python train_eval.py --compare dice_ce_f0_scores.json ftversky_ce_boundary_f0_scores.json
#
# The npy's distance-map channels are ignored at test time (label_slice=1). Network is
# the untouched baseline networks/UNET.py.
#
# Kaggle:
#   DATA_DIR=/kaggle/input/<prep>/Task08_HepaticVessel TASK=Task08_HepaticVessel \
#       python test_losses.py --tag ftversky_ce_boundary --fold 0 --out-dir /kaggle/working/results
#

import argparse
import os

import torch

import config
from pipeline_loss import pick_device, build_loaders, evaluate_test
from networks.UNET import UNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True,
                   help="e.g. dice_ce / ftversky_ce_boundary / cldice_dice_ce")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--patch-size", type=int, default=128, help="unused at test (full slice) but kept for the loader API")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--fg-fraction", type=float, default=0.0)
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    p.add_argument("--pp-min-size", type=int, default=0,
                   help="post-processing: remove 3D connected components smaller than this many voxels "
                        "(0 = off). Writes a SEPARATE <tag>_pp_f<fold>_scores.json (metrics AFTER post-proc) "
                        "next to the raw one (metrics BEFORE).")
    p.add_argument("--out-dir", default=os.path.join(config.PROJECT_ROOT, "results"))
    p.add_argument("--ckpt", default=None, help="checkpoint path (default <out-dir>/<tag>_f<fold>_best.pth)")
    args = p.parse_args()

    device = pick_device()
    _, _, test_loader, in_channels = build_loaders(args, n_dist=0, want_train=False)
    model = UNet(num_classes=args.num_classes, in_channels=in_channels).to(device)

    stem = f"{args.tag}_f{args.fold}"
    ckpt = args.ckpt or os.path.join(args.out_dir, f"{stem}_best.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:      # tolerate a full-state last.pth
        state = state["model"]
    model.load_state_dict(state)
    print(f"[{stem}] loaded {ckpt}  (full-slice deterministic Dice/ASSD)")

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"{stem}_scores.json")                       # BEFORE post-proc
    pp_json_path = (os.path.join(args.out_dir, f"{args.tag}_pp_f{args.fold}_scores.json")
                    if args.pp_min_size > 0 else None)                                  # AFTER post-proc
    scores = evaluate_test(model, test_loader, device, json_path,
                           pp_json_path=pp_json_path, pp_min_size=args.pp_min_size)

    print(f"[{stem}] raw scores -> {json_path}")
    if pp_json_path:
        print(f"[{stem}] post-processed (min_size={args.pp_min_size}) scores -> {pp_json_path}")
    for label, md in scores["mean"].items():
        print(f"  label {label}: Dice={md.get('Dice')} "
              f"ASSD={md.get('Avg. Symmetric Surface Distance')}")


if __name__ == "__main__":
    main()
