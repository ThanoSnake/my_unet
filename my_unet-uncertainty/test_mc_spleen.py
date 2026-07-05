#
# Test the Spleen MC-Dropout U-Net -- classic metrics only (deterministic)
#
# Spleen counterpart of test_mc.py. Loads a trained checkpoint, runs a SINGLE
# deterministic forward pass with dropout OFF (model.eval()) over EVERY test
# slice, and writes a <tag>_f<fold>_scores.json (same schema as the baseline).
# run/run_all_spleen.sh aggregates the per-fold Dice from these files directly
# (no train_eval import, so nothing morphological is pulled in).
#
# Metrics are computed in the working (resized SxS) space, exactly as in the
# Hippocampus track. The stochastic / uncertainty side lives in
# uncertainty_mc_spleen.py.
#

import argparse
import os

import torch

import config
from mc_common_spleen import build_plain_loaders_spleen, pick_device, evaluate_test
from networks.UNET_mc import MCDropoutUNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="mcdropout")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    p.add_argument("--out-dir", default=os.path.join(config.PROJECT_ROOT, "results"))
    p.add_argument("--ckpt", default=None,
                   help="checkpoint path (default <out-dir>/<tag>_f<fold>_best.pth)")
    args = p.parse_args()

    device = pick_device()
    _, _, test_loader, in_channels = build_plain_loaders_spleen(args)
    model = MCDropoutUNet(num_classes=args.num_classes, in_channels=in_channels,
                          dropout_p=args.dropout_p).to(device)

    stem = f"{args.tag}_f{args.fold}"
    ckpt = args.ckpt or os.path.join(args.out_dir, f"{stem}_best.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"[{stem}] loaded {ckpt}  (dropout OFF -> deterministic classic metrics)")

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"{stem}_scores.json")
    # evaluate_test does model.eval() internally -> dropout disabled
    scores = evaluate_test(model, test_loader, device, json_path)

    print(f"[{stem}] scores -> {json_path}")
    for label, md in scores["mean"].items():
        print(f"  label {label}: Dice={md.get('Dice')} "
              f"ASSD={md.get('Avg. Symmetric Surface Distance')}")


if __name__ == "__main__":
    main()
