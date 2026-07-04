#
# Preprocessing for the loss-experiment U-Net (image, label, + boundary distance maps)
#
# Same generic steps as datasets/preprocessing_plain.py -- modality-aware
# normalisation, medpy 4-D axis fix, padding -- but ALSO precomputes, per 2-D
# axial slice and per FOREGROUND class, the level-set (signed) distance map used
# by the Boundary loss (Kervadec et al., 2019). Precomputing here means training
# never pays the (slow, CPU) distance transform on the fly.
#
# Output is a (2 + K)-channel npy, K = num_classes - 1 foreground classes:
#
#   channel 0        = image
#   channel 1        = label
#   channel 2..1+K   = phi_c, signed distance map of class c (c = 1..K),
#                      >0 OUTSIDE the object, <0 INSIDE (canonical one_hot2dist)
#
# The extra channels ride in the loader's `seg` (a tuple label_slice), so the
# geometric augmentation keeps them aligned with the label. clDice needs nothing
# precomputed (its skeletons are cheap GPU pooling), so the SAME dataset serves
# BOTH loss combos. Other preprocessing files are untouched.
#

import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image
from scipy.ndimage import distance_transform_edt

from utilities.morph_explore import load_any, align_axes, preprocess as modality_preprocess


def _slice_sdf(label_2d, num_fg):
    """Signed distance map per foreground class for one 2-D label slice.
    Returns (K, H, W) float32: phi>0 outside the class region, phi<0 inside
    (canonical one_hot2dist of the boundary-loss paper). Absent class -> zeros.

    The map is NORMALISED by the slice's largest side, so phi is dimensionless
    (roughly [-1, 1]) and RESOLUTION-INDEPENDENT. This matters because the loader
    resizes every slice to the patch size (e.g. 512 -> 64); raw pixel distances
    would then be ~8x too large for the 64-grid and the boundary term would swamp
    the region loss. Normalising keeps a single, task-independent lam sensible."""
    out = np.zeros((num_fg,) + label_2d.shape, dtype=np.float32)
    norm = float(max(label_2d.shape)) or 1.0
    for k in range(num_fg):
        posmask = label_2d == (k + 1)
        if posmask.any():
            negmask = ~posmask
            sdf = distance_transform_edt(negmask) * negmask \
                - (distance_transform_edt(posmask) - 1) * posmask
            out[k] = sdf / norm
    return out


def _sdf_volume(label, num_fg):
    """Per-slice SDF over a (D, H, W) label volume -> (K, D, H, W)."""
    per_slice = [_slice_sdf(label[d], num_fg) for d in range(label.shape[0])]
    return np.stack(per_slice, axis=1).astype(np.float32)      # (K, D, H, W)


def _process_case(f, image_dir, label_dir, output_dir, mod, channel, n_mod, y_shape, z_shape, num_fg):
    """Preprocess one case -> save (2 + K)-channel npy. Module-level so Pool can pickle it."""
    image = load_any(os.path.join(image_dir, f))
    label = load_any(os.path.join(label_dir, f.replace('_0000', '')))
    if label.ndim == 4:
        label = label[..., 0]

    # modality-aware normalisation + multi-modal channel selection, then fix
    # medpy's permuted 4-D spatial axes against the label
    image = modality_preprocess(image, mod, channel, n_mod)
    image = align_axes(image, label)

    pad = (image.shape[0], y_shape, z_shape)
    image = pad_nd_image(image, pad, "constant", kwargs={'constant_values': image.min()})
    label = pad_nd_image(label, pad, "constant", kwargs={'constant_values': label.min()})

    # per-class signed distance maps for the boundary loss (computed on the padded label)
    phi = _sdf_volume(np.rint(label).astype(np.int16), num_fg)     # (K, D, H, W)

    # channel order: 0=image, 1=label, 2..=phi_c ; float32 halves the disk vs float64
    result = np.concatenate([image[None], label[None], phi], axis=0).astype(np.float32)
    np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), result)
    return f


def preprocess_data_boundary(root_dir, modality=None, channel=0, num_classes=3,
                             y_shape=64, z_shape=64, num_workers=None):
    image_dir = os.path.join(root_dir, 'imagesTr')
    label_dir = os.path.join(root_dir, 'labelsTr')
    output_dir = os.path.join(root_dir, 'preprocessed')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    modality = modality or {"0": "MRI"}
    mod = modality[str(channel)] if str(channel) in modality else modality.get("0", "MRI")
    n_mod = len(modality)
    num_fg = max(1, num_classes - 1)

    nii_files = [fn for fn in sorted(os.listdir(image_dir))
                 if fn.endswith((".nii", ".nii.gz")) and not fn.startswith("._")]
    if not nii_files:
        raise FileNotFoundError(f"no .nii/.nii.gz images found in {image_dir}")

    worker = partial(_process_case, image_dir=image_dir, label_dir=label_dir, output_dir=output_dir,
                     mod=mod, channel=channel, n_mod=n_mod, y_shape=y_shape, z_shape=z_shape, num_fg=num_fg)
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    if num_workers <= 1:
        for f in nii_files:
            print(worker(f), flush=True)
    else:
        with Pool(num_workers) as pool:
            for done in pool.imap_unordered(worker, nii_files):
                print(done, flush=True)
