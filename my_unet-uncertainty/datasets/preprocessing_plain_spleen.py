#
# Minimal preprocessing for the MC-Dropout U-Net on Task09_Spleen (no morphology)
#
# The Hippocampus track (datasets/preprocessing_plain.py) pads tiny ~40^3 volumes
# to (D, 64, 64) and lets the loader resize each slice to patch_size. That does
# NOT transfer to the Spleen, which is a large abdominal CT (~512x512xZ) with a
# large but OFF-CENTRE organ. Two things change here:
#
#   1. AXIAL orientation. The loader slices along array axis 0. For the Spleen we
#      must put the through-plane (axial) axis first so each 2D slice is a
#      512x512 axial cross-section, not a 512xZ sagittal/coronal strip.
#   2. RESIZE happens HERE, not in the loader. The loader batches slices coming
#      from DIFFERENT volumes; if every volume kept its own body-crop size the
#      batch stack would be ragged. So every case is brought to a common SxS:
#         body-crop (drop the air margin) -> square-pad (keep aspect) -> resize S.
#      Image uses bilinear, LABEL uses nearest-neighbour (no fractional classes).
#
# This file is deliberately SELF-CONTAINED (no morph_* imports): CT windowing and
# axis handling are re-implemented inline. Output is a 2-channel npy, identical in
# layout to the Hippocampus MC track so mc_common_spleen / the loader read it the
# same way:
#
#   channel 0 = image (CT-windowed to [0,1]), channel 1 = label   -> (2, Z, S, S)
#

import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from medpy.io import load


# ---- CT intensity normalisation (fixed Hounsfield window) --------------------
# A FIXED soft-tissue window (center 40, width 400 HU -> [-160, 240]) mapped to
# [0, 1]. Fixed (not per-volume min-max) so the intensity->confidence mapping is
# identical across cases, which matters for calibration. Air (~-1000 HU) -> 0.
def normalize_ct(vol, center=40.0, width=400.0):
    lo, hi = center - width / 2.0, center + width / 2.0
    v = np.clip(vol.astype(np.float32), lo, hi)
    return (v - lo) / (hi - lo)


# ---- axial (through-plane) axis ----------------------------------------------
# CT slice thickness (~1.5-5 mm) is larger than the in-plane spacing (~0.6-1 mm),
# so the axial axis is the one with the LARGEST voxel spacing. Fallback when the
# header has no usable spacing: the shortest axis (Z < 512 for the Spleen).
def axial_axis(header, shape):
    try:
        spacing = header.get_voxel_spacing()
        if spacing is not None and len(spacing) == len(shape):
            return int(np.argmax(spacing))
    except Exception:
        pass
    return int(np.argmin(shape))


# ---- body bounding box (label-free, so identical at train and test) ----------
# Everything above air is body/tissue. Collapse the volume to a 2D in-plane
# occupancy mask, keep the largest connected component (drops the scanner table
# and speckle), and return its bounding box padded by `margin`.
def body_bbox(vol01, thr=0.1, margin=8):
    inplane = (vol01 > thr).any(axis=0)
    if not inplane.any():
        return None
    try:
        from scipy.ndimage import label as cc_label
        lab, n = cc_label(inplane)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            inplane = lab == int(sizes.argmax())
    except Exception:
        pass
    rows = np.where(inplane.any(axis=1))[0]
    cols = np.where(inplane.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    r0 = max(0, r0 - margin); c0 = max(0, c0 - margin)
    r1 = min(vol01.shape[1], r1 + margin); c1 = min(vol01.shape[2], c1 + margin)
    return r0, r1, c0, c1


# ---- square-pad + resize a (Z, H, W) volume ----------------------------------
# Pad the in-plane to a square (side = max(H, W)) so the following resize keeps
# the anatomical aspect ratio, then zoom to (size, size). order=1 (bilinear) for
# the image, order=0 (nearest) for the label -> labels stay integer-valued.
def _square_resize(vol, size, order):
    from scipy.ndimage import zoom
    z, h, w = vol.shape
    side = max(h, w)
    if h != side or w != side:
        pad = ((0, 0), (0, side - h), (0, side - w))
        vol = np.pad(vol, pad, mode="constant", constant_values=0.0)
    if side != size:
        vol = zoom(vol, (1.0, size / side, size / side), order=order)
    # guarantee EXACTLY (Z, size, size): zoom's rounding can land on size +/- 1,
    # and every case must share the in-plane shape or the batch stack is ragged
    if vol.shape[1] != size or vol.shape[2] != size:
        fixed = np.zeros((vol.shape[0], size, size), dtype=vol.dtype)
        h2, w2 = min(size, vol.shape[1]), min(size, vol.shape[2])
        fixed[:, :h2, :w2] = vol[:, :h2, :w2]
        vol = fixed
    return vol


def _process_case(f, image_dir, label_dir, output_dir, size, ct_center, ct_width,
                  body_thr, body_margin):
    """Preprocess one Spleen case -> save 2-channel npy. Module-level for Pool."""
    image, header = load(os.path.join(image_dir, f))
    label, _ = load(os.path.join(label_dir, f.replace('_0000', '')))
    if label.ndim == 4:
        label = label[..., 0]

    image = normalize_ct(image, center=ct_center, width=ct_width)

    # put the axial axis first (both share the file geometry -> same axis)
    ax = axial_axis(header, image.shape)
    image = np.moveaxis(image, ax, 0)
    label = np.moveaxis(label, ax, 0)

    # drop the air margin (bbox from the image only)
    bbox = body_bbox(image, thr=body_thr, margin=body_margin)
    if bbox is not None:
        r0, r1, c0, c1 = bbox
        image = image[:, r0:r1, c0:c1]
        label = label[:, r0:r1, c0:c1]

    # bring every case to a common (Z, size, size); image bilinear, label nearest
    image = _square_resize(image, size, order=1)
    label = _square_resize(label, size, order=0)
    label = np.rint(label)   # guard against tiny zoom round-off

    result = np.stack((image, label)).astype(np.float32)
    np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), result)
    return f


def preprocess_data_spleen(root_dir, size=256, ct_center=40.0, ct_width=400.0,
                           body_thr=0.1, body_margin=8, num_workers=None):
    image_dir = os.path.join(root_dir, 'imagesTr')
    label_dir = os.path.join(root_dir, 'labelsTr')
    output_dir = os.path.join(root_dir, 'preprocessed')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    nii_files = [fn for fn in sorted(os.listdir(image_dir))
                 if fn.endswith((".nii", ".nii.gz")) and not fn.startswith("._")]
    if not nii_files:
        raise FileNotFoundError(f"no .nii/.nii.gz images found in {image_dir}")

    worker = partial(_process_case, image_dir=image_dir, label_dir=label_dir,
                     output_dir=output_dir, size=size, ct_center=ct_center,
                     ct_width=ct_width, body_thr=body_thr, body_margin=body_margin)
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    if num_workers <= 1:
        for f in nii_files:
            print(worker(f), flush=True)
    else:
        with Pool(num_workers) as pool:
            for done in pool.imap_unordered(worker, nii_files):
                print(done, flush=True)
