#
# Minimal preprocessing for the MC-Dropout U-Net (no morphology)
#
# Same generic steps as datasets/preprocessing.py -- modality-aware
# normalisation (CT window / MRI percentile), medpy 4-D axis fix, padding --
# but WITHOUT the top-hat / bottom-hat channels, since the plain MC-Dropout
# U-Net only ever consumes the image. Output is a 2-channel npy:
#
#   channel 0 = image, channel 1 = label      -> shape (2, D, H, W)
#
# (The morphological pipeline saves 4 channels with the label at index 3; here
# the label is at index 1. mc_common.build_plain_loaders reads it accordingly.)
#

import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image

from utilities.morph_explore import load_any, align_axes, preprocess as modality_preprocess


def _process_case(f, image_dir, label_dir, output_dir, mod, channel, n_mod, y_shape, z_shape):
    """Preprocess one case -> save 2-channel npy. Module-level so Pool can pickle it."""
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

    # channel order: 0=image, 1=label; float32 halves the disk vs float64
    result = np.stack((image, label)).astype(np.float32)
    np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), result)
    return f


def preprocess_data_plain(root_dir, modality=None, channel=0, y_shape=64, z_shape=64, num_workers=None):
    image_dir = os.path.join(root_dir, 'imagesTr')
    label_dir = os.path.join(root_dir, 'labelsTr')
    output_dir = os.path.join(root_dir, 'preprocessed')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    modality = modality or {"0": "MRI"}
    mod = modality[str(channel)] if str(channel) in modality else modality.get("0", "MRI")
    n_mod = len(modality)

    nii_files = [fn for fn in sorted(os.listdir(image_dir))
                 if fn.endswith((".nii", ".nii.gz")) and not fn.startswith("._")]
    if not nii_files:
        raise FileNotFoundError(f"no .nii/.nii.gz images found in {image_dir}")

    worker = partial(_process_case, image_dir=image_dir, label_dir=label_dir, output_dir=output_dir,
                     mod=mod, channel=channel, n_mod=n_mod, y_shape=y_shape, z_shape=z_shape)
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    if num_workers <= 1:
        for f in nii_files:
            print(worker(f), flush=True)
    else:
        with Pool(num_workers) as pool:
            for done in pool.imap_unordered(worker, nii_files):
                print(done, flush=True)
