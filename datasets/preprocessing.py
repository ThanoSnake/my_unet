#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from batchgenerators.augmentations.utils import pad_nd_image
from medpy.io import load
import os
from functools import partial
from multiprocessing import Pool
import numpy as np
import torch
from scipy.ndimage import grey_opening, grey_closing
from skimage.morphology import ball

from utilities.morph_explore import load_any, align_axes, preprocess as modality_preprocess


def _process_case(f, image_dir, label_dir, output_dir, mod, channel, n_mod, se_radius, y_shape, z_shape):
    """Preprocess one case -> save 4-channel npy. Module-level so it is picklable for Pool."""
    image = load_any(os.path.join(image_dir, f))
    label = load_any(os.path.join(label_dir, f.replace('_0000', '')))
    if label.ndim == 4:
        label = label[..., 0]

    # modality-aware normalisation (CT window / MRI percentile) + multi-modal channel
    # selection, then fix medpy's permuted 4-D spatial axes against the label
    image = modality_preprocess(image, mod, channel, n_mod)
    image = align_axes(image, label)

    # x - opening(x) (top-hat), closing(x) - x (bottom-hat)
    tophat = np.clip(image - grey_opening(image, footprint=ball(se_radius)), 0, None)
    bottomhat = np.clip(grey_closing(image, footprint=ball(se_radius)) - image, 0, None)

    pad = (image.shape[0], y_shape, z_shape)
    image = pad_nd_image(image, pad, "constant", kwargs={'constant_values': image.min()})
    tophat = pad_nd_image(tophat, pad, "constant", kwargs={'constant_values': 0.0})
    bottomhat = pad_nd_image(bottomhat, pad, "constant", kwargs={'constant_values': 0.0})
    label = pad_nd_image(label, pad, "constant", kwargs={'constant_values': label.min()})

    # channel order: 0=image, 1=tophat, 2=bottomhat, 3=label; float32 halves the disk vs float64
    result = np.stack((image, tophat, bottomhat, label)).astype(np.float32)
    np.save(os.path.join(output_dir, f.split('.')[0]+'.npy'), result)
    return f


def preprocess_data(root_dir, modality=None, channel=0, y_shape=64, z_shape=64, se_radius=2, num_workers=None):
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

    # cases are independent -> run them across worker processes (one volume each,
    # ~1-1.5 GB); imap_unordered prints each filename as that case finishes
    worker = partial(_process_case, image_dir=image_dir, label_dir=label_dir, output_dir=output_dir,
                     mod=mod, channel=channel, n_mod=n_mod, se_radius=se_radius,
                     y_shape=y_shape, z_shape=z_shape)
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    if num_workers <= 1:
        for f in nii_files:
            print(worker(f), flush=True)
    else:
        with Pool(num_workers) as pool:
            for done in pool.imap_unordered(worker, nii_files):
                print(done, flush=True)


def preprocess_single_file(image_file):
    image, image_header = load(image_file)
    image = (image - image.min()) / (image.max() - image.min())
    data = np.expand_dims(image, 1)
    return torch.from_numpy(data), image_header


def postprocess_single_image(image):
    # desired shape is [b w h]
    result_converted = image[::, 0, ::, ::]
    result_mapped = [i * 255 for i in result_converted]
    return result_mapped
