#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Augmentation for the LOSS-EXPERIMENT track (native-resolution patches, NO resize).
#
# Ported from the thanasis pipeline. Unlike datasets/two_dim/data_augmentation.py
# (which resizes the whole slice to target_size and thus destroys thin structures on
# large CT), here the loader (NumpyDataLoader_loss.py) already produces fixed-size
# crops at NATIVE resolution. So training only adds mirror + spatial deformation;
# val/test add nothing (full slices go straight into the fully-convolutional U-Net).
# The existing (resize-based) augmentation file is left untouched.
#

from batchgenerators.transforms import Compose, MirrorTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

import numpy as np


def get_transforms(mode="train", target_size=128):
    transform_list = []

    if mode == "train":
        # patch already cropped by the loader at native resolution -> only deform it
        transform_list = [
            MirrorTransform(axes=(1,)),
            SpatialTransform(patch_size=(target_size, target_size), random_crop=False,
                             patch_center_dist_from_border=target_size // 2,
                             do_elastic_deform=True, alpha=(0., 900.), sigma=(20., 30.),
                             do_rotation=True, p_rot_per_sample=0.8,
                             angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                             angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                             scale=(0.85, 1.25), p_scale_per_sample=0.8,
                             border_mode_data="nearest", border_mode_seg="nearest"),
        ]

    # val / test: no spatial augmentation (loader already produced fixed-size / full slices)
    transform_list.append(NumpyToTensor())

    return Compose(transform_list)
