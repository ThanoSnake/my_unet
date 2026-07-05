#
# Foreground-aware 2D loader for the Spleen MC-Dropout track
#
# In an abdominal CT most axial slices contain NO spleen (~60-90%). Training on
# all of them buries the organ under background and the net drifts to predicting
# all-background. This loader keeps, for TRAINING only, the slices that contain
# the organ PLUS `margin` neighbouring empty slices (the hard, near-boundary
# negatives) -- a softer, better-balanced alternative to a hard empty-slice drop
# that still exposes the net to true negatives so it does not hallucinate spleen.
#
# val / test build the plain NumpyDataSet (foreground_only=False) so EVERY slice
# is scored and the volume metrics / uncertainty maps stay honest.
#
# Everything else (the augmentation transforms, the multithreaded wrapper) is
# reused by import from the existing two_dim loader -- no existing file is touched.
#

from collections import defaultdict

import numpy as np

from datasets.data_loader import MultiThreadedDataLoader
from datasets.two_dim.data_augmentation import get_transforms
from datasets.two_dim.NumpyDataLoader import NumpyDataLoader


class NumpyDataLoaderSpleen(NumpyDataLoader):
    """NumpyDataLoader that (optionally) restricts to organ-bearing axial slices.

    foreground_only=True keeps slices whose label has any foreground, widened by
    `margin` neighbours on each side; label lives at channel `label_channel` of
    the (C, Z, H, W) npy (1 in the 2-channel image+label format)."""

    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,), keys=None,
                 foreground_only=True, margin=3, label_channel=1):
        super().__init__(base_dir, mode=mode, batch_size=batch_size, num_batches=num_batches,
                         file_pattern=file_pattern, label_slice=label_slice,
                         input_slice=input_slice, keys=keys)
        if foreground_only:
            self._keep_foreground(margin, label_channel)

    def _keep_foreground(self, margin, label_channel):
        # group the (file_idx, slice_idx) pairs by file so each npy is read once
        by_file = defaultdict(list)
        for fi, sj in self.slices:
            by_file[fi].append(sj)

        kept = []
        for fi, sjs in by_file.items():
            arr = np.load(self.files[fi], mmap_mode="r")
            lbl = np.asarray(arr[label_channel])                 # (Z, H, W)
            fg_z = np.where((lbl > 0).any(axis=(1, 2)))[0]
            if fg_z.size == 0:
                continue                                          # case without organ
            keep_z = set()
            for z in fg_z.tolist():
                keep_z.update(range(z - margin, z + margin + 1))  # widen by margin
            kept.extend((fi, sj) for sj in sjs if sj in keep_z)

        if not kept:
            raise RuntimeError("foreground filtering removed every slice -- check the "
                               "preprocessed labels / label_channel")

        # rebuild the index bookkeeping the base class derived from the full list
        self.slices = kept
        self._data = self.slices                                  # generate_train_batch source
        self.slice_idxs = list(range(len(self.slices)))
        self.data_len = len(self.slices)
        self.num_batches = (self.data_len // self.batch_size) + 10
        self.np_data = np.asarray(self.slices)


class NumpyDataSetSpleen(object):
    """Drop-in NumpyDataSet whose inner loader can filter empty slices (train)."""

    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 num_processes=8, num_cached_per_queue=8 * 4, target_size=256,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,),
                 do_reshuffle=True, keys=None, foreground_only=True, margin=3,
                 label_channel=1):
        data_loader = NumpyDataLoaderSpleen(
            base_dir=base_dir, mode=mode, batch_size=batch_size, num_batches=num_batches,
            file_pattern=file_pattern, input_slice=input_slice, label_slice=label_slice,
            keys=keys, foreground_only=foreground_only, margin=margin,
            label_channel=label_channel)

        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1

        self.transforms = get_transforms(mode=mode, target_size=target_size)
        self.augmenter = MultiThreadedDataLoader(data_loader, self.transforms,
                                                 num_processes=num_processes,
                                                 num_cached_per_queue=num_cached_per_queue,
                                                 shuffle=do_reshuffle)
        self.augmenter.restart()

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        if self.do_reshuffle:
            self.data_loader.reshuffle()
        self.augmenter.renew()
        return self.augmenter

    def __next__(self):
        return next(self.augmenter)
