#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Data loader for the LOSS-EXPERIMENT track -- ported from the thanasis pipeline.
#
# Key differences vs the existing datasets/two_dim/NumpyDataLoader.py (untouched):
#   * NATIVE-resolution patches: train/val/test crop or pad at full resolution
#     instead of resizing the whole slice to target_size (which erases thin vessels
#     on large CT). The U-Net is fully-convolutional, so full slices go straight in.
#   * class-balanced FOREGROUND oversampling (nnU-Net style): a fraction (fg_fraction)
#     of train crops are centred on a randomly chosen foreground class, so rare
#     classes (tumour) reach parity instead of being drowned by background.
#   * mmap per-slice reads: only the needed slice's bytes are materialised (big win
#     for large CT npy), not the whole volume.
#   * full-slice val/test: each slice is padded to a multiple of the U-Net pool
#     factor (16) and inferred whole; batches pad to the batch-max to stack.
#
# MODIFICATION vs thanasis (as agreed): on a foreground-oversampled crop the target
# pixel is placed at a RANDOM position inside the patch (not the centre), so training
# does not learn a "the target is always centred" prior. See _crop_origin.
#

import os
import fnmatch
import random

import numpy as np

from batchgenerators.dataloading import SlimDataLoaderBase
from datasets.data_loader import MultiThreadedDataLoader
from .data_augmentation_loss import get_transforms


def load_dataset(base_dir, pattern='*.npy', slice_offset=5, keys=None):
    fls = []
    files_len = []
    slices_ax = []

    for root, dirs, files in os.walk(base_dir):
        i = 0
        for filename in sorted(fnmatch.filter(files, pattern)):

            if keys is not None and filename[:-4] in keys:
                npy_file = os.path.join(root, filename)
                numpy_array = np.load(npy_file, mmap_mode="r")

                fls.append(npy_file)
                files_len.append(numpy_array.shape[1])

                slices_ax.extend([(i, j) for j in range(slice_offset, files_len[-1] - slice_offset)])

                i += 1

    return fls, files_len, slices_ax,


class NumpyDataSet(object):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000, num_processes=8,
                 num_cached_per_queue=8 * 4, target_size=128, file_pattern='*.npy', label_slice=1,
                 input_slice=(0,), do_reshuffle=True, keys=None, fg_fraction=0.33):

        data_loader = NumpyDataLoader(base_dir=base_dir, mode=mode, batch_size=batch_size, num_batches=num_batches,
                                      file_pattern=file_pattern, input_slice=input_slice, label_slice=label_slice,
                                      keys=keys, target_size=target_size, fg_fraction=fg_fraction)

        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1

        self.transforms = get_transforms(mode=mode, target_size=target_size)
        self.augmenter = MultiThreadedDataLoader(data_loader, self.transforms, num_processes=num_processes,
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


class NumpyDataLoader(SlimDataLoaderBase):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,), keys=None,
                 target_size=64, fg_fraction=0.33, pad_factor=16):

        self.files, self.file_len, self.slices = load_dataset(base_dir=base_dir, pattern=file_pattern, slice_offset=0, keys=keys, )
        super(NumpyDataLoader, self).__init__(self.slices, batch_size, num_batches)

        self.batch_size = batch_size
        self.mode = mode
        self.target_size = target_size      # patch size for train/val crops
        self.fg_fraction = fg_fraction      # fraction of train crops centred on foreground
        self.pad_factor = pad_factor        # test/val full-slice pad multiple (U-Net pool factor)

        self.use_next = False
        if mode == "train":
            self.use_next = False

        self.slice_idxs = list(range(0, len(self.slices)))
        self.data_len = len(self.slices)
        self.num_batches = min((self.data_len // self.batch_size) + 10, num_batches)

        if isinstance(label_slice, int):
            label_slice = (label_slice,)
        self.input_slice = input_slice
        self.label_slice = label_slice

        self.np_data = np.asarray(self.slices)

        # slice-level class-balanced oversampling (train only): class -> slice-indices containing it
        self.class_slices, self.fg_classes = {}, []
        if mode == "train":
            self._build_class_index()

    def _build_class_index(self):
        """Map each foreground class -> indices (into self.slices) of slices that contain it, so a
        training batch can oversample rare classes (e.g. tumour) to parity, not by frequency. Each
        volume's label channel is read once via mmap to keep startup cheap."""
        from collections import defaultdict
        by_case = defaultdict(list)
        for idx, (ci, z) in enumerate(self.slices):
            by_case[int(ci)].append((idx, int(z)))
        lc = self.label_slice[0]
        cls = defaultdict(list)
        for ci, items in by_case.items():
            lbl = np.load(self.files[ci], mmap_mode="r")[lc]          # label volume (D0, D1, D2)
            for idx, z in items:
                for c in np.unique(lbl[z]):
                    if c > 0:
                        cls[int(c)].append(idx)
        self.class_slices = {c: v for c, v in cls.items() if v}
        self.fg_classes = list(self.class_slices.keys())
        print("class-balanced sampling -> " +
              ", ".join(f"class {c}: {len(v)} slices" for c, v in sorted(self.class_slices.items())), flush=True)

    def reshuffle(self):
        print("Reshuffle...")
        random.shuffle(self.slice_idxs)
        print("Initializing... this might take a while...")

    def generate_train_batch(self):
        open_arr = random.sample(self._data, self.batch_size)
        return self.get_data_from_array(open_arr)

    def __len__(self):
        n_items = min(self.data_len // self.batch_size, self.num_batches)
        return n_items

    def __getitem__(self, item):
        slice_idxs = self.slice_idxs
        data_len = len(self.slices)
        np_data = self.np_data

        if item > len(self):
            raise StopIteration()
        if (item * self.batch_size) == data_len:
            raise StopIteration()

        start_idx = (item * self.batch_size) % data_len
        stop_idx = ((item + 1) * self.batch_size) % data_len

        if ((item + 1) * self.batch_size) == data_len:
            stop_idx = data_len

        if stop_idx > start_idx:
            idxs = slice_idxs[start_idx:stop_idx]
        else:
            raise StopIteration()

        open_arr = np_data[idxs]

        return self.get_data_from_array(open_arr)

    def get_data_from_array(self, open_array):
        data = []
        fnames = []
        slice_idxs = []
        labels = []

        for slice in open_array:
            # class-balanced foreground oversampling (nnU-Net style): for a fraction of TRAIN samples
            # draw a random foreground class and a slice that CONTAINS it, then crop around it. Choosing
            # class -> slice -> location samples rare classes (tumour) to parity, at both slice and pixel
            # level; the rest use the uniformly-drawn slice with a random crop.
            force_class = None
            if self.mode == "train" and self.class_slices and random.random() < self.fg_fraction:
                force_class = random.choice(self.fg_classes)
                slice = self.slices[random.choice(self.class_slices[force_class])]

            fn_name = self.files[slice[0]]

            # memory-map: read only this slice's bytes from disk, not the whole volume
            numpy_array = np.load(fn_name, mmap_mode="r")
            numpy_slice = np.asarray(numpy_array[:, slice[1], ])   # materialise the one slice
            img = numpy_slice[list(self.input_slice)]                                     # (C_in, H, W)
            seg = numpy_slice[list(self.label_slice)] if self.label_slice is not None else None

            # native-resolution patch (train/val) or full padded slice (test); all samples come out a
            # fixed size, so batches are homogeneous by construction
            img, seg = self._sample(img, seg, force_class)

            data.append(img)
            if seg is not None:
                labels.append(seg)
            fnames.append(self.files[slice[0]])
            slice_idxs.append(slice[1])

        # full-slice batches (val/test with batch>1) can hold volumes of different H,W; pad each up to
        # the batch-max so they stack (image with its min, label/aux with 0). No-op when uniform.
        if len(data) > 1:
            hmax = max(d.shape[-2] for d in data)
            wmax = max(d.shape[-1] for d in data)
            if any((d.shape[-2], d.shape[-1]) != (hmax, wmax) for d in data):
                def _pad(a, cval):
                    _, h, w = a.shape
                    dh, dw = hmax - h, wmax - w
                    t, l = dh // 2, dw // 2
                    return np.pad(a, ((0, 0), (t, dh - t), (l, dw - l)), constant_values=cval)
                data = [_pad(d, d.min()) for d in data]
                if labels:
                    labels = [_pad(s, 0) for s in labels]

        ret_dict = {'data': np.asarray(data), 'fnames': fnames, 'slice_idxs': slice_idxs}
        if self.label_slice is not None:
            ret_dict['seg'] = np.asarray(labels)

        return ret_dict

    def _pad_to(self, img, seg, ph, pw):
        """Pad (C,H,W) up to at least (ph, pw), centred; image with its min, label/aux with 0."""
        _, h, w = img.shape
        dh, dw = max(ph - h, 0), max(pw - w, 0)
        if dh or dw:
            t, l = dh // 2, dw // 2
            width = ((0, 0), (t, dh - t), (l, dw - l))
            img = np.pad(img, width, mode="constant", constant_values=img.min())
            if seg is not None:
                seg = np.pad(seg, width, mode="constant", constant_values=0)
        return img, seg

    def _sample(self, img, seg, force_class=None):
        ps = self.target_size
        if self.mode == "test":
            # full slice at native resolution, padded up to a multiple of the U-Net factor
            f = self.pad_factor
            _, h, w = img.shape
            return self._pad_to(img, seg, ((h + f - 1) // f) * f, ((w + f - 1) // f) * f)
        # train / val: guarantee at least a patch, then crop ps x ps at native resolution
        img, seg = self._pad_to(img, seg, ps, ps)
        _, h, w = img.shape
        if self.mode == "train":
            top, left = self._crop_origin(seg, h, w, ps, force_class)
        else:                                         # non-test patch mode: deterministic centre crop
            top, left = (h - ps) // 2, (w - ps) // 2
        img = img[:, top:top + ps, left:left + ps]
        if seg is not None:
            seg = seg[:, top:top + ps, left:left + ps]
        return img, seg

    def _crop_origin(self, seg, h, w, ps, force_class=None):
        """Top-left of a ps x ps crop. If force_class is given (foreground-oversampled sample) put a
        pixel of that class at a RANDOM position inside the patch -- NOT the centre -- so the model
        does not learn a centring prior; the clip keeps that pixel inside the patch even at the border.
        Otherwise a plain random crop. The foreground draw + class choice happen once, at the slice
        level in get_data_from_array, so there is no second draw here."""
        if force_class is not None and seg is not None:
            px = np.argwhere(seg[0] == force_class)
            if len(px):
                cy, cx = px[random.randint(0, len(px) - 1)]
                oy = random.randint(0, ps - 1)                        # target's position within the patch
                ox = random.randint(0, ps - 1)
                return (int(np.clip(cy - oy, 0, max(h - ps, 0))),
                        int(np.clip(cx - ox, 0, max(w - ps, 0))))
        top = random.randint(0, h - ps) if h > ps else 0
        left = random.randint(0, w - ps) if w > ps else 0
        return top, left
