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

import random
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    """Seed random/numpy/torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init(worker_id, base_seed):
    """Top-level worker initialiser."""
    set_seed(worker_id + base_seed)


class WrappedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.transform = transform
        self.dataset = dataset

        self.is_indexable = False
        if hasattr(self.dataset, "__getitem__") and not (hasattr(self.dataset, "use_next") and self.dataset.use_next is True):
            self.is_indexable = True

    def __getitem__(self, index):

        if not self.is_indexable:
            item = next(self.dataset)
        else:
            item = self.dataset[index]
        item = self.transform(**item)
        return item

    def __len__(self):
        return int(self.dataset.num_batches)


class MultiThreadedDataLoader(object):
    def __init__(self, data_loader, transform, num_processes, **kwargs):

        self.cntr = 1
        self.ds_wrapper = WrappedDataset(data_loader, transform)

        self.generator = DataLoader(self.ds_wrapper, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                                    num_workers=num_processes, pin_memory=torch.cuda.is_available(), drop_last=False,
                                    persistent_workers=num_processes > 0,
                                    worker_init_fn=self.get_worker_init_fn())

        self.num_processes = num_processes
        self.iter = None

    def get_worker_init_fn(self):
        return partial(_worker_init, base_seed=self.cntr)

    def __iter__(self):
        self.iter = iter(self.generator)
        return self.iter

    def __next__(self):
        if self.iter is None:
            self.iter = iter(self.generator)
        return next(self.iter)

    def renew(self):
        self.cntr += 1
        self.iter = iter(self.generator)

    def restart(self):
        pass
        # self.iter = iter(self.generator)

    def kill_iterator(self):
        if self.iter is None:
            return

        try:
            shutdown = getattr(self.iter, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            pass
        finally:
            self.iter = None
