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

import os
import pickle
import random

from utilities.file_and_folder_operations import subfiles

def create_splits(output_dir, image_dir, seed=42, k=5):
    """
    True seeded k-fold cross-validation splits.
    Every case appears in the test set of exactly one fold.
    for fold i, test = chunk i, val = chunk (i+1) % k, train = rest
    """
    npy_files = subfiles(image_dir, suffix=".npy", join=False)
    samples = sorted(s[:-4] for s in npy_files)        # deterministic base order
    if len(samples) < k:
        raise ValueError(f"{k}-fold CV needs at least {k} samples, have {len(samples)}")

    rng = random.Random(seed)                          # isolated, seeded RNG
    rng.shuffle(samples)

    # k disjoint, near-equal chunks, their union is every case
    chunks = [samples[i::k] for i in range(k)]

    splits = []
    for i in range(k):
        test = sorted(chunks[i])
        val = sorted(chunks[(i + 1) % k])
        used = set(test) | set(val)
        train = sorted(s for s in samples if s not in used)
        splits.append({'train': train, 'val': val, 'test': test})

    # every case is tested exactly once across the k folds
    all_test = [s for sp in splits for s in sp['test']]
    assert sorted(all_test) == sorted(samples), "k-fold test coverage broken"
    assert len(all_test) == len(set(all_test)), "a case is in >1 test fold"

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)
