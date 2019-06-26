#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#
from __future__ import unicode_literals

import glob
import pdb
from builtins import range
from imp import reload
from io import open

import numpy as np
from webloader import  loader
import itertools as itt


def test_webloader():
    ds = loader.WebLoader("testdata/imagenet-000000.tgz", 90, fields=["png;jpg", "cls"])
    for i, s in enumerate(ds):
        if i>3: break
        assert isinstance(s[0], np.ndarray)

def test_WebLoader_keys():
    wl  = loader.WebLoader("testdata/sample.tgz", 90, fields=None)
    for sample in wl :
        break
    print("sample", sample)
    assert set(["__key__", "png"]) < set(sample.keys()), list(sample.keys())
    assert isinstance(sample["png"], np.ndarray)
    the_key = sample["__key__"]
    for sample in wl :
        break
    assert the_key == sample["__key__"], (the_key, sample["__key__"])

def test_WebLoader_basic_decode():
    wl  = loader.WebLoader("testdata/sample.tgz", 90, decode="rgb8", fields="png cls".split())
    for sample in wl :
        break
    assert len(sample) == 2
    assert isinstance(sample[0], np.ndarray)
    assert sample[0].dtype == np.uint8
    assert isinstance(sample[1], int)

def test_WebLoader_two_tiers():
    wl  = loader.WebLoader("testdata/sample.tgz", 90, fields="png cls".split())
    cls1 = [sample[1] for sample in itt.islice(wl , 0, 10)]
    cls2 = [sample[1] for sample in itt.islice(wl , 0, 10)]
    assert cls1 == cls2, (cls1, cls2)

def test_WebLoader_epoch_end():
    wl  = loader.WebLoader("testdata/sample.tgz", 90, fields="png cls".split(), epochs=1)
    cls = [sample[1] for sample in wl ]
    assert len(cls) == 90
    cls = [sample[1] for sample in wl ]
    assert len(cls) == 90

def test_WebLoader_batch():
    import torch
    wl  = loader.WebLoader("testdata/sample.tgz", 90,
                                 fields="png cls".split(),
                                 batch_size=32)
    for sample in wl :
        break
    assert len(sample) == 2
    assert isinstance(sample[0], np.ndarray), sample[0]
    assert isinstance(sample[1], np.ndarray), sample[1]
    assert sample[0].dtype == np.float32, sample[0].dtype
    assert sample[1].dtype == np.int64, sample[1].dtype
    assert len(sample[0].shape) == 4, sample[0].shape
    assert len(sample[1].shape) == 1, sample[1].shape
    assert sample[0].shape[0] == 32, sample[0].shape
    assert sample[1].shape[0] == 32, sample[1].shape
    assert sample[0].shape[3] == 3, sample[0].size()

def test_WebLoader_torch():
    import torch
    wl  = loader.WebLoader("testdata/sample.tgz", 90,
                                 fields="png cls".split(),
                                 batch_size=32,
                                 converters=loader.totorch())
    for sample in wl :
        break
    assert len(sample) == 2
    assert isinstance(sample[0], torch.Tensor), sample[0]
    assert isinstance(sample[1], torch.Tensor), sample[1]
    assert sample[0].dtype == torch.float32, sample[0]
    assert sample[1].dtype == torch.int64, sample[1]
    assert len(sample[0].shape) == 4, sample[0].shape
    assert len(sample[1].shape) == 1, sample[1].shape
    assert sample[0].shape[0] == 32, sample[0].shape
    assert sample[1].shape[0] == 32, sample[1].shape
    assert sample[0].shape[1] == 3, sample[0].size()

def test_loader_test():
    wl  = loader.WebLoader("testdata/sample.tgz", 90,
                                 fields="png cls".split(),
                                 batch_size=32)
    loader.loader_test(wl)
