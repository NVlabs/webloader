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


def test_WebLoader_torch_gpu():
    import torch
    wl  = loader.WebLoader("testdata/sample.tgz", 90,
                                 fields="png cls".split(),
                                 batch_size=32,
                                 converters="torch_cuda")
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
    assert sample[0].device.type == "cuda", sample[0].device
    assert sample[1].device.type == "cuda", sample[1].device

def FIXME_test_MultiWebLoader_torch_gpu():
    import torch
    wl  = loader.MultiWebLoader("testdata/sample.tgz", 90,
                                 fields="png cls".split(),
                                 use_torch_mp=True,
                                 batch_size=32,
                                 converters="torch_cuda")
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
    assert sample[0].device.type == "cuda", sample[0].device
    assert sample[1].device.type == "cuda", sample[1].device
    wl.terminate()

def test_MultiWebLoader_torch_sync_gpu():
    import torch
    wl  = loader.MultiWebLoader("testdata/sample.tgz", 90,
                                 multi_pipe="sync_gpu_transfer",
                                 fields="png cls".split(),
                                 batch_size=32,
                                 converters="torch")
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
    assert sample[0].device.type == "cuda", sample[0].device
    assert sample[1].device.type == "cuda", sample[1].device
    wl.terminate()

def test_MultiWebLoader_torch_async_gpu():
    import torch
    wl  = loader.MultiWebLoader("testdata/sample.tgz", 90,
                                 multi_pipe="async_gpu_transfer",
                                 fields="png cls".split(),
                                 batch_size=32,
                                 converters="torch")
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
    assert sample[0].device.type == "cuda", sample[0].device
    assert sample[1].device.type == "cuda", sample[1].device
    wl.terminate()

