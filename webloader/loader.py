#!/usr/bin/python
#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import logging
import time
import collections

from past.utils import old_div

from . import filters, gopen, paths, utils

_big = 1 << 60

try:
    from torch import Tensor as TorchTensor
except:
    class TorchTensor(object): pass

try:
    from numpy import ndarray
except:
    class ndarray(object): pass

def tonumpy(dtype=None, transpose=True):
    """Curried function to convert to NumPy.

    :param dtype: target dtype (Default value = None)
    :param transpose: transpose from PyTorch to NumPy conventions (Default value = True)

    """
    def f(a):
        """

        :param a: 

        """
        import numpy as np
        if isinstance(a, TorchTensor):
            if a.ndim == 3 and a.shape[0] in [3, 4]:
                a = a.permute(1, 2, 0)
            elif a.ndim == 4 and a.shape[1] in [3, 4]:
                a = a.transpose(0, 2, 3, 1)
            return a.detach().cpu().numpy()
        else:
            return a
    return f

def totorch(dtype=None, device="cpu", transpose=True):
    """Curried conversion to PyTorch.

    :param dtype: desired dtype, None=auto (Default value = None)
    :param device: desired target device (Default value = "cpu")
    :param transpose: transpose images to PyTorch conventions (Default value = True)

    """
    def f(a):
        import torch
        import numpy as np
        if isinstance(a, np.ndarray):
            dtype_ = dtype
            if dtype_ is None:
                if a.dtype in [np.float16, np.float32, np.float64]:
                    dtype_ = torch.float32
                elif a.dtype in [np.int16, np.uint16, np.int32, np.int64]:
                    dtype_ = torch.int64
            elif isinstance(dtype_, str):
                dtype_ = getattr(torch, dtype_)
            if a.ndim == 3 and a.shape[2] in [3, 4]:
                a = a.transpose(2, 0, 1)
            elif a.ndim == 4 and a.shape[3] in [3, 4]:
                a = a.transpose(0, 3, 1, 2)
            if device=="numpy":
                return a
            else:
                return torch.as_tensor(a, device=device, dtype=dtype_)
        else:
            return a
    return f

def transform_with(sample, transformers):
    """Transform a list of values using a list of functions.

    :param sample: list of values
    :param transformers: list of functions

    """
    assert not isinstance(sample, dict)
    assert isinstance(sample, (tuple, list))
    if transformers is None or len(transformers) == 0:
        return sample
    result = list(sample)
    ntransformers = len(transformers)
    for i in range(len(sample)):
        f = transformers[i%ntransformers]
        if f is not None:
            result[i] = f(sample[i])
    return result

def transformer(transformers):
    """Curried version of `transform_with`.

    :param transformers: 

    """
    def f(x): return transform_with(x, transformers)
    return f

def listify(x):
    """Turn a value into a list.

    Lists and tuples are turned into lists, everything else is turned
    into a one element list.

    :param x: value to be converted
    :param x): return transform_with(x: 
    :param transformers)return flistify(x: 

    """
    if x is None:
        return None
    elif isinstance(x, (list, tuple)):
        return x
    else:
        return [x]


converter_table = dict(
    torch=totorch(),
    torch_cuda=totorch(device="cuda"),
    torch_np=totorch(device="numpy"), # torch conventions, NumPy representation
    torch_numpy=totorch(device="numpy"), # torch conventions, NumPy representation
    numpy=tonumpy()
)

class Tracker(object):
    def __init__(self, n=100):
        self.keys = {}
        self.recent = [None] * n
        self.count = 0
    def __call__(self, source):
        for sample in source:
            assert isinstance(sample, dict), sample
            key = sample.get("__key__", None)
            if key is not None:
                self.keys[key] = self.keys.get(key, 0) + 1
            self.recent[self.count % len(self.recent)] = sample
            self.count += 1
            yield sample
    def counts(self):
        return self.keys.values()


class WebLoader(object):
    """Iterate over sharded datasets."""
    def __init__(self,
                 url_or_generator,
                 batches,
                 epochs=1,
                 fields=None,
                 batch_size=None,
                 tensor_batches=True,
                 partial_batches=True,
                 shuffle=0,
                 shardshuffle=None,
                 transforms=None,
                 batch_transforms=None,
                 converters=None,
                 decode=True,
                 pipeline=None,
                 verbose=False,
                 use_tracker=True,
                 use_torch_mp=False,
                 processes=1,
                 queue_size=1):
        """Create a WebLoader

        :param urls: URLs to load from, as a shard spec expanded with braceexpand
        :param size: formal size of the dataset (used by len() function)
        :param fields: fields to extract (Default value = None)
        :param batch_size: batch size to return (Default value = 0, no batching)
        :param partial_batches: return partial batches if necessary (Default value = True)
        :param tensor_batches: stack tensors in batches into tensors (Default value = True)
        :param shuffle: shuffle size (Default value = 0, no shuffle)
        :param shardshuffle: shuffle shards (Default value = None, shuffle shards if shuffling)
        :param transforms: list of functions to apply to unbatched samples (Default value = None)
        :param batch_transforms: list of functions to apply to batched samples (Default value = None)
        :param converters: list of functions to apply after batch_transforms (Default value = None)
        :param decode: decoder to apply to tarfiles (Default value = True)
        :param epochs: number of epochs to iterate for (Default value = 1, only used with sharditerators)
        :param pipeline: pipeline to apply to samples before field extraction (Default value = None)
        :param verbose: output extra information (Default value = False)
        :param use_tracker: ignored (for interface compatiblity with MultiWebLoader)
        :param use_torch_mp: ignored (for interface compatiblity with MultiWebLoader)
        :param processes: ignored (for interface compatiblity with MultiWebLoader)
        :param queue_size: ignored (for interface compatiblity with MultiWebLoader)
        """

        self.debug = int(os.environ.get("DEBUG_WEBLOADER", 0))
        self.shuffle = shuffle
        self.shardshuffle = shardshuffle if shardshuffle is not None else shuffle
        self.url_or_generator = url_or_generator
        self.batches = batches
        self.epochs = epochs
        self.decode = decode
        self.batch_size = batch_size
        self.tensor_batches = tensor_batches
        self.partial_batches = partial_batches
        if isinstance(pipeline, list):
            pipeline = filters.compose(pipeline)
        self.pipeline = pipeline
        self.fields = utils.parse_field_spec(fields) if fields is not None else None
        if fields is None:
            assert transforms is None, "cannot specify transforms without fields"
            assert batch_transforms is None, "cannot specify batch transforms without fields"
            assert converters is None, "cannot specify converters without fields"
        self.transforms = listify(transforms)
        self.batch_transforms = listify(batch_transforms)
        converters = converter_table.get(converters, converters)
        self.converters = listify(converters)
        self.tracker = None
        if use_tracker:
            self.tracker = Tracker()
        self.verbose = verbose
        self.sampler = None # for compatibility with DataLoader

    def __iter__(self):
        """Iterate over samples."""
        finished = False
        self.sample = 0
        self.batch = 0
        if isinstance(self.url_or_generator, str):
            source = gopen.sharditerator(self.url_or_generator,
                                         shuffle=self.shardshuffle,
                                         epochs=self.epochs,
                                         decode=self.decode,
                                         verbose=self.verbose)
        elif hasattr(self.url_or_generator, "__iter__"):
            source = iter(self.url_or_generator)
        else:
            raise ValueError(f"{self.url_or_generator}: not understood as a source")
        if self.tracker is not None:
            source = self.tracker(source)
        if self.pipeline is not None:
            source = self.pipeline(source)
        if self.fields is not None:
            source = filters.extract(*self.fields)(source)
        if self.transforms is not None:
            source = filters.transform(transformer(self.transforms))(source)
        if self.shuffle > 0:
            source = filters.shuffle(self.shuffle)(source)
        if self.batch_size is not None:
            source = filters.batched(self.batch_size,
                                     combine_tensors=self.tensor_batches,
                                     partial=self.partial_batches)(source)
        for sample in source:
            if self.batch_transforms is not None:
                if isinstance(sample, dict):
                    raise ValueError("expect list for batch_transforms; did you specify fields= for WebLoader?")
                sample = transform_with(sample, self.batch_transforms)
            if self.converters is not None:
                if isinstance(sample, dict):
                    raise ValueError("expect list for batch_transforms; did you specify fields= for WebLoader?")
                sample = transform_with(sample, self.converters)
            self.last_sample = sample
            self.batch += 1
            try: self.sample += len(sample[0])
            except: self.sample = -999999
            if self.debug>=200:
                for i, x in enumerate(sample):
                    print(i, self.sample, type(x), repr(x)[:50], x.shape if hasattr(x, "shape") else None)
            elif self.debug==199:
                print("batch", self.batch, "sample", self.sample, "::", repr(sample)[:50])
            yield sample
            if self.batch >= self.batches: break

    def __len__(self):
        """Return the length of the dataset (the size argument passed on initialization)."""
        return self.batches

def make_loader(args, kw, queue, index):
    kw["use_tracker"] = False
    data = WebLoader(*args, **kw)
    for sample in data:
        queue.put(sample)

def maybe_gpu(a, device=None, non_blocking=False):
    if isinstance(a, ndarray):
        import torch
        a = torch.from_numpy(a)
    if isinstance(a, TorchTensor):
        return a.contiguous().to(device=device, non_blocking=non_blocking)
    else:
        return a

def sync_gpu_transfer(device="cuda"):
    def f(source):
        for data in source:
            if isinstance(data, (tuple, list)):
                data = [maybe_gpu(a, device, True) for a in data]
            elif isinstance(data, dict):
                data = {k: maybe_gpu(a, device, True) for k, a in data.items()}
            yield data
    return f

def async_gpu_transfer(device="cuda", inflight=2):
    def f(source):
        q = collections.deque()
        done = False
        while not done:
            while not done and len(q) < inflight:
                try:
                    data = next(source)
                except StopIteration:
                    done = True
                    break
                if isinstance(data, (tuple, list)):
                    data = [maybe_gpu(a, device, True) for a in data]
                elif isinstance(data, dict):
                    data = {k: maybe_gpu(a, device, True) for k, a in data.items()}
                q.append(data)
            yield q.popleft()
            if done and len(q) == 0: break
    return f

multi_pipes = dict(
    sync_gpu_transfer=sync_gpu_transfer(),
    async_gpu_transfer=async_gpu_transfer()
)

class MultiWebLoader(object):
    """Multiprocessing version of WebLoader """
    def __init__(self, urls, batches, epochs=1,
                 processes=4, use_torch_mp=False, queue_size=10, multi_pipe=None, **kw):
        """Instantiate multiple WebLoaders in parallel.

        :param urls: input URLs
        :param batches: formal size of dataset
        :param epochs: number of epochs
        :param processes: number of subprocesses to use (Default value = 4)
        :param use_torch_mp: use the Torch version of multiprocessing (Default value = False)
        :param queue_size: number of samples buffered in the queue (Default value = 10)
        :param **kw: other keyword arguments are passed to WebLoader

        """
        self.batches = batches
        self.epochs = epochs
        # TODO: split up shards among subprocesses
        self.args = (urls, batches)
        self.kw = kw
        self.use_torch_mp = use_torch_mp
        self.processes = processes
        self.queue_size = queue_size
        self.multi_pipe = multi_pipes.get(multi_pipe, multi_pipe)
        assert self.multi_pipe is None or callable(self.multi_pipe)
        self.jobs = None
        self.sampler = None # for compatibility with DataLoader

    def raw_iter(self):
        """Iterate over samples.

        Note that multiple iterators share the same input queue."""
        if self.use_torch_mp:
            import torch.multiprocessing as mp
        else:
            import multiprocessing as mp
        total = 0
        while total < self.batches * self.epochs:
            if self.jobs is None:
                self.queue = mp.Queue(self.queue_size)
                self.jobs = [mp.Process(target=make_loader,
                                        args=(self.args, self.kw, self.queue, i))
                             for i in range(self.processes)]
                for job in self.jobs:
                    job.start()
            try:
                while total < self.batches * self.epochs:
                    sample = self.queue.get()
                    total += 1
                    yield sample
            except FileNotFoundError as exn:
                print("restarting MultiWebLoader jobs", repr(exn)[:100])
                self.terminate()

    def __iter__(self):
        result = self.raw_iter()
        if self.multi_pipe is not None:
            result = self.multi_pipe(result)
        return result

    def __len__(self):
        """ """
        return self.batches

    def terminate(self, soft=False):
        """Terminate all subprocesses"""
        for job in self.jobs:
            try:
                if soft:
                    job.terminate()
                    job.join()
                else:
                    os.kill(job.pid, 15)
                    time.sleep(0.1)
                    os.kill(job.pid, 9)
            except Exception as exn:
                print(job)
                print(exn)
        del self.queue
        self.jobs = None
        self.queue = None

def asdict(l):
    if isinstance(l, dict):
        return l
    return {i: v for i, v in enumerate(l)}

def loader_test(source, nbatches=10, skip=10):
    """Run a test against a loader."""
    for i, sample in enumerate(source):
        if i >= skip-1: break

    start = time.time()
    count = 0
    for i, sample in enumerate(source):
        sample = asdict(sample)
        for xs in sample.values():
            if isinstance(xs, (list, TorchTensor, ndarray)):
                count += len(xs)
                break
        if i >= nbatches-1: break
    finish = time.time()

    delta = finish-start
    print("{:.2f} samples/s {:.2f} batches/s".format(count/delta, nbatches/delta))

    print("Example:")
    sample = asdict(sample)
    for index, a in sorted(sample.items()):
        if isinstance(index, str) and index[0]=="_":
            if isinstance(a, list):
                print(index, a[0], "...")
            else:
                print(index, str(a)[:100], "...")
        elif isinstance(a, TorchTensor):
            print(index, ":", "Tensor", a.shape, a.device, a.dtype, a.min().item(), a.max().item())
        elif isinstance(a, ndarray):
            import numpy as np
            print(index, ":", "ndarray", a.shape, a.dtype, np.amin(a), np.amax(a))
        else:
            print(index, ":", type(a))
