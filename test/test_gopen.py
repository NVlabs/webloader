#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#
from __future__ import unicode_literals

import pdb
from builtins import next

import numpy as np
from webloader import gopen
import itertools as itt


def test_gopen():
    assert gopen.gopen("/etc/passwd").read().decode()[:4] == "root"


def test_url():
    assert gopen.test_url("/etc/passwd")
    assert not gopen.test_url("/etc/")
    assert not gopen.test_url("/etc/LSKDJFLKJLFJ")
    assert gopen.test_url("http://www.google.com/")
    assert not gopen.test_url("http://www.slkJLKJLFKDJLJ.com/")


def test_find_shards():
    target = "http://storage.googleapis.com/lpr-ocr/uw3-lines.tgz"
    url = gopen.find_shards([
        "http://www.nvidia.com/lpr-ocr/uw3-lines.tgz",
        target
    ])
    assert url == target


def test_sharditerator():
    url = "http://storage.googleapis.com/lpr-ocr/uw3-lines.tgz"
    data = gopen.sharditerator(url)
    for sample in data:
        break
    assert set(["__key__", "png"]) < set(sample.keys()), list(sample.keys())


def test_sharditerator_once():
    url = "http://storage.googleapis.com/lpr-ocr/uw3-lines.tgz"
    data = gopen.sharditerator_once(url)
    for sample in data:
        break
    assert set(["__key__", "png"]) < set(sample.keys()), list(sample.keys())

def test_sharditerator_epochs():
    url = "testdata/imagenet-000000.tgz"
    data = gopen.sharditerator(url, epochs=1)
    total = 0
    for sample in data:
        total += 1
        assert total < 1000
    assert total == 47
    data = gopen.sharditerator(url, epochs=2)
    total2 = 0
    for sample in data:
        total2 += 1
        assert total2 < 1000
    assert total2 == 2*total
