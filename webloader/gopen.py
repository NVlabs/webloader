#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#
from __future__ import absolute_import, print_function

import os
import sys
import random
from builtins import range
from io import open
from subprocess import PIPE, Popen, check_call
import logging
import tarfile
import time
import subprocess

from future import standard_library
from future.moves.urllib.parse import urlparse

from . import paths, tarrecords

standard_library.install_aliases()

warn_pipe_exit = False

def test_curl_write(self, location):
    """Test whether we can write to a location using curl."""
    proc = Popen(["curl", "--fail", "-s", "-T", "-", location], stdin=PIPE)
    try:
        proc.stdin.close()
        if proc.wait() != 0:
            raise Exception("{}: cannot write location".format(location))
        check_call(["curl", "--fail", "-X", "DELETE", location])
    finally:
        proc.wait()

gopen_buffer = os.environ.get("GOPEN_BUFFER", "")
gopen_download = os.environ.get("GOPEN_DOWNLOAD", "")
gopen_fatal = int(os.environ.get("GOPEN_FATAL", "0"))
gopen_verbose = int(os.environ.get("GOPEN_VERBOSE", "0"))

#activated_service_account = False
#gs_key_file = os.environ.get("GS_KEY_FILE", None)

def gopen(url, mode="rb"):
    """Open the given URL. Supports unusual schemes and uses subprocesses."""
    parsed = urlparse(url)
    def pipe(cmd, mode):
        if int(os.environ.get("GOPEN_VERBOSE", 0)) > 0:
            print("gopen", cmd, mode)
        if mode=="w":
            #stream = Popen(cmd, stdin=PIPE, shell=True).stdin
            proc = Popen(cmd, stdin=PIPE, shell=True)
            stream = proc.stdin
            stream.pipe_proc = proc
        elif mode=="r":
            #stream = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).stdout
            proc = Popen(cmd, stdout=PIPE, shell=True)
            stream = proc.stdout
            stream.pipe_proc = proc
        stream.pipe_cmd = cmd
        return stream
    if parsed.scheme == "gs":
        if mode[0]=="r":
            return pipe("gsutil -q cat '%s'" % url, "r")
        elif mode[0]=="w":
            return pipe("gsutil -q cp - '%s'" % url, "w")
        else:
            raise ValueError("{}: unknown mode".format(mode))
    elif parsed.scheme in "http https ftp".split():
        if mode[0]=="r":
            if gopen_download != "":
                os.system("rm -f '{}'".format(gopen_download))
                cmd = "curl '%s' --output '%s'" % (url, gopen_download)
                if os.system(cmd) != 0:
                    raise ValueError("cannot download {} to {}".format(
                        url, gopen_download))
                return open(gopen_download, "rb")
            else:
                cmd = "curl --fail -s '%s' --output -" % url
                if gopen_buffer != "":
                    cmd = cmd + " | " + gopen_buffer
                return pipe(cmd, "r")
        elif mode[0]=="w":
            test_curl_write(url)
            return pipe("curl --fail -s -T - '%s'" % url, "w")
        else:
            raise ValueError("{}: unknown mode".format(mode))
    elif parsed.scheme in ["", "file"]:
        if mode[0] == "r":
            if not os.path.exists(parsed.path):
                raise ValueError("{}: not readable".format(parsed.path))
            return open(parsed.path, "rb")
        elif mode[0] == "w":
            return open(parsed.path, "wb")
        else:
            raise ValueError("{}: unknown mode".format(mode))


def close_fully(stream):
    if stream is not None:
        stream.close()
        if hasattr(stream, "pipe_proc"):
            returncode = stream.pipe_proc.wait()
            stream.pipe_proc = None
            return returncode
        else:
            return 0

def test_url(url, size=17):
    """Test whether the given URL is accessible."""
    stream = None
    try:
        stream = gopen(url)
        data = stream.read(size)
        if len(data) == size:
            return True
        return False
    except Exception as e:
        print(e)
        return False
    finally:
        close_fully(stream)


def test_shards(url, size=17, complete=False):
    """Test whether the shard spec is accessible."""
    shards = list(paths.path_shards(url))
    if complete:
        return all(test_url(s, size=size) for s in shards)
    else:
        return test_url(shards[0], size=size)


def find_shards(urls, size=17, complete=False):
    """Given a list of shard URLs, find the first one that exists."""
    for url in urls:
        if test_shards(url, size=size, complete=False):
            return url


def verify_shards(url, report=print, nrecords=1, verbose=False, **kw):
    """Test whether the shard spec is accessible."""
    shards = sorted(list(paths.path_shards(url)))
    for count, shard in enumerate(shards):
        stream = None
        try:
            stream = gopen(shard)
            for i, sample in enumerate(tarrecords.tariterator(stream, **kw)):
                if i >= nrecords:
                    break
            if verbose==True or (isinstance(verbose, int) and count%verbose==0):
                print(count, shard, "OK", flush=True)
        except Exception as exn:
            print(count, shard, str(exn).replace("\n", " ")[:50], flush=True)
        finally:
            assert close_fully(stream)==0

def sharditerator(url, epochs=1000000, shuffle=True, maxerrs=10, verbose=gopen_verbose, **kw):
    """Iterate over sharded tar records."""
    debug = int(os.environ.get("DEBUG_SHARDITERATOR", 0))
    shards = list(paths.path_shards(url))
    if verbose: 
        print("sharditerator over", len(shards), "shards:", str(shards)[:20], flush=True)
    for epoch in range(epochs):
        if debug: print("SHARDITERATOR", "epoch", epoch)
        if shuffle:
            random.shuffle(shards)
        errs = 0
        if debug: print("SHARDITERATOR", "#shards", len(shards))
        for shard in shards:
            if debug: print("SHARDITERATOR", "shard", shard)
            if verbose: print("starting", shard, flush=True)
            stream = None
            try:
                total = 0
                stream = gopen(shard)
                for sample in tarrecords.tariterator(stream, **kw):
                    errs = max(0, errs-1)
                    sample["__source__"] = shard
                    total += 1
                    yield sample
                if debug: print("SHARDITERATOR", "total", total)
            except tarrecords.TarError as exn:
                logging.error("TarError: {}".format(str(exn)))
                if gopen_fatal or errs>maxerrs:
                    raise exn
                errs += 1
                time.sleep(1.0)
            except Exception as exn:
                logging.exception("unexpected exception during shard iterator")
                if gopen_fatal or errs>maxerrs:
                    raise exn
                errs += 1
                time.sleep(1.0)
            finally:
                status = close_fully(stream)
                if status!=0 and warn_pipe_exit:
                    print("exit", status, "for:", getattr(stream, "pipe_cmd"))
            if errs >= maxerrs:
                raise RuntimeError("Number of exceptions exceeded the limit!")
            if verbose: print("finishing", shard, flush=True)


def sharditerator_multi(url, epochs=1000000, shuffle=True, multi=1, **kw):
    """Iterate over sharded tar records, opening multiple shards in parallel."""
    assert multi == 1, "multi>1 is unimplemented"  # FIXME
    return sharditerator(url, epochs=epochs, shuffle=shuffle, **kw)


def sharditerator_once(url, **kw):
    """Iterate over sharded tar records (no shuffling, one epoch only)."""
    return sharditerator(url, epochs=1, shuffle=False, **kw)


def shardwriter(url, encode=True, pack=True):
    parsed = urlparse(url)
    stream = gopen(url, "wb")
    return tarrecords.TarWriter(stream, encode=encode)
