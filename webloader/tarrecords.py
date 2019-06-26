#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#
from __future__ import absolute_import, division, print_function

import codecs
import getpass
import io
import os
import os.path
import re
import socket
import sys
import tarfile
import time
import warnings
from builtins import object

import six
from future import standard_library
from past.utils import old_div

from . import utils

standard_library.install_aliases()

errors_are_fatal = False


class TarError(Exception):
    pass


if sys.version_info[0] == 3:
    from builtins import str
    buffer = str
    unicode = str


def splitallext(path):
    """Helper method that splits off all extension.
    
    Returns base, allext.

    :param path: path with extensions
    :returns: path with all extensions removed

    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def base_plus_ext(fname):
    """Splits pathnames into the file basename plus the extension.

    :param fname: file name

    """
    return splitallext(fname)


def dir_plus_file(fname):
    """Splits pathnames into the dirname plus the filename.

    :param fname: file name

    """
    return os.path.split(fname)


def last_dir(fname):
    """Splits pathnames into the last dir plus the filename.

    :param fname: file name

    """
    dirname, plain = os.path.split(fname)
    prefix, last = os.path.split(dirname)
    return last, plain


def trivial_decode(sample):
    """Decode samples to strings/buffers/unicode as needed.

    :param sample: dictionary of values

    """
    result = {}
    for k, v in list(sample.items()):
        if isinstance(v, buffer):
            v = str(v)
        elif isinstance(v, unicode):
            # If it is a unicode string, return utf-8 encoded version
            v = str(codecs.encode(v, "utf-8"))
        else:
            if v is not None:
                # It has to bytes string in Python 3. Or simple str which contains bytes in Python 2.
                assert isinstance(v, (bytes, str))
        result[k] = v
    return result


def valid_sample(sample):
    """Check whether a sample is valid.

    :param sample: sample to be checked

    """
    return (sample is not None and
            isinstance(sample, dict) and
            len(list(sample.keys())) > 0 and
            not sample.get("__bad__", False))


def group_by_keys(keys=base_plus_ext, lcase=True):
    """Returns function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (Default value = base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)

    """
    def iterator(data):
        """

        :param data: 

        """
        current_sample = None
        for fname, value in data:
            prefix, suffix = keys(fname)
            if prefix is None:
                continue
            if current_sample is not None and prefix == current_sample["__key__"]:
                current_sample[suffix] = value
                continue
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix)
            if lcase:
                suffix = suffix.lower()
            current_sample[suffix] = value
        if valid_sample(current_sample):
            yield current_sample
    return iterator


def tardata(fileobj, skip_meta=r"__[^/]*__($|/)"):
    """Iterator yielding filename, content pairs for the given tar stream.

    :param fileobj: byte stream suitable for tarfile
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")

    """
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        if not tarinfo.isreg():
            continue
        fname = tarinfo.name
        if fname is None:
            continue
        if skip_meta is not None and re.match(skip_meta, fname):
            continue
        data = stream.extractfile(tarinfo).read()
        yield fname, data
    del stream


def decoder(decode=True):
    """Apply tariterator-like decoding to the stream of samples.

    :param decode: decoder function, or False=trivial, True=defaults (Default value = True)

    """
    if decode is True:
        decode = utils.autodecode
    elif decode is False:
        decode = trivial_decode

    def iterator(data):
        """

        :param data: 

        """
        for sample in data:
            yield decode(sample)
    return iterator


def tariterator1(fileobj, check_sorted=False, keys=base_plus_ext, decode=True):
    """Alternative (new) implementation of tariterator.

    :param fileobj: 
    :param check_sorted:  (Default value = False)
    :param keys:  (Default value = base_plus_ext)
    :param decode:  (Default value = True)

    """
    content = tardata(fileobj)
    samples = group_by_keys(keys=keys)(content)
    decoded = decoder(decode=decode)(samples)
    return decoded


def zipdata(fname):
    """Iterator yielding filename, content pairs for the given zip file.

    :param fname: ZIP file name

    """
    import zipfile
    zf = zipfile.ZipFile(fname)
    fnames = sorted(zf.namelist())
    for fname in fnames:
        data = zf.open(fname).read()
        yield fname, data


def zipiterator(fname, check_sorted=False, keys=base_plus_ext, decode=True):
    """Iterate over ZIP Files.

    :param fname: ZIP file name
    :param check_sorted: check that file names are sorted (Default value = False)
    :param keys: key+extension extraction function (Default value = base_plus_ext)
    :param decode: decoding function (Default value = True)

    """
    content = zipdata(fname)
    samples = group_by_keys(keys=keys)(content)
    decoded = decoder(decode=decode)(samples)
    return decoded


def maybe_decode(current_sample, decode, current_count=None, info=""):
    """Try to decode a sample, warning about failures.

    :param current_sample: sample to decode
    :param decode: decoding functions
    :param current_count: record count (for reporting) (Default value = None)
    :param info: additional info (for error reporting) (Default value = "")

    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            decoded = decode(current_sample)
            return decoded
    except Exception as exn:
        warnings.warn("decoding error {} at {} key {} {}".format(
            exn,
            current_count,
            current_sample.get("__key__")),
            info)
        return None

def tariterator(fileobj, check_sorted=False, keys=base_plus_ext, decode=True,
                source=None, lcase=True, filename=None, fatal=None):
    """Iterate over samples from a tar archive, either locally or given by URL.

    Tar archives are assumed to be sorted by file name. For each basename,
    reads all the files with different extensions and returns a dictionary
    with the extension as key and the file contents as value.

    :param fileobj: string or file object referring to tar file
    :param bool: check_sorted: verify that file names are sorted
    :param keys: key extraction function (Default value = base_plus_ext)
    :param decode: decoding function (Default value = True)
    :param lcase: convert suffixes to lower case before using as dictionary keys (Default value = True)
    :param source: data source (for error reporting) (Default value = None)
    :param filename: file name (for error reporting) (Default value = None)
    :param fatal: flag indicating whether decoding errors should be fatal (Default value=False)
    :returns: iterator over samples

    TODO: unify source and filename?
    TODO: create pluggable decoder (in utils.decode)

    """
    if fatal is None:
        fatal = errors_are_fatal
    if filename is None:
        filename = getattr(fileobj, "pipe_cmd", getattr(fileobj, "name", "?"))
    if decode is True:
        decode = utils.autodecode
    elif decode is False:
        decode = trivial_decode
    elif isinstance(decode, str):
        decode = utils.autodecoder(decode)
    current_count = 0
    current_prefix = None
    current_sample = None
    try:
        stream = tarfile.open(fileobj=fileobj, mode="r|*")
    except tarfile.ReadError:
        raise TarError(filename, "cannot open")
    for tarinfo in stream:
        if not tarinfo.isreg():
            continue
        fname = tarinfo.name
        if fname is None:
            warnings.warn("tarinfo.name is None")
            continue
        prefix, suffix = keys(fname)
        if prefix is None:
            warnings.warn("prefix is None for: %s" % (tarinfo.name,))
            continue
        if prefix != current_prefix:
            if check_sorted and prefix <= current_prefix:
                raise TarError(filename, "keys not sorted")
            if valid_sample(current_sample):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        decoded = decode(current_sample)
                    yield decoded
                except Exception as exn:
                    if fatal:
                        raise ValueError(filename, "decoding error", current_prefix) from exn
                    else:
                        warnings.warn("{}: decoding error".format(current_prefix))
            current_prefix = prefix
            current_sample = dict(__key__=prefix, __source__=source)
        try:
            data = stream.extractfile(tarinfo).read()
        except tarfile.ReadError as exn:
            raise TarError(filename, "read error at", tarinfo.name, current_count)
        except Exception as exn:
            raise TarError(filename, "unexpected exception", tarinfo.name, current_count) from exn
        else:
            if lcase:
                suffix = suffix.lower()
            current_sample[suffix] = data
            current_count += 1
    if valid_sample(current_sample):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                decoded = decode(current_sample)
            yield decoded
        except Exception as exn:
            if fatal:
                raise ValueError(filename, "decoding error", current_prefix) from exn
            else:
                warnings.warn("{}: decoding error".format(current_prefix))


class TarWriter(object):
    """ """
    def __init__(self, fileobj, keep_meta=False, encode=True, user=None, group=None):
        """A class for writing dictionaries to tar files.

        :param fileobj: fileobj: file name for tar file (.tgz)
        :param bool: keep_meta: keep fields starting with "_"
        :param function: encoder: encoding of samples prior to writing
        :param keep_meta:  (Default value = False)
        :param encode:  (Default value = True)
        :param user:  (Default value = None)
        :param group:  (Default value = None)

        """
        if isinstance(fileobj, str):
            fileobj = open(fileobj, "wb")
        if encode is True:
            encode = utils.autoencode
        elif encode is False:
            def encode(x): return x
        self.keep_meta = keep_meta
        self.encode = encode
        self.stream = fileobj
        self.tarstream = tarfile.open(fileobj=fileobj, mode="w:gz")
        self.user = user or getpass.getuser()
        self.group = group or socket.gethostname()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def close(self):
        """Close the tar file."""
        self.finish()

    def finish(self):
        """Close the tar file."""
        self.tarstream.close()
        if self.stream:
            self.stream.close()
            time.sleep(0.3)
            if hasattr(self.stream, "pipe_proc"):
                returncode = self.stream.pipe_proc.wait()
                self.stream.pipe_proc = None
                assert returncode==0, \
                    (returncode, getattr(self.stream, "pipe_cmd"))
            self.stream = None

    def write(self, obj):
        """Write a dictionary to the tar file.

        :param obj: dictionary of objects to be stored
        :returns: size of the entry

        """
        total = 0
        obj = self.encode(obj)
        assert "__key__" in obj, "object must contain a __key__"
        for k, v in list(obj.items()):
            if k[0] == "_":
                continue
            if sys.version_info[0] == 2:
                assert isinstance(
                    v, str), "{} doesn't map to a string after encoding ({})".format(k, type(v))
            else:
                assert isinstance(
                    v, bytes), "{} doesn't map to a bytes after encoding ({})".format(k, type(v))
        key = obj["__key__"]
        for k in sorted(obj.keys()):
            if not self.keep_meta and k[0] == "_":
                continue
            v = obj[k]
            if sys.version_info[0] == 2:
                assert isinstance(v, (str, buffer)),  \
                    "converter didn't yield a string: %s" % ((k, type(v)),)
            else:
                assert isinstance(v, (bytes)),  \
                    "converter didn't yield bytes: %s" % ((k, type(v)),)
            now = time.time()
            ti = tarfile.TarInfo(key + "." + k)
            ti.size = len(v)
            ti.mtime = now
            ti.mode = 0o666
            ti.uname = "bigdata"
            ti.gname = "bigdata"
            # Since, you are writing to file, it should be of type bytes
            assert isinstance(v, bytes), type(v)
            stream = six.BytesIO(v)
            self.tarstream.addfile(ti, stream)
            total += ti.size
        return total


class ShardWriter(object):
    """ """
    def __init__(self, pattern, maxcount=100000, maxsize=3e9, keep_meta=False, encode=True, user=None, group=None):
        """Like TarWriter but splits into multiple shards.

        :param pattern: 
        :param maxcount:  (Default value = 100000)
        :param maxsize:  (Default value = 3e9)
        :param keep_meta:  (Default value = False)
        :param encode:  (Default value = True)
        :param user:  (Default value = None)
        :param group:  (Default value = None)

        """
        self.verbose = 1
        self.args = dict(keep_meta=keep_meta, encode=encode,
                         user=user, group=group)
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.tarstream = None
        self.shard = 0
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.next_stream()

    def next_stream(self):
        """ """
        if self.tarstream is not None:
            self.tarstream.close()
        self.fname = self.pattern % self.shard
        if self.verbose:
            print("# writing", self.fname, self.count, "%.1f GB" %
                  (old_div(self.size, 1e9)), self.total)
        self.shard += 1
        stream = open(self.fname, "wb")
        self.tarstream = TarWriter(stream, **self.args)
        self.count = 0
        self.size = 0

    def write(self, obj):
        """

        :param obj: 

        """
        if self.tarstream is None or self.count >= self.maxcount or self.size >= self.maxsize:
            self.next_stream()
        size = self.tarstream.write(obj)
        self.count += 1
        self.total += 1
        self.size += size

    def close(self):
        """ """
        self.tarstream.close()
        del self.tarstream
        del self.shard
        del self.count
        del self.size
