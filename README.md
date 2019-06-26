WebLoader
=========

WebLoader is a DataLoader for PyTorch and iterator for Keras suitable
for simple, efficient, large scale, distributed training jobs.

WebLoader:

- can read data sets from the file system, web servers, and object store
- performs large, sequential reads
- performs distributed, parallel I/O across many disks
- requires no "volume plugins" or "file system mounts"
- requires no local disk space or caches
- operates directly on POSIX tar/tgz archives of training data
- comes with command line tools for large scale data transformations
- comes with command line tools for benchmarking and testing data sources

Getting Started
===============

Make sure you have `python3` installed.

Then, you can setup your environment with `./setup-venv`

Afterwards, you can start up the notebook server with `./start-notebook`
and look at the example notebooks.

Imagenet Training
=================

You can run the example programs with:

    $ python3 pytorch-resnet50.py
    $ python3 keras-resnet50.py

Or, inside a Docker container:

    $ wldocker python3 pytorch-resnet50.py
    $ wldocker python3 keras-resnet50.py

These train on pre-augmented data stored in Google Cloud; you
can copy that to local disk and train on local files, or you
can use a web cache and http proxy using the `http_proxy`
argument to make accessing the data over the web transparent.

Data Transformations
====================

WebLoader provides a couple of command line tools for data transformation:

- `wl-map` permits Python-based transformations of samples
- `wl-xargs` permits shell script based transformations of samples

Each of these support multicore processing

Examples:

    # subtract 1 from each class
    $ wl-map -c '_["cls"] -= 1' http://server/input.tgz -o output.tgz

    # convert PNG samples to JPEG samples
    $ wl-xargs -c 'convert sample.png sample.jpeg && rm sample.png' gs://bucket1/input.tgz -o gs://bucket2/output.tgz

    # perform edge detection using 4 cores
    $ wl-xargs -P 4 -c 'mogrify -edge 5 sample.jpeg' gs://bucket1/input.tgz -o gs://bucket2/output.tgz

Test Tools
==========

Note that datasets are simply tar files, so you can use standard tools for
extraction and testing, e.g.:

    $ gsutil cat gs://bucket/data.tgz | tar -ztvf - | sed 10q

- wl-cat -- output a single sample from a shard
- wl-extract -- extract data from a shard and outputin tsv or JSON format
- wl-dump -- dump selected files to the local file system (like `tar x` but with additional features)
- wl-info -- provide summary info about tensors/images stored in a dataset
- wl-test -- test performance of a data source

AIStore
=======

The WebLoader library is designed to work with [AIStore](http://github.com/NVIDIA/AIstore),
a high performance web server/web cache/object store specifically built
for big data and deep learning applications.

Tensorcom
=========

Another companion project is Tensorcom, a tool for scalable distributed on-the-fly
data augmentation and RDMA transfer of data into the GPU.

Documentation
=============

(TBD)
