#
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# This file is part of webloader (see TBD).
# See the LICENSE file for licensing terms (BSD-style).
#
import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torchvision import models

import webloader as wl
import torchtrainers as tt

torch.set_num_threads(4)

parser = argparse.ArgumentParser("""
Train a model on the Imagenet dataset.
""")
parser.add_argument("--training-urls", 
                    default="http://storage.googleapis.com/lpr-imagenet-augmented/imagenet_train-{0000..0147}-{000..019}.tgz")
parser.add_argument("--training-size", 
                    type=int,
                    default=1000000)
parser.add_argument("--val-urls",
                    default="http://storage.googleapis.com/lpr-imagenet-augmented/imagenet_val-0000-000.tgz")
parser.add_argument("--val-size",
                    type=int,
                    default=50000)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--processes", type=int, default=4)
parser.add_argument("--shuffle", type=int, default=1000)
parser.add_argument("--model", default="resnet50")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--silent", action="store_true")
args = parser.parse_args()

mean = np.array([0.485, 0.456, 0.406], "f")
std = np.array([0.229, 0.224, 0.225], "f")

def norm_image(xs):
    return (xs-mean[None,None,None,:])/std[None,None,None,:]

def norm_cls(ys):
    ys = ys.astype(np.int64)
    return ys-1

print(f"\nTraining {args.training_urls}\n")
loader = wl.WebLoader if args.processes < 2 else wl.MultiWebLoader
training = loader(
    args.training_urls, args.training_size,
    fields="ppm;png;jpg cls",
    batch_transforms=[norm_image, norm_cls],
    batch_size=args.batchsize,
    converters="torch",
    shuffle=args.shuffle,
    verbose=False,
    use_torch_mp=True,
    queue_size=200,
    processes=4)
if not args.silent:
    wl.loader_test(training)

print(f"\nVal {args.val_urls}\n")
testing = wl.WebLoader(
    args.val_urls, args.val_size,
    fields="ppm;png;jpg cls",
    batch_transforms=[norm_image, norm_cls],
    batch_size=args.batchsize,
    epochs=1,
    converters="torch")
if not args.silent:
    wl.loader_test(testing)

print(f"\nModel {args.model}\n")
model = eval(f"models.{args.model}")()
if args.verbose:
    print(model)
model = model.to("cuda")

criterion = nn.CrossEntropyLoss().cuda()
trainer = tt.Trainer(model=model,
                     criterion=criterion,
                     metrics=[tt.Misclassification],
                     device="cuda")

print("\nTraining\n")
tt.training(trainer, training, 1000000, testing, 50000, epochs=100,
            learning_rates=[(0,1e-2), (300000, 1e-3)])
