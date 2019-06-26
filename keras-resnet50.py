#!/usr/bin/python3

import sys
import os
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras import models, layers
import numpy as np

from keras.utils.training_utils import multi_gpu_model
import numpy as np
import webloader as wl

from keras.applications.resnet50 import ResNet50

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
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--val-batch-size", type=int, default=64)
parser.add_argument("--processes", type=int, default=4)
parser.add_argument("--keras-processes", type=int, default=1)
parser.add_argument("--model", default="resnet50")
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--devices", action="store_true")
args = parser.parse_args()

if args.devices:
    from tensorflow.python.client import device_lib
    for item in device_lib.list_local_devices():
        print(item.name)
    sys.exit(0)
else:
    from tensorflow.python.client import device_lib
    names = [item.name for item in device_lib.list_local_devices()]
    if "/device:GPU:0" not in names:
        print("----------------------------------------------------------------")
        print("WARNING")
        print("No /device:GPU:0 found, using CPU instead, things will be slow.")
        print("If you have a GPU, try running with wldocker instead of virtualenv.")
        print("List devices with the --devices option.")
        print("----------------------------------------------------------------")

mean = np.array([0.485, 0.456, 0.406], "f")
std = np.array([0.229, 0.224, 0.225], "f")
def norm_image(xs):
    return (xs-mean[None,None,None,:])/std[None,None,None,:]
def norm_cls(ys):
    return keras.utils.to_categorical(ys-1, 1000)

print(f"\nTraining {args.training_urls}\n")
num_batches = args.training_size//args.batch_size
loader = wl.WebLoader if args.processes < 2 else wl.MultiWebLoader
training = loader(args.training_urls, args.training_size,
                       fields="ppm;png;jpg cls",
                       shuffle=1000,
                       batch_transforms=[norm_image, norm_cls],
                       batch_size=args.batch_size,
                       processes=args.processes)
wl.loader_test(training)

print(f"\nVal {args.val_urls}\n")
test_num_batches = args.val_size//args.val_batch_size

testing = wl.WebLoader(args.val_urls, args.val_size,
                       fields="ppm;png;jpg cls",
                       batch_transforms=[norm_image, norm_cls],
                       batch_size=args.val_batch_size,
                       epochs=-1)
wl.loader_test(testing)


print(f"\nModel\n")

model = ResNet50()

if args.gpus>1:
    model = multi_gpu_model(model, gpus=args.gpus)

opt = keras.optimizers.rmsprop(lr=args.lr, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit_generator(iter(training),
              epochs=1,
              steps_per_epoch=num_batches, #num_batches,
              validation_data=iter(testing),
              validation_steps=test_num_batches, #test_num_batches,
              shuffle=False,
              use_multiprocessing=(args.keras_processes>0),
              workers=args.keras_processes)
