import numpy as np
from model_trainer import run_training
import backbone_models
import model_lr
import argparse
import tensorflow as tf
import os
import imageio as io
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', dest='model')
parser.add_argument('-lr', dest='lr')
parser.add_argument('-n','--name', dest='name')
parser.add_argument('-e','--epochs', dest='epochs')
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-mg','--memorygrow', dest='mem_growth',
                    action='store_true',default=False)
args = parser.parse_args()

if args.mem_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

with open('meta.json','r') as f:
    label_dict = json.load(f)

with open('id_to_name.json','r') as f:
    id_to_name = json.load(f)

with open('val_label.json','r') as f:
    val_labels = json.load(f)


train_dir = '/home/jaentrouble/data/imagenet/train'
val_dir = '/home/jaentrouble/data/imagenet/valid'

model_f = getattr(backbone_models, args.model)
lr_f = getattr(model_lr, args.lr)
name = args.name
epochs = int(args.epochs)
mixed_float = args.mixed_float

kwargs = {}
kwargs['model_f'] = model_f
kwargs['lr_f'] = lr_f
kwargs['name'] = name
kwargs['epochs'] = epochs
kwargs['batch_size'] = 32
kwargs['train_dir'] = train_dir
kwargs['label_dict'] = label_dict
kwargs['val_dir'] = val_dir
kwargs['val_labels'] = val_labels
kwargs['id_to_name'] = id_to_name
kwargs['img_size'] = (240,320)
kwargs['mixed_float'] = mixed_float
kwargs['notebook'] = False

run_training(**kwargs)