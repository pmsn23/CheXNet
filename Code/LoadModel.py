import os

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, EarlyStopping
from keras import backend as K

from sklearn.metrics import roc_auc_score, average_precision_score
from LearningFunctions import train_flow, test_flow, compile_model, roc_callback
from keras.models import load_model

# -----------------------------LOAD IN OUR DATA---------------------------------- #
TRAIN = 'CheXpert-v1.0-small/train.csv'
VALID = 'CheXpert-v1.0-small/valid.csv'

# read in the training dataset
train = pd.read_csv(TRAIN, dtype=str)
# generate validation set
valid = pd.read_csv(VALID, dtype=str)

#mapping to different labels
label = {'0.0': '0', '1.0' : '1', '-1.0' : '1'}

#convert the missing data to zero as nothing means no mention of the effect
train['Pleural Effusion'].loc[train['Pleural Effusion'].isna()] = '0.0'
train = train[train['Pleural Effusion'] != '-1.0']
train['label'] = train['Pleural Effusion'].map(label)
num_samples = len(train)

print(train['label'].value_counts())

#same for validation set even though there should be no issue here with missing data
valid['label'] = valid['Pleural Effusion'].map(label)
num_valid = len(valid)

BATCH_SIZE = 16
CONV_BASE = 'Xception'
EPOCHS = 9
WEIGHTS = None

print(valid['label'].value_counts())

train_generator = train_flow(train, (320,320), BATCH_SIZE)
valid_generator = test_flow(valid, (320,320))

model = load_model('MobileNetV2_16_6_best_weights_320x320_v2.hdf5')

pred = model.predict_generator(valid_generator, steps=int(len(valid_generator.labels)))
pr_val = average_precision_score(valid_generator.labels, pred)
roc_val = roc_auc_score(valid_generator.labels, pred)

print('--------------------------')
print('')
print('Average Precision: %s' % str(round(pr_val, 4)))
print('')
print('--------------------------')
print('')
print('Model AUC: %s' % str(round(roc_val, 4)))
print('')
print('--------------------------')