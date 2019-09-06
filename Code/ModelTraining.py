import os

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, EarlyStopping
from keras import backend as K

from sklearn.metrics import roc_auc_score, average_precision_score
from LearningFunctions import train_flow, test_flow, compile_model, roc_callback

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

print(valid['label'].value_counts())

# -------------------------------------------------------------------------------- #

# -------------------------Process for training Data---------------------------- #
#set some model hyperparameters
BATCH_SIZE = 16
CONV_BASE = 'NASNetMobile'
EPOCHS = 15
WEIGHTS = None
OPT_START = optimizers.Adam(lr=0.01)


train_generator = train_flow(train, (320,320), BATCH_SIZE)
valid_generator = test_flow(valid, (320,320))

# STEPS_PER_EPOCH = int(len(train_generator.labels)/BATCH_SIZE)
STEPS_PER_EPOCH = 4800
VALID_STEPS = valid_generator.n

print('-----------------------------------------')
print('Batched training shapes')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

print('Batched valid shapes')
for data_batch, labels_batch in valid_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
print('-----------------------------------------')

def auroc(y_true, y_pred):
    try:
        auroc = tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)
    except ValueError:
        pass

    return auroc

model = compile_model(loss = "binary_crossentropy",
                      opt = OPT_START,
                      metrics = ["accuracy"],
                      weights = WEIGHTS,
                      conv_base = CONV_BASE,
                      shape = train_generator.image_shape)

print('-----------------------------------------')
print('Model Summary for Training')
print(model.summary())
print('-----------------------------------------')


#-----------------------------SETUP CHECKPOINTS, TRAIN MODEL, & STORAGE-------------------------------------#

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = "{m}_{b}_{e}_model_320x320_v2.h5".format(m=CONV_BASE, b=BATCH_SIZE, e=EPOCHS)

weight_path="{m}_{b}_{e}_best_weights_320x320_v2.hdf5".format(m=CONV_BASE, b=BATCH_SIZE, e=EPOCHS)

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.00001)

roc = roc_callback(validation_data=valid_generator)

checkitout = [checkpoint, reduce_lr, roc]


history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch= STEPS_PER_EPOCH,
    validation_data=valid_generator,
    validation_steps= VALID_STEPS,
    callbacks=checkitout,
    verbose=1)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
print(model_path)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# pred = model.predict_generator(valid_generator, steps=int(len(valid_generator.labels)))
# pr_val = average_precision_score(valid_generator.labels, pred)
# roc_val = roc_auc_score(valid_generator.labels, pred)
#
# print('--------------------------')
# print('')
# print('Average Precision: %s' % str(round(pr_val, 4)))
# print('')
# print('--------------------------')
# print('')
# print('Model AUC: %s' % str(round(roc_val, 4)))
# print('')
# print('--------------------------')