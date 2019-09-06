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
labelMap = {'0.0': 0, '1.0' : 1, '-1.0' : 1}

#convert the missing data to zero as nothing means no mention of the effect
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
for l in labels:
    train[l].loc[train[l].isna()] = '0.0'
    train[l] = train[l].map(labelMap)
    # print(train[l].value_counts())

train['label'] = train[labels].values.tolist()
num_samples = len(train)


#same for validation set even though there should be no issue here with missing data
for l in labels:
    valid[l].loc[valid[l].isna()] = '0.0'
    valid[l] = valid[l].map(labelMap)
    # print(valid[l].value_counts())

valid['label'] = valid[labels].values.tolist()
num_valid = len(valid)

# -------------------------------------------------------------------------------- #

# -------------------------Process for training Data---------------------------- #
BATCH_SIZE = 16
CONV_BASE = 'DenseNet121'
EPOCHS = 15
# WEIGHTS = 'imagenet'
WEIGHTS = 'DenseNet121_16_3_best_weights_320x320_multilabel.hdf5'


train_generator = train_flow(train = train,
                             target_size = (320,320),
                             y = labels,
                             batch_size = BATCH_SIZE,
                             class_mode = "other")

valid_generator = test_flow(valid = valid,
                            target_size = (320,320),
                            y = labels,
                            class_mode = "other")


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

# def auroc(y_true, y_pred):
#     try:
#         auroc = tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)
#     except ValueError:
#         pass
#
#     return auroc

model = compile_model(loss = "binary_crossentropy",
                      weights = WEIGHTS,
                      opt = optimizers.Adam(lr=0.01),
                      metrics = ["accuracy"],
                      conv_base = 'DenseNet121',
                      shape = train_generator.image_shape)

print('-----------------------------------------')
print('Model Summary for Training')
print(model.summary())
print('-----------------------------------------')


#-----------------------------SETUP CHECKPOINTS, TRAIN MODEL, & STORAGE-------------------------------------#

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = "{m}_{b}_{e}_model_320x320_multilabel.h5".format(m=CONV_BASE, b=BATCH_SIZE, e=EPOCHS)

weight_path="{m}_{b}_{e}_best_weights_320x320_multilabel.hdf5".format(m=CONV_BASE, b=BATCH_SIZE, e=EPOCHS)

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.00001)

# roc = roc_callback(validation_data=valid_generator)

checkitout = [checkpoint, reduce_lr]


history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch= STEPS_PER_EPOCH,
    validation_data=valid_generator,
    validation_steps= VALID_STEPS,
    callbacks=checkitout)


# scoreSeg = model.evaluate_generator(valid_generator, steps=1)
# print('--------------------------')
# print('')
# print("Accuracy (Evaluation Generator)= ",scoreSeg[1])
# print('')
#
# # pred = model.predict_generator(valid_generator, steps=1)
# # pr_val = average_precision_score(valid_generator.labels, pred)
# # roc_val = roc_auc_score(valid_generator.labels, pred)
# #
# # print('--------------------------')
# # print('')
# # print('Average Precision: %s' % str(round(pr_val, 4)))
# # print('')
# # print('--------------------------')
# # print('')
# # print('Model AUC: %s' % str(round(roc_val, 4)))
# # print('')
# # print('--------------------------')
#
# # n=3
# # auroc_hist = np.asarray(train_history.auroc).ravel()
# # top_auroc = auroc_hist[np.argsort(auroc_hist)[-n:]]
# # print('--------------------------')
# # print('')
# # print('Average AUROC: %s' % str(round(np.mean(top_auroc), 4)))
# # print('')
# # print('--------------------------')
# # print('--------------------------')
# # print('')
# # print('Std Dev AUROC: %s' % str(round(np.std(top_auroc), 4)))
# # print('')
# # print('--------------------------')
# # print('--------------------------')
# # print('')
# # print('Max AUROC: %s' % str(round(np.max(top_auroc), 4)))
# # print('')
# # print('--------------------------')
#
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
#
# model_path = os.path.join(save_dir, model_name)
# print(model_path)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)
