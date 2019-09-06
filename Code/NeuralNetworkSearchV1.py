import pandas as pd
import os
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import ModelCheckpoint

#-----------------------------INITIATE_CONFIG-------------------------------------#

TRAIN = 'CheXpert-v1.0-small/train.csv'
VALID = 'CheXpert-v1.0-small/valid.csv'

TARGET_SIZE = (320,320)
BATCH_SIZE = 16
CONV_BASE = DenseNet121
EPOCHS = 3
OPT = optimizers.Adam(lr=1e-4)
WEIGHTS = 'imagenet'
TRAINABLE = True
POOLING = avg

if CONV_BASE == 'VGG16':
    from keras.applications.vgg16 import VGG16
elif CONV_BASE == 'ResNet152':
    from keras.applications.resnet import ResNet152
elif CONV_BASE == 'DenseNet121':
    from keras.applications.densenet import DenseNet121
elif CONV_BASE == 'NASNetLarge':
    from keras.applications.nasnet import NASNetLarge
else:
    raise ValueError('Unknown model: {}'.format(CONV_BASE))

# -----------------------------LOAD IN OUR DATA---------------------------------- #

# read in the training dataset
train = pd.read_csv(TRAIN, dtype=str)
# generate validation set
valid = pd.read_csv(VALID, dtype=str)

#convert the missing data to zero as nothing means no mention of the effect
train['Pleural Effusion'].loc[train['Pleural Effusion'].isna()] = '0.0'
train = train[train['Pleural Effusion'] != '-1.0']
num_samples = len(train)

#same for validation set even though there should be no issue here with missing data
valid['Pleural Effusion'].loc[valid['Pleural Effusion'].isna()] = '0.0'
num_valid = len(train)

# -----------------------------DATA PREPROCESSING---------------------------------- #
#declare the datagen options
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.05,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   shear_range=0.05,
                                   horizontal_flip=True,
                                   fill_mode="nearest")

# set up the test data set
valid_datagen = ImageDataGenerator(rescale=1. / 255)

#generate training dataset
train_generator = train_datagen.flow_from_dataframe(dataframe=train,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="Pleural Effusion",
                                                    class_mode="binary",
                                                    color_mode="rgb",
                                                    target_size=TARGET_SIZE,
                                                    batch_size=BATCH_SIZE)


valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="Pleural Effusion",
                                                    class_mode="binary",
                                                    color_mode="rgb",
                                                    target_size=TARGET_SIZE,
                                                    batch_size=BATCH_SIZE)

# -----------------------------COMPILE THE MODEL---------------------------------- #

conv_base = CONV_BASE(weights=WEIGHTS,
                        include_top=False,
                        input_shape=train_generator.image_shape,
                        pooling=POOLING)

conv_base.trainable = TRAINABLE

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
print(model.summary())