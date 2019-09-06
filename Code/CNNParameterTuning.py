import pandas as pd
import os
import keras
import keras_applications
keras_applications.set_keras_submodules(
    backend=keras.backend,
    layers=keras.layers,
    models=keras.models,
    utils=keras.utils
)
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# -----------------------------LOAD IN OUR DATA---------------------------------- #
TRAIN = 'CheXpert-v1.0-small/train.csv'
VALID = 'CheXpert-v1.0-small/valid.csv'

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
num_valid = len(valid)

#-----------------------------INITIATE_SOME VARS-------------------------------------#


TARGET_SIZE = (224, 224)
BATCH_SIZE = 16
CONV_BASE = 'MobileNetV2'
EPOCHS = 3
OPT = optimizers.Adam(0.00001)
WEIGHTS = 'DenseNet121_24_6_weights_lr_reduce_from32_16.hdf5'
TRAINABLE = True

STEPS_PER_EPOCH = (num_samples/BATCH_SIZE) - 1

if CONV_BASE == 'VGG16':
    from keras.applications.vgg16 import VGG16 as BASE
elif CONV_BASE == 'ResNet152':
    from keras_applications.resnet import ResNet152 as BASE
elif CONV_BASE == 'DenseNet121':
    from keras.applications.densenet import DenseNet121 as BASE
elif CONV_BASE == 'NASNetLarge':
    from keras.applications.nasnet import NASNetLarge as BASE
elif CONV_BASE == 'MobileNetV2':
    from keras.applications.mobilenet_v2 import MobileNetV2 as BASE
else:
    raise ValueError('Unknown model: {}'.format(CONV_BASE))




#-----------------------------SETUP CHECKPOINTS AND MODEL STORAGE-------------------------------------#
save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = "{m}_{b}_{e}_model_lr_reduce_from32_16_8.h5".format(m=CONV_BASE, b=BATCH_SIZE, e=EPOCHS)

weight_path="{m}_{b}_{e}_weights_test_lr_reduce_from32_16_8.hdf5".format(m=CONV_BASE, b=BATCH_SIZE, e=EPOCHS)

checkpoint = ModelCheckpoint(weight_path, monitor='accuracy', verbose=1,
                             save_best_only=True, mode='max', save_weights_only = True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.00001, cooldown = 1)

checkitout = [checkpoint, reduce_lr]


# -----------------------------DATA PREPROCESSING---------------------------------- #
#declare the datagen options
train_datagen = ImageDataGenerator(rescale=1./255)

# set up the test data set
valid_datagen = ImageDataGenerator(rescale=1./255)

#generate training dataset
train_generator = train_datagen.flow_from_dataframe(dataframe=train,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="label",
                                                    class_mode="categorical",
                                                    color_mode="rgb",
                                                    target_size=TARGET_SIZE,
                                                    batch_size=BATCH_SIZE)


valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="label",
                                                    class_mode="binary",
                                                    color_mode="rgb",
                                                    target_size=TARGET_SIZE,
                                                    batch_size=BATCH_SIZE)

# -----------------------------COMPILE THE MODEL---------------------------------- #

conv_base = BASE(include_top=True,
                input_shape=train_generator.image_shape,
                pooling=max)

conv_base.layers.pop()
# print(conv_base.summary())

conv_base.trainable = True
for layer in conv_base.layers:
    layer.trainable = True

model = models.Sequential()
model.add(conv_base)
model.add(layers.Dense(1, activation='sigmoid'))

if WEIGHTS:
    model.load_weights(WEIGHTS)

model.compile(loss='binary_crossentropy', optimizer= OPT, metrics=['accuracy'])
print(model.summary())

# -----------------------------ADD SOME CHECKPOINTS---------------------------------- #

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch= STEPS_PER_EPOCH,
    validation_data=valid_generator,
    validation_steps= num_valid,
    callbacks=checkitout)


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
print(model_path)
model.save(model_path)
print('Saved trained model at %s ' % model_path)