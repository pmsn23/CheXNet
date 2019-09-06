import pandas as pd
import os
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.applications.densenet import DenseNet121
from keras.callbacks import ModelCheckpoint


#setting some hyper parameters at the top to play with
batch_size = 20
epochs = 3
opt = optimizers.Adam(lr=1e-4)
steps_per_epoch = 10000


save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'dense121_05919_3Epoch16Step.h5'

weight_path="{}_weights.best.hdf5".format('dense121_3Epoch16Step')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)


#read in the training dataset
train=pd.read_csv("CheXpert-v1.0-small/train.csv", dtype = str)
#generate validation set
valid=pd.read_csv("CheXpert-v1.0-small/valid.csv", dtype = str)


#convert the missing data to zero as nothing means no mention of the effect
train['Pleural Effusion'].loc[train['Pleural Effusion'].isna()] = '0.0'
train = train[train['Pleural Effusion'] != '-1.0']
print(train['Pleural Effusion'].value_counts())
num_samples = len(train)

#same for validation set even though there should be no issue here with missing data
valid['Pleural Effusion'].loc[valid['Pleural Effusion'].isna()] = '0.0'
num_valid = len(train)

print(valid['Pleural Effusion'].value_counts())

#declare the datagen options
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.05,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   shear_range=0.05,
                                   horizontal_flip=True,
                                   fill_mode="nearest")

#generate training dataset
train_generator = train_datagen.flow_from_dataframe(dataframe=train,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="Pleural Effusion",
                                                    class_mode="binary",
                                                    color_mode="rgb",
                                                    target_size=TARGET_SIZE,
                                                    batch_size=batch_size)



#set up the test data set
valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="Pleural Effusion",
                                                    class_mode="binary",
                                                    color_mode="rgb",
                                                    batch_size=batch_size)

# print(dir(train_generator))
print(valid_generator.image_shape)

conv_base = DenseNet121(weights='imagenet',
                        include_top=False,
                        input_shape=train_generator.image_shape,
                        pooling=max)

conv_base.trainable = True

for layer in conv_base.layers:
    layer.trainable = True


# print(conv_base.summary())

####USING PRE_TRAINED MODELS RETRAINING THE TOP LAYERS
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
print(model.summary())

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=20,
    callbacks=[checkpoint])


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)