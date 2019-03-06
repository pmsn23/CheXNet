import numpy as np
from numpy import genfromtxt
import cv2
import os
import skimage
import string
import random
import time

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Embedding, SpatialDropout1D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

batch_size = 32
num_classes = 3
epochs = 10
data_augmentation = True

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cheXnet_trained_model.h5'

def getdata(train_subset):
    M = []
    L = []
    
    for i in range(train_subset.shape[0]):
        img_path = train_subset["Path"][i].decode('UTF-8')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = skimage.transform.resize(img, (224, 224, 3))
            img = np.asarray(img)
            M.append(img)
            L.append(train_subset["Pleural_Effusion"][i])
    
    M = np.asarray(M)
    L = np.asarray(L)
    return M, L

def create_model():
    #refactored to allow for num_labels
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=M_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    loss = 'binary_crossentropy'
    #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    opt = Adam()
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    model.summary()
    
    return model

def model_fit(model, data_augmentation):

'''
    Data Augmentation Options:
    1) rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures.
    2) width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate 
    pictures vertically or horizontally.
    3) shear_range is for randomly applying shearing transformations.
    4) zoom_range is for randomly zooming inside pictures.
    5) horizontal_flip is for randomly flipping half of the images horizontally -- relevant when 
    there are no assumptions of horizontal asymmetry (e.g. real-world pictures).
    6) fill_mode is the strategy used for filling in newly created pixels, which can appear 
    after a rotation or a width/height shift.

'''
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(M_train, L_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(M_test, L_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(M_train)

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(M_train, L_train,
                                         batch_size=batch_size),
                            epochs=epochs,steps_per_epoch=10,
                            validation_data=(M_test, L_test),
                            workers=4)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(M_test, L_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    return history
    
def plot_acc_loss(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    return
#=-=-=-=-=-=- Start -=-=-=-=-=-=-=
tic = time.clock()
types = ["S64", # Path
         "S6",  # Sex
         "i4",  # Age
         "S8",  # Frontal/Lateral
         "S4",  # AP/PA
         "S4",  # No Finding
         "S4",  # Enlarged Cardiomediastinum
         "S4",  # Cardiomegaly
         "S4",  # Lung Capacity
         "S4",  # Lung Lesion
         "S4",  # Edema
         "S4",  # Consolidation      
         "S4",  # Pneumonia
         "S4",  # Atelectasis
         "S4",  # Pneumothorax
         "S4",  # Pleural Effusion
         "S4",  # Pleural Other
         "S4",  # Fracture
         "S4"   # Support Devices
              ]
cat_features = ['Path', 'Sex', 'FrontalLateral', 'APPA',]
num_features = ['Age', 'No_Finding','Enlarged_Cardiomediastinum', 'Cardiomegaly', 
                'Lung_Opacity','Lung_Lesion', 'Edema', 'Consolidation', 'Pneumonia', 
                'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other', 
                'Fracture', 'Support_Devices']
				
train = genfromtxt('CheXpert-v1.0-small/train.csv', dtype=types, delimiter=',', names=True, 
                   usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])

test = genfromtxt('CheXpert-v1.0-small/valid.csv', dtype=types, delimiter=',', names=True, 
                   usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])

train_subset = train[0:20000]
total_rows = train.shape[0]

M_train, L_train = getdata(train_subset)
M_test, L_test = getdata(test)

blank = np.where(L_train == b"")
L_train[blank] = 0.0

y = L_train.astype(np.float)
L_train = y.astype(int)

blank = np.where(L_test == b"")
L_test[blank] = 0.0

y = L_test.astype(np.float)
L_test = y.astype(int)

tic = time.clock()
# M_train, M_test, L_train, L_test = train_test_split(M, L, test_size = 0.33, random_state = 0)

L_train = keras.utils.to_categorical(L_train, num_classes)
L_test = keras.utils.to_categorical(L_test, num_classes)

M_train = M_train.astype('float32')
M_test = M_test.astype('float32')
M_train /= 255
M_test /= 255

model = create_model()
history = model_fit(model, data_augmentation)
plot_acc_loss(history)

toc=time.clock()
print("Total execution time (hrs): ", round((toc-tic)*0.000277777778,4))
    
import pdb; pdb.set_trace()