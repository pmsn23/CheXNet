import numpy as np
from numpy import genfromtxt
import cv2
import os
import skimage
import string
import random

from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Embedding, SpatialDropout1D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from sklearn.model_selection import StratifiedShuffleSplit 
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D


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

train_subset = train[0:10000]
                   
total_rows = train.shape[0]
# print (train.dtype.names)
# print(os.listdir("CheXpert-v1.0-small/train/patient00001/study1"))
def getdata():
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

def create_model(input_length, vocab_size, num_labels=1):
    #refactored to allow for num_labels
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  
    model.add(Dense(101))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    #model.compile(loss='binary_crossentropy',
    #              optimizer='rmsprop',
    #              metrics=['accuracy'])
                  
    #model = Sequential([Embedding(vocab_size, 32, input_length=input_length,
    #                              dropout=0.2),
    #                    SpatialDropout1D(0.2),
    #                    Dropout(0.25),
    #                    Convolution1D(64, 5, padding='same', activation='relu', input_shape=(224, 224, 3)),
    #                    Dropout(0.25),
    #                    MaxPooling1D(),
    #                    Flatten(),
    #                    Dense(100, activation='relu', input_shape=(224,224,3)),
    #                    Dropout(0.7),
    #                    Dense(num_labels, activation='softmax')])

    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    return model

M, L = getdata()

blank = np.where(L == b"")
#minusone = np.where(L == b'-1.0')
L[blank] = 0.0
#L[minusone] = 0.0

y = L.astype(np.float)
L = y.astype(int)

num_cv_iterations = 10
num_instances = len(L)
cv = StratifiedShuffleSplit(n_splits=num_cv_iterations,test_size=0.25, random_state=0)

for train_indices, test_indices in cv.split(M,L): 
    M_train = M[train_indices]
    L_train = L[train_indices]
    
    M_test = M[test_indices]
    L_test = L[test_indices]


#M_train =  M_train.reshape(7500,3,224,224)
#M_test = M_test.reshape(2500,3,224,224)

#L_train = to_categorical(L_train, 2)
#L_test = to_categorical(L_test, 2)

# construct the model
vocab_size = 10000
clf = create_model(M.shape[1], vocab_size, num_labels=3)
# train model
clf.fit(M_train, L_train, validation_data=(M_test, L_test), epochs=10, batch_size=256)

import pdb; pdb.set_trace()