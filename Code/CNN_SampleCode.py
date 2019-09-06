import string
import random
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Embedding, SpatialDropout1D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D

# Standard off the shelf
def create_model(input_length, vocab_size, num_labels=1):
    #refactored to allow for num_labels
    model = Sequential([Embedding(vocab_size, 32, input_length=input_length,
                                  dropout=0.2),
                        SpatialDropout1D(0.2),
                        Dropout(0.25),
                        Convolution1D(64, 5, padding='same', activation='relu'),
                        Dropout(0.25),
                        MaxPooling1D(),
                        Flatten(),
                        Dense(100, activation='relu'),
                        Dropout(0.7),
                        Dense(num_labels, activation='softmax')])
    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    return model

def generate_changes(a_string, start, stop):
    char_loc = random.randint(start, stop)
    character = a_string[char_loc]
    new_char = chr(ord(character)-1)
    a_string[char_loc] = new_char
    return a_string

# generate raw sythentic data in 3 distributions
strings = []
for r in range(3): # one loop for each label
    for _ in range(1000): # 1000 examples of each label
        strings.append(list(string.printable))
        for _ in range(50): # 50 perterbations within distribtuion per string
            strings[-1] = generate_changes(strings[-1], (r*25), ((r+2)*25-1))
        for _ in range(100): # noisy perterbations across string
            strings[-1] = generate_changes(strings[-1], 0, 99)

# here we have three different distribtuions of slightly changed strings, with more noise added
# think of this as three types of strings, the first has changes in the first half, the second is
# in the middle half and the third at the last half.  ie 0-50,25-75,50-100,  addtionally noise has
# been added to make the problem more complicated.

# Construct Labels for our three distribtuions.
L = np_utils.to_categorical([0 for x in range(1000)] +
                            [1 for x in range(1000)] +
                            [2 for x in range(1000)])

# to convert our data set to an appropriate matrix representation
input_size = max([len(x) for x in strings]) # lenth of longest string
unique_chars = list(set([x for y in strings for x in y])) # set of possible chars in string

# construct empty data matrix
M = np.zeros((len(strings), input_size))

# fill data matrix with data
for idx, d in enumerate(strings):
    for idy, _ in enumerate(d):
        M[idx, idy] = unique_chars.index(d[idy])

# construct the model
clf = create_model(M.shape[1], len(unique_chars), num_labels=3)

# setup a random test / train index with 75% train, 25% test
train = random.sample(range(len(strings)), round((len(strings)*.75)))
test = list(set(range(len(strings))).difference(set(train)))

# train model
clf.fit(M[train], L[train], validation_data=(M[test], L[test]), epochs=10, batch_size=256)
# import pdb; pdb.set_trace()