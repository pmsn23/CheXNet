import pandas as pd
import numpy as np

from keras import models
from keras import layers
from keras import optimizers

from keras_preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121 as BASE

from sklearn.metrics import roc_auc_score, average_precision_score

# -----------------------------LOAD IN OUR DATA---------------------------------- #
VALID = 'CheXpert-v1.0-small/valid.csv'
WEIGHTS = 'DenseNet121_16_6_weights_lr_reduce_from32.hdf5'

# generate validation set
valid = pd.read_csv(VALID, dtype=str)

#mapping to different labels
label = {'0.0': '0', '1.0' : '1', '-1.0' : '1', '0': '0', '1': '1'}

#same for validation set even though there should be no issue here with missing data
valid['label'] = valid['Pleural Effusion'].map(label)
num_valid = len(valid)
valid['label'] = valid['label'].astype('str')

print(valid['label'].value_counts())

#---------------------------------CREATE TEST DATA GENERATOR---------------------------------#
def test_flow(valid, target_size, batch_size = 1, color_mode="grayscale",  x = 'Path', y= 'label'):

    #declare the datagen options
    valid_datagen = ImageDataGenerator(rescale=1./255)

    valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid,
                                                        directory=None,
                                                        x_col=x,
                                                        y_col=y,
                                                        class_mode="binary",
                                                        color_mode=color_mode,
                                                        seed = 42,
                                                        shuffle = False,
                                                        target_size=target_size,
                                                        batch_size=batch_size)

    return valid_generator


valid_generator = test_flow(valid, (224,224), color_mode="rgb")


# # -----------------------------LOAD MODEL---------------------------------- #
conv_base = BASE(include_top=True,
                input_shape=valid_generator.image_shape,
                pooling=max)

conv_base.layers.pop()
# print(conv_base.summary())

# conv_base.trainable = True
# for layer in conv_base.layers:
#     layer.trainable = True

model = models.Sequential()
model.add(conv_base)
model.add(layers.Dense(1, activation='sigmoid'))

if WEIGHTS:
    model.load_weights(WEIGHTS)

model.compile(loss='binary_crossentropy', optimizer= optimizers.Adam(), metrics=['accuracy'])
print(model.summary())

# # -----------------------------EVALUATE MODEL---------------------------------- #
scoreSeg = model.evaluate_generator(valid_generator, steps=len(valid))
print('--------------------------')
print('')
print("Accuracy (Evaluation Generator)= ",scoreSeg[1])
print('')

from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# make a prediction
y_pred_keras = model.predict_generator(valid_generator, steps=len(valid), verbose=1)

#for i in len(y_pred_keras):
#    print("Prediction: {p} || {c}".format(p=str(round(y_pred_keras[i], 4)), c=str(valid_generator.classes[i])))

fpr_keras, tpr_keras, thresholds_keras = roc_curve(valid_generator.classes, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
print('--------------------------')
print('')
print('Model AUC: %s' % str(round(auc_keras, 4)))
print('')
print('--------------------------')

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# pred = model.predict_generator(valid_generator, steps=39)
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

