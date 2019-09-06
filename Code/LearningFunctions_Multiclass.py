
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, average_precision_score

# -----------------------------DATA PREPROCESSING---------------------------------- #

def train_flow(train, target_size, batch_size, x = 'Path', y= 'label', class_mode = "binary"):

    #declare the datagen options
    train_datagen = ImageDataGenerator(rescale=1./255
                                       )

    #generate training dataset
    train_generator = train_datagen.flow_from_dataframe(dataframe=train,
                                                        directory=None,
                                                        x_col=x,
                                                        y_col=y,
                                                        class_mode=class_mode,
                                                        color_mode="rgb",
                                                        shuffle=True,
                                                        target_size=target_size,
                                                        batch_size=batch_size)

    return train_generator

# def train_flow_v2(train, target_size, batch_size, x = 'Path', y= 'label'):
#
#     train_datagen = ImageDataGenerator(re)

def test_flow(valid, target_size, batch_size = 1, x = 'Path', y= 'label', class_mode = "binary"):

    #declare the datagen options
    valid_datagen = ImageDataGenerator(rescale=1./255)

    valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid,
                                                        directory=None,
                                                        x_col=x,
                                                        y_col=y,
                                                        class_mode=class_mode,
                                                        color_mode="rgb",
                                                        shuffle=False,
                                                        target_size=target_size,
                                                        batch_size=batch_size)

    return valid_generator


def compile_model(loss, opt, metrics, shape , weights = None, conv_base = 'DenseNet121'):

    if conv_base == 'VGG16':
        from keras.applications.vgg16 import VGG16 as BASE
    elif conv_base == 'ResNet152':
        from keras_applications.resnet import ResNet152 as BASE
    elif conv_base == 'DenseNet121':
        from keras.applications.densenet import DenseNet121 as BASE
    elif conv_base == 'NASNetLarge':
        from keras.applications.nasnet import NASNetLarge as BASE
    elif CONV_BASE == 'MobileNetV2':
        from keras.applications.mobilenet_v2 import MobileNetV2 as BASE
    else:
        raise ValueError('Unknown model: {}'.format(CONV_BASE))

    #load the base layer with everything

    if weights:
        conv_base = BASE(include_top=False,
                         input_shape=shape,
                         pooling=max)

        # conv_base = BASE(include_top=True)

        #pop the top layer off
        # conv_base.layers.pop()
        # print(conv_base.summary())

        #set all the layers to trainable
        conv_base.trainable = True
        for layer in conv_base.layers:
            layer.trainable = True

        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(5, activation='sigmoid'))

        model.load_weights(weights)

        model.compile(loss=loss, optimizer=opt, metrics=metrics)
    else:
        conv_base = BASE(include_top=False,
                         weights=None,
                         input_shape=shape,
                         pooling=max)

        #pop the top layer off
        # conv_base.layers.pop()
        # print(conv_base.summary())

        #set all the layers to trainable
        conv_base.trainable = True
        for layer in conv_base.layers:
            layer.trainable = True

        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(5, activation='sigmoid'))

        model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model


class roc_callback(Callback):
    def __init__(self, validation_data):
        self.x_val = validation_data
        self.y_val = validation_data.labels


    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.auroc = []
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.x_val.reset()
        y_pred_val = self.model.predict_generator(self.x_val, steps=self.x_val.n)
        try:
            roc_val = roc_auc_score(self.y_val, y_pred_val)
        except ValueError:
            pass
        print('--------------------------')
        print('')
        print('AUC: %s' % str(round(roc_val,4)))
        print('')
        print('--------------------------')
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.auroc.append(logs.get('auroc'))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

