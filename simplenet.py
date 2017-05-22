# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os
import sys

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.initializers import RandomNormal

from lesion_extraction_2d.lesion_extractor_2d import get_train_data
from utils.auc_callback import AucHistory
from utils.generator_from_config import get_generator
from utils.train_test_split import train_test_split

AUGMENTATION_CONFIGURATION = 'baseline'

## Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation=LeakyReLU(), kernel_initializer='he_normal', bias_initializer=RandomNormal(mean=0.1), input_shape=(16, 16, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2), activation=LeakyReLU(), kernel_initializer='he_normal', bias_initializer=RandomNormal(mean=0.1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2), activation=LeakyReLU(), kernel_initializer='he_normal', bias_initializer=RandomNormal(mean=0.1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (2, 2), activation=LeakyReLU(), kernel_initializer='he_normal', bias_initializer=RandomNormal(mean=0.1)))
model.add(BatchNormalization())

model.add(Conv2D(64, (1, 1), activation=LeakyReLU(), kernel_initializer='he_normal', bias_initializer=RandomNormal(mean=0.1)))
model.add(BatchNormalization())

model.add(Conv2D(1, (1, 1), activation='sigmoid'))
model.add(Flatten())

# For a binary classification problem
sgd = SGD(lr=0.0005, momentum=0.9)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

## Data
h5_file_location = os.path.join('/media/koen/Stack/Stack/uni/Machine Learning in Practice', 'prostatex-train.hdf5')
h5_file = h5py.File(h5_file_location, 'r')
train_data_list, train_labels_list, attr = get_train_data(h5_file, ['ADC'])


def reshape(input_list):
    new_shape = list(input_list.shape)
    new_shape.append(1)

    output_list = np.reshape(input_list, new_shape)

    return output_list.astype(np.float32)


train_data, val_data, train_labels, val_labels = train_test_split(train_data_list, train_labels_list, attr, test_size=0.33)

train_data = reshape(train_data)
val_data = reshape(val_data)
train_labels = reshape(train_labels)
val_labels = reshape(val_labels)

## Stuff for training
generator = get_generator(configuration=AUGMENTATION_CONFIGURATION)

train_generator = generator.flow(train_data, train_labels)  #, save_to_dir="/nfs/home4/schellev/augmented_images")
batch_size = 128
steps_per_epoch = len(train_labels_list) // batch_size

auc_history = AucHistory(train_data, train_labels, val_data, val_labels, output_graph_name=AUGMENTATION_CONFIGURATION)

model.fit_generator(train_generator, steps_per_epoch, epochs=100, verbose=2, callbacks=[auc_history], max_q_size=50, workers=8)
