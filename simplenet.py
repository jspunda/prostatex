# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.initializers import RandomNormal

from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use('Agg')

from lesion_extraction_2d.lesion_extractor_2d import get_train_data
from utils.auc_callback import AucHistory
from utils.generator_from_config import get_generator

AUGMENTATION_CONFIGURATION = 'more_channel_shift'

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
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])
## Data
h5_file_location = os.path.join('/scratch-shared/ISMI/prostatex','prostatex-train.hdf5')
h5_file = h5py.File(h5_file_location, 'r')
train_data_list, train_labels_list = get_train_data(h5_file, ['ADC'])

data = np.zeros((len(train_data_list),16,16,1), dtype=np.float32)
labels = np.zeros((len(train_labels_list), 1), dtype=np.float32)

for index, image in enumerate(train_data_list):
    data[index, :, :, 0] =image
for index, label in enumerate(train_labels_list):
    labels[index, 0] = label

train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.33, random_state=42, stratify=labels)

## Stuff for training
generator = get_generator(configuration=AUGMENTATION_CONFIGURATION)

train_generator = generator.flow(train_data, train_labels)#, save_to_dir="/nfs/home4/schellev/augmented_images")
batch_size = 128
steps_per_epoch = len(train_labels_list)//batch_size

auc_history = AucHistory(train_data, train_labels, val_data, val_labels, output_graph_name=AUGMENTATION_CONFIGURATION)

model.fit_generator(train_generator, steps_per_epoch, epochs=100, verbose=2, callbacks = [auc_history], max_q_size = 50, workers = 8)


