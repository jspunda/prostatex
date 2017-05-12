# -*- coding: utf-8 -*-
import h5py
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use('Agg')

from lesion_extraction_2d.lesion_extractor_2d import get_train_data
from callbacks.auc_callback import AucHistory


## Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(1, (1, 1), activation='softmax'))
model.add(Flatten())

# For a binary classification problem
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])
## Data
h5_file = h5py.File('prostatex-train.hdf5', 'r')
train_data_list, train_labels_list = get_train_data(h5_file, ['ADC'])

data = np.zeros((len(train_data_list),16,16,1), dtype=np.float32)
labels = np.zeros((len(train_labels_list), 1), dtype=np.float32)

for index, image in enumerate(train_data_list):
    data[index, :, :, 0] =image
for index, label in enumerate(train_labels_list):
    labels[index, 0] = label

train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.33, random_state=42, stratify=labels)

## Stuff for training
generator = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True)

train_generator = generator.flow(train_data, train_labels)#, save_to_dir="/nfs/home4/schellev/augmented_images")
batch_size = 128
steps_per_epoch = len(train_labels_list)//batch_size

auc_history = AucHistory(train_data, train_labels, val_data, val_labels)

model.fit_generator(train_generator, steps_per_epoch, epochs=1000, verbose=2, callbacks = [auc_history], max_q_size = 50, workers = 8)


