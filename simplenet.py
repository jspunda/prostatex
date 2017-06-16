# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os

SEED=1337
np.random.seed(SEED)

from keras.models import Model, Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras.layers.core import Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

from lesion_extraction_2d.lesion_extractor_2d import get_train_data
from utils.auc_callback import AucHistory
from utils.generator_from_config import get_generator
from utils.train_test_split import train_test_split
from data_visualization.adc_lesion_values import apply_window

def get_model(configuration='baseline'):
    AUGMENTATION_CONFIGURATION = configuration

    ## Model
    image_inputs= Input(shape=(32, 32, 3))

    adc = Lambda(lambda x: x[:,:,:,0:1])(image_inputs)# , output_shape=(128,) + input_shape[2:]
    t2_tra = Lambda(lambda x: x[:,:,:,1:2])(image_inputs)
    t2_sag =Lambda(lambda x: x[:,:,:,2:3])(image_inputs)
    #combo = Lambda(lambda x: x[:,:,:,0:2])(image_inputs)

    parallel_model = Sequential()
    parallel_model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal',input_shape=(32,32,2)))
    parallel_model.add(LeakyReLU())
    parallel_model.add(BatchNormalization())
    
    parallel_model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal'))
    parallel_model.add(LeakyReLU())
    parallel_model.add(BatchNormalization())
    parallel_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    parallel_model.add(Conv2D(32, (2, 2), kernel_initializer='he_normal'))
    parallel_model.add(LeakyReLU())
    parallel_model.add(BatchNormalization())
    
    parallel_model.add(Conv2D(32, (2, 2), kernel_initializer='he_normal'))
    parallel_model.add(LeakyReLU())
    parallel_model.add(BatchNormalization())
    parallel_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    parallel_model.add(Conv2D(32, (2, 2), kernel_initializer='he_normal'))
    parallel_model.add(LeakyReLU())
    parallel_model.add(BatchNormalization())
    
    parallel_model.add(Conv2D(32, (2, 2), kernel_initializer='he_normal'))
    parallel_model.add(LeakyReLU())
    parallel_model.add(BatchNormalization())
    parallel_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    parallel_model.add(BatchNormalization())
    parallel_model.add(Conv2D(64, (2, 2), kernel_initializer='he_normal'))
    parallel_model.add(LeakyReLU())
    parallel_model.add(BatchNormalization())
    
    #combo = parallel_model(combo)
    ## adc part
    adc = parallel_model(adc)
    
    #does this prevent weight sharing? is this even necessary?
    json_string  = parallel_model.to_json()
    parallel_model2 = model_from_json(json_string)
    
    #does this prevent weight sharing? is this even necessary?
    parallel_model3 = model_from_json(json_string)
    
    #t2_tra part
    t2_tra = parallel_model2(t2_tra)
    
    #t2_sag part
    t2_sag = parallel_model3(t2_sag)
    
    #merge stuff  back together
    network = concatenate([adc, t2_tra])

    network = Conv2D(32, (1, 1), kernel_initializer='he_normal')(network)
    network = LeakyReLU()(network)
    network = BatchNormalization()(network)
    
    network = Conv2D(32, (1, 1), kernel_initializer='he_normal')(network)
    network = LeakyReLU()(network)
    network = BatchNormalization()(network)

    network = Conv2D(1, (1, 1), activation='sigmoid')(network)
    predictions = Flatten()(network)
    model = Model(inputs=[image_inputs], outputs=predictions)

    # For a binary classification problem
    sgd = SGD(lr=0.0005, momentum=0.9)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    ## Data
    h5_file_location = os.path.join('/scratch-shared/ISMI/prostatex', 'prostatex-train.hdf5')
    h5_file = h5py.File(h5_file_location, 'r')
    
    data, labels, attr = get_train_data(h5_file, ['ADC', 't2_tse_tra', 't2_tse_sag'], size_px=32, size_mm=24)
    # , 't2_tse_tra', 't2_tse_sag'
    
    print(data.shape)
    for index in range(data.shape[0]):
        data[index,:,:,0:1] = apply_window(data[index,:,:,0:1], (400, 1100))
    
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, attr, test_size=0.33, random_state=SEED)

    # train_data = np.expand_dims(train_data, axis=-1)
    # val_data = np.expand_dims(val_data, axis=-1)
    train_labels = np.expand_dims(train_labels, axis=-1)
    val_labels = np.expand_dims(val_labels, axis=-1)

    ## Stuff for training
    generator = get_generator(configuration=AUGMENTATION_CONFIGURATION)

    batch_size = 128
    train_generator = generator.flow(train_data, train_labels, batch_size=batch_size, seed=SEED)#, save_to_dir="/nfs/home4/schellev/16pixels")
    steps_per_epoch = len(labels) // batch_size
    
    train_generator.next()

    def schedule(epoch):
        if epoch <10:
            return 0.05
        elif epoch <30:
            return 0.01
        elif epoch <100:
            return 0.005
        elif epoch < 300:
            return 0.002
        elif epoch < 600:
            return 0.001
        elif epoch < 1000:
            return 0.001
        else:
            return 0.00005
            
    auc_history = AucHistory(train_data, train_labels, val_data, val_labels, output_graph_name=AUGMENTATION_CONFIGURATION)
    lr_schedule = LearningRateScheduler(schedule)

    model.fit_generator(train_generator, steps_per_epoch, epochs=500, verbose=2, callbacks=[auc_history, lr_schedule], max_q_size=50, workers=8)

    return model

if __name__ == "__main__":
    m = get_model(configuration="channel_shift_0")
