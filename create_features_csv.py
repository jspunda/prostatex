import os
import h5py

import numpy as np
from keras.models import load_model, Model
from keras import backend as K

from lesion_extraction_2d.lesion_extractor_2d import get_train_data
from data_visualization.adc_lesion_values import apply_window

IS_TRAIN = False

def get_activations(model, layer, X_batch):
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
        activations = get_activations([X_batch,0])
        return activations

def predict_to_file(filename, path_to_model):
    model = load_model(path_to_model)
    model.summary()
    
    # Data
    if IS_TRAIN:
        file = 'prostatex-train.hdf5'
    else:
        file = 'prostatex-test.hdf5'
    h5_file_location = os.path.join('/scratch-shared/ISMI/prostatex', file)
    h5_file = h5py.File(h5_file_location, 'r')
    x, y, attr = get_train_data(h5_file, ['ADC'])
    x = np.expand_dims(x, axis=-1)
    windowed = []
    for lesion in x:
        windowed.append(apply_window(lesion, (500, 1100)))
    x = np.asarray(windowed)
    
    print(attr[0])
    
    activations = get_activations(model, -3, x)
    print('length of predictions:', len(activations))
    print('length of prediction[0]:', len(activations[0]))
    print(activations[0].shape)
    
    activations = activations[0]
    
    with open(filename, 'w') as f:
        header_items = ['proxid']
        if IS_TRAIN:
            header_items.append('clinsig')
            
        for index in range(activations.shape[-1]):
            header_items.append('feature'+str(index).zfill(2))
            
        attr_features = ['Age', 'Zone']
        header_items.extend(attr_features)
        
        header = ','.join(header_items) + '\n'
        
        f.write(header)

        for i in range(activations.shape[0]):
            patient_id = attr[i]['patient_id']
            fid = attr[i]['fid']

            line = ["ProstateX-%s-%s" % (patient_id, fid)]
            
            if IS_TRAIN:
                line.append('%f' % float(y[i]))
                
            features = ['%f' % feature for feature in activations[i,0,0,:]]
            line.extend(features)
            
            attribute_values = []
            for attribute  in attr_features:
                attribute_values.append(str(attr[i][attribute],'utf-8'))
            
            line.extend(attribute_values)
            
            line = ','.join(line)+'\n'
            f.write(line)
            print(line)


if __name__ == "__main__":
    if IS_TRAIN:
        outfile = 'features_train.csv'
    else:
        outfile = 'features_test.csv'
    predict_to_file(outfile, 'best_model_adc.hdf5')
