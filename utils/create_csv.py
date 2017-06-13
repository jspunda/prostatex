from lesion_extraction_2d.lesion_extractor_2d import get_train_data
from keras.models import load_model
import os
import h5py
import numpy as np
from data_visualization.adc_lesion_values import apply_window

def predict_to_file(filename, path_to_model):
    model = load_model(path_to_model)

    # Data
    h5_file_location = os.path.join('/scratch-shared/ISMI/prostatex', 'prostatex-test.hdf5')
    h5_file = h5py.File(h5_file_location, 'r')

    x, _, attr = get_train_data(h5_file, ['ADC', 't2_tse_tra', 't2_tse_sag'], size_mm=32, size_px=16)
    # x = np.expand_dims(x, axis=-1)
    # windowed = []
    # for lesion in x:
    #     windowed.append(apply_window(lesion, (500, 1100)))
    # x = np.asarray(windowed)

    predictions = model.predict(x, verbose=1)

    with open(filename, 'w') as f:
        f.write('proxid,clinsig\n')

        for i in range(len(predictions)):
            patient_id = attr[i]['patient_id']
            fid = attr[i]['fid']

            line = "ProstateX-%s-%s,%f\n" % (patient_id, fid, predictions[i])
            f.write(line)
            print(line)


if __name__ == "__main__":
    predict_to_file('predictions.csv', '../../best_model.hdf5')
