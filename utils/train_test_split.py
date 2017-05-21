import os
import h5py
import numpy as np
import sys
from lesion_extraction_2d.lesion_extractor_2d import get_train_data


def x_y_shuffle(X, y, attr):
    indices = np.random.permutation(len(X))

    return X[indices], y[indices], attr[indices]


def train_test_split(X, y, attr, **options):
    if len(X) != len(y) or len(X) != len(attr):
        raise ValueError("Array lengths not matching")

    # test size defaults to 0.25
    test_size = options.pop('test_size', 0.25)

    X, y, attr = x_y_shuffle(X, y, attr)

    X_out = {}
    y_out = {}
    patient_ids = {}

    arrs = ['train', 'test']
    for a in arrs:
        X_out[a] = []
        y_out[a] = []
        patient_ids[a] = []

    for i in range(len(X)):
        patient_id = attr[i]['patient_id']

        destination = ''

        for which, arr in patient_ids.items():
            if patient_id in arr:
                # this patient already exists in one of the arrays, put this one
                # there as well
                destination = which
                break

        if len(X_out['test']) == 0 and len(X_out['train']) == 0:
            # first element, put in test to prevent division by zero during ratio calculation
            destination = 'test'

        if destination == '':  # no array for this patient yet
            # try to maintain ratio between test and train
            if len(X_out['test']) / (len(X_out['test']) + len(X_out['train'])) < test_size:
                destination = 'test'
            else:
                destination = 'train'

        X_out[destination].append(X[i])
        y_out[destination].append(y[i])
        patient_ids[destination].append(patient_id)

    return np.asarray(X_out['train']), np.asarray(X_out['test']), np.asarray(y_out['train']), np.asarray(y_out['test'])

if __name__ == "__main__":
    """ Example usage """
    h5_file_location = os.path.join('/media/koen/Stack/Stack/uni/Machine Learning in Practice', 'prostatex-train.hdf5')
    h5_file = h5py.File(h5_file_location, 'r')
    X, y, attr = get_train_data(h5_file, ['ADC'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, attr, test_size=0.25)

    print("Split with ratio: %0.3f" % (len(X_test) / len(X)))
    print(len(X_test))
    print(len(X_train))
