import os
import h5py
import numpy as np
import sys
from lesion_extraction_2d.lesion_extractor_2d import get_train_data


def x_y_shuffle(X, y, attr):
    indices = np.random.permutation(len(X))

    return X[indices], y[indices], attr[indices]


def stratify(X, y, attr, skip_patients, ratio):
    current_ratio = stratify_ratio(y['train'])
    diff = current_ratio - ratio

    # number of elements we have to swap:
    change_elements = int(round(len(y['train']) * diff))
    direction = change_elements < 0
    change_elements = abs(change_elements)

    swapped = 0
    i = 0
    j = 0
    while True:
        if i == len(y['train']) or j == len(y['test']):
            # we ran out of elements to swap...
            break

        if y['train'][i] == direction or attr['train'][i]['patient_id'] in skip_patients:
            i += 1
            continue

        if y['test'][j] != direction or attr['test'][j]['patient_id'] in skip_patients:
            j += 1
            continue

        # swap i and j if they are of different label:
        temp_X_test = X['test'][j]
        temp_y_test = y['test'][j]
        temp_attr_test = attr['test'][j]

        X['test'][j] = X['train'][i]
        y['test'][j] = y['train'][i]
        attr['test'][j] = attr['train'][i]

        X['train'][i] = temp_X_test
        y['train'][i] = temp_y_test
        attr['train'][i] = temp_attr_test

        swapped += 1
        i += 1
        j += 1

        if swapped >= change_elements:
            break

    # print this ratio for confirming the set is now stratified
    new_ratio = stratify_ratio(y['train'])

    return X, y, attr


def stratify_ratio(y):
    c = np.bincount(y)

    return c[1] / len(y)


def train_test_split(X, y, attr, **options):
    if len(X) != len(y) or len(X) != len(attr):
        raise ValueError("Array lengths not matching")

    # test size defaults to 0.25
    test_size = options.pop('test_size', 0.25)
    rand_state = options.pop('random_state', None)

    if rand_state is not None:
        np.random.seed(rand_state)

    X, y, attr = x_y_shuffle(X, y, attr)

    X_out = {}
    y_out = {}
    attr_out = {}
    patient_ids = {}

    arrs = ['train', 'test']
    for a in arrs:
        X_out[a] = []
        y_out[a] = []
        attr_out[a] = []
        patient_ids[a] = []

    patients_occuring_multiple_times = []
    for i in range(len(X)):
        patient_id = attr[i]['patient_id']

        destination = ''

        for which, arr in patient_ids.items():
            if patient_id in arr:
                # this patient already exists in one of the arrays, put this one
                # there as well
                destination = which
                patients_occuring_multiple_times.append(patient_id)
                break

        if len(X_out['test']) == 0 and len(X_out['train']) == 0:
            # first element, put in test to prevent division by zero during ratio calculation
            destination = 'test'

        if destination == '':  # no array for this patient yet
            # try to maintain ratio between test and train
            ratio = len(X_out['test']) / (len(X_out['test']) + len(X_out['train']))
            if ratio < test_size:
                destination = 'test'
            else:
                destination = 'train'

        X_out[destination].append(X[i])
        y_out[destination].append(y[i])
        attr_out[destination].append(attr[i])
        patient_ids[destination].append(patient_id)

    for a in arrs:
        X_out[a] = np.asarray(X_out[a])
        y_out[a] = np.asarray(y_out[a])

    X_out, y_out, attr_out = stratify(X_out, y_out, attr_out, skip_patients=patients_occuring_multiple_times, ratio=stratify_ratio(y))

    return X_out['train'], X_out['test'], y_out['train'], y_out['test']

if __name__ == "__main__":
    """ Example usage """
    h5_file_location = os.path.join('/media/koen/Stack/Stack/uni/Machine Learning in Practice', 'prostatex-train.hdf5')
    h5_file = h5py.File(h5_file_location, 'r')
    X, y, attr = get_train_data(h5_file, ['ADC'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, attr, test_size=0.25)

    print(type(X_train))

    print("Split with ratio: %0.3f" % (len(X_test) / len(X)))
    print(len(X_test))
    print(len(X_train))