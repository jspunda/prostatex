import os
import h5py
from lesion_extraction_2d.lesion_extractor_2d import get_train_data

def train_test_split(*arrays, **options):
    pass

if __name__ == "__main__":
    """ Example usage """
    h5_file_location = os.path.join('/scratch-shared/ISMI/prostatex', 'prostatex-train.hdf5')
    h5_file = h5py.File(h5_file_location, 'r')
    X, y = get_train_data(h5_file, ['ADC'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    print(X)
    print(y)