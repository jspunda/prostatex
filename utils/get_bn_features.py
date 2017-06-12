import h5py
from lesion_extraction_2d.lesion_extractor_2d import get_train_data
from data_visualization.adc_lesion_values import get_pixels_in_window
from collections import OrderedDict


def get_adc_features(h5_file_path, window, lesion_size):
    """
    Gathers amount of pixels within window and the mean of those pixels for ADC images.
    Returns an ordered dictionary of the form:

    'ProstateX-0000-1'/
    |
    |---'mean'/
    |   |
    |   |--- Mean value of pixels in window
    |
    ---'count'/
    |   |
    |   |--- Amount of pixels within window
    |
    'ProstateX-0001-1'/
    |
    |--- ...
    .
    .
    .
    """
    h5_file = h5py.File(h5_file_path, 'r')
    x, _, attrs = get_train_data(h5_file, ['ADC'], size_px=lesion_size)

    feature_dict = OrderedDict()
    for i in range(len(attrs)):
        lesion = x[i]
        attr = attrs[i]
        prox_id = 'ProstateX-{}-{}'.format(attr['patient_id'], attr['fid'])
        pixels_in_window = get_pixels_in_window(lesion, window)
        feature_dict[prox_id] = {}
        if pixels_in_window is not None:
            feature_dict[prox_id]['count'] = len(pixels_in_window)
            feature_dict[prox_id]['mean'] = pixels_in_window.mean()
        else:
            feature_dict[prox_id]['count'] = 0
            feature_dict[prox_id]['mean'] = 0

    return feature_dict

if __name__ == "__main__":
    """ Example usage: """
    h5_path = 'C:\\users\\Jeftha\\stack\\Rommel\\ISMI\\data\\prostatex-train.hdf5'
    features = get_adc_features(h5_path, (500, 1100), 16)
    print(features)

    # Get info for patient 42, lesion 1
    print(features['ProstateX-0042-1']['count'], features['ProstateX-0042-1']['mean'])

    # Get info for all patients and lesions
    for patient_lesion_id in features.keys():
        print(patient_lesion_id, features[patient_lesion_id]['count'], features[patient_lesion_id]['mean'])
