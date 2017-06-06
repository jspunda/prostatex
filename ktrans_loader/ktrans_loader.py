import h5py
import numpy as np
# np.seterr(divide='ignore', invalid='ignore')

def dicom_series_query(h5_file, query_words):
    """Returns a list of HDF5 groups of DICOM series that match words in query_words."""
    query_result = [
        h5_file[patient_id][dcm_series]  # We want patients with DICOM series such that:
        for patient_id in h5_file.keys()  # For all patients
        for dcm_series in h5_file[patient_id].keys()  # For all DICOM series
        for word in query_words  # For every word in query words
        if word in dcm_series  # The word is present in DICOM series name
        ]
    return query_result

def filename_to_patient_id(name):
    return name[11:15]

def get_lesion_info(h5_file, query_words):
    query = dicom_series_query(h5_file, query_words)

    # list of attributes to include in the lesion info
    include_attrs = ['ijk', 'VoxelSpacing', 'Zone', 'ClinSig']

    lesions_info = []
    for h5_group in query:
        pixel_array = h5_group['pixel_array'][:]  # The actual DICOM pixel data
        patient_age = h5_group['pixel_array'].attrs.get('Age')

        lesion_info = []
        for finding_id in h5_group['lesions'].keys():
            lesion_dict = {
                'name': h5_group.name,
                'patient_id': filename_to_patient_id(h5_group.name)
            }
            for attr in include_attrs:
                # Per lesion finding, gather the attributes necessary for actual lesion extraction from DICOM image
                lesion_dict[attr] = h5_group['lesions'][finding_id].attrs.get(attr)
            lesion_dict['fid'] = finding_id
            lesion_dict['Age'] = patient_age
            lesion_info.append(lesion_dict)

        lesions_info.append([lesion_info, pixel_array])

    return lesions_info

class Centroid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)


def extract_lesion_2d(img, centroid_position, size=None, realsize=10, imagetype='ADC'):
    if imagetype == 'ADC':
        if size is None:
            sizecal = math.ceil(realsize / 1.5)
        else:
            sizecal = size
    else:
        sizecal = size
    x_start = int(centroid_position.x - sizecal / 2)
    x_end = int(centroid_position.x + sizecal / 2)
    y_start = int(centroid_position.y - sizecal / 2)
    y_end = int(centroid_position.y + sizecal / 2)

    if centroid_position.z < 0 or centroid_position.z >= len(img):
        return None

    img_slice = img[centroid_position.z]

    return img_slice[y_start:y_end, x_start:x_end]


def parse_centroid(ijk):
    coordinates = ijk.split(b" ")
    return Centroid(int(coordinates[0]), int(coordinates[1]), int(coordinates[2]))

def image_normalise(raw_np_image, max_rate):
    X_lst = []
    for image in raw_np_image:
        X_sort = np.sort(image.flatten())
        range_max = np.int(max_rate * len(X_sort)) - 1
        X_max = X_sort[range_max]
        X_min = 0.0
        X_range_shrink = np.clip(image, X_min, X_max)
        X_out = (X_range_shrink - X_range_shrink.flatten().min())/(X_range_shrink.flatten().max() - X_range_shrink.flatten().min())
        X_lst.append(X_out)
    return X_lst

def get_train_data_ktrans(h5_file, query_words, size_px=12):
    lesion_info = get_lesion_info(h5_file, query_words)

    X = []
    y = []
    lesion_attributes = []
    for infos, image in lesion_info:
        for lesion in infos:

            centroid = parse_centroid(lesion['ijk'])
            lesion_img = extract_lesion_2d(image, centroid, size=size_px)
            if lesion_img is None:
                continue

            X.append(lesion_img)

            lesion_attributes.append(lesion)

            y.append(lesion['ClinSig'] == b"TRUE")

    X = image_normalise(X, max_rate=0.97)

    return np.asarray(X), np.asarray(y), np.asarray(lesion_attributes)

'''To get normalised ktans images'''

# data_dir = 'h5/prostatex-train-ALL.hdf5'
# h5_file = h5py.File(data_dir)

# X, y, attr = get_train_data_ktrans(h5_file, ['Ktrans'])

'''plot the result'''
# import matplotlib.pyplot as plt
#
# plt.plot()
# plt.imshow(X[0], cmap='gray', interpolation='none')
# plt.show()