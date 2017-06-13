import math
import numpy as np
import h5py
from scipy.misc import imresize
from .h5_query import get_lesion_info


class Centroid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)


class VoxelSpacing:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def __repr__(self):
        return '({}, {}, {})'.format(self.width, self.height, self.depth)


def extract_lesion_2d(img, centroid_position, size_px=None):
    x_start = int(centroid_position.x - size_px // 2)
    x_end = int(centroid_position.x + size_px // 2)
    y_start = int(centroid_position.y - size_px // 2)
    y_end = int(centroid_position.y + size_px // 2)

    if centroid_position.z < 0 or centroid_position.z >= len(img):
        return None

    img_slice = img[centroid_position.z]

    return img_slice[y_start:y_end, x_start:x_end]


def parse_centroid(ijk):
    coordinates = ijk.split(b" ")
    return Centroid(int(coordinates[0]), int(coordinates[1]), int(coordinates[2]))


def parse_voxelspacing(spacing):
    spacing = spacing.split(b",")

    return VoxelSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))

  
def str_to_modality(in_str):
    modalities = ['ADC', 't2_tse_tra', 't2_tse_sag']
    for m in modalities:
        if m in in_str:
            return m
    
    return "NONE"


def get_train_data(h5_file, query_words, size_px=16, size_mm=16):
    lesion_info = get_lesion_info(h5_file, query_words)

    X = []
    y = []
    lesion_attributes = []
    previous_patient = ''
    previous_modality = ''
    
    unique_patient_ids = []
    
    for infos, image in lesion_info:
        _, current_patient, current_modality = infos[0]['name'].split('/')
        current_modality = str_to_modality(current_modality)
        
        if current_patient == previous_patient and current_modality == previous_modality:
            print('Warning in {}: Found duplicate match for {}. Skipping...'
                  .format(get_train_data.__name__, current_patient))
            continue
        for lesion in infos:
            if not ((lesion['patient_id'], lesion['fid']) in unique_patient_ids):
                unique_patient_ids.append((lesion['patient_id'], lesion['fid']))
            
            centroid = parse_centroid(lesion['ijk'])

            # convert mm to pix
            try:
                voxel_sizes = parse_voxelspacing(lesion['VoxelSpacing'])
            except IndexError:
                print(lesion['name'])
                print(lesion)
                import sys
                sys.exit(0)

            lesion_img = extract_lesion_2d(image, centroid, size_px=size_mm // voxel_sizes.width)

            if lesion_img is None:
                print('Warning in {}: ijk out of bounds for {}. No lesion extracted'
                      .format(get_train_data.__name__, lesion))
                continue

            # resample
            lesion_img = imresize(lesion_img, (size_px, size_px), interp='bilinear')

            X.append(lesion_img)

            lesion_attributes.append(lesion)

            y.append(lesion['ClinSig'] == b"TRUE")

        previous_patient = current_patient
        previous_modality = current_modality

    X_final = []
    y_final = []
    attr_final = []
    for patient_id, fid in unique_patient_ids:
        x_new = []
        y_new = 0
        for i in range(len(lesion_attributes)):
            attr = lesion_attributes[i]
            
            if attr['patient_id'] == patient_id and attr['fid'] == fid:
                x_new.append(X[i])
                y_new = y[i]
        
        if not len(x_new) == len(query_words):
            print("Missing modalities for patient %s" % patient_id)
            continue
        
        X_final.append(x_new)
        y_final.append(y_new)
        attr_final.append({'patient_id': patient_id, 'fid': fid})
    X_final = np.asarray(X_final)
    X_final = np.rollaxis(X_final, 1, 4)
    return X_final, np.asarray(y_final), np.asarray(attr_final)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    """ Example usage: """
    h5_file = h5py.File('C:\\Users\\Jeftha\\stack\\Rommel\\ISMI\\prostatex-train.hdf5', 'r')

    X, y, _ = get_train_data(h5_file, ['ADC'])

    print(y[0])
    print(attr[0])
    plt.imshow(X[0], cmap='gray')
    plt.show()
