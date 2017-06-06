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


def extract_lesion_2d(img, centroid_position, size=None, realsize=16, imagetype='ADC'):
    if imagetype == 'ADC':
        if size is None:
            sizecal = math.ceil(realsize / 2)
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


def parse_voxelspacing(spacing):
    spacing = spacing.split(b",")
    return VoxelSpacing(int(spacing[0]), int(spacing[1]), int(spacing[2]))


def get_train_data(h5_file, query_words, size_px=16):
    lesion_info = get_lesion_info(h5_file, query_words)

    X = []
    y = []
    lesion_attributes = []
    previous_patient = ''
    for infos, image in lesion_info:
        current_patient = infos[0]['name'].split('/')[1]
        if current_patient == previous_patient:
            print('Warning in {}: Found duplicate match for {}. Skipping...'
                  .format(get_train_data.__name__, current_patient))
            continue
        for lesion in infos:

            centroid = parse_centroid(lesion['ijk'])

            # convert mm to pix
            voxel_sizes = parse_voxelspacing(lesion['VoxelSpacing'])
            size_px = size_px // voxel_sizes.width

            lesion_img = extract_lesion_2d(image, centroid, size=size_px)

            # resample
            lesion_img = imresize(lesion_img, (size_px, size_px), interp='bilinear')

            if lesion_img is None:
                print('Warning in {}: ijk out of bounds for {}. No lesion extracted'
                      .format(get_train_data.__name__, lesion))
                continue

            X.append(lesion_img)

            lesion_attributes.append(lesion)

            y.append(lesion['ClinSig'] == b"TRUE")

        previous_patient = current_patient

    return np.asarray(X), np.asarray(y), np.asarray(lesion_attributes)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    """ Example usage: """
    h5_file = h5py.File('C:\\Users\\Jeftha\\stack\\Rommel\\ISMI\\prostatex-train.hdf5', 'r')

    X, y, attr = get_train_data(h5_file, ['ADC'])

    print(y[0])
    print(attr[0])
    plt.imshow(X[0], cmap='gray')
    plt.show()
