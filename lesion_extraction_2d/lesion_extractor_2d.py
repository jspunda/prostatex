import math
import h5py
from h5_query import get_lesion_info
from matplotlib import pyplot as plt


class Centroid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)


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


def get_train_data(h5_file, query_words, keep_lesion_data=False, size_px=16):
    lesion_info = get_lesion_info(h5_file, query_words)

    X = []
    y = []
    for infos, image in lesion_info:
        for lesion in infos:

            centroid = parse_centroid(lesion['ijk'])
            lesion_img = extract_lesion_2d(image, centroid, size=size_px)
            if lesion_img is None:
                continue

            X.append(lesion_img)
            if keep_lesion_data:
                y.append(lesion)
            else:
                y.append(lesion['ClinSig'] == b"TRUE")

    return X, y

if __name__ == "__main__":
    """ Example usage: """
    h5_file = h5py.File('C:\\Users\\Jeftha\\stack\\Rommel\\ISMI\\prostatex-train.hdf5', 'r')

    X, y = get_train_data(h5_file, ['ADC'])

    print(y[0])
    plt.imshow(X[0], cmap='gray')
    plt.show()
