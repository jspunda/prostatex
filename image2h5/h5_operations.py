from matplotlib import pyplot
from image2h5 import h5_to_img


class Centroid():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def extract_lesion_2d(img, centroid_position, size=10):
    x_start = int(centroid_position.x - size / 2)
    x_end = int(centroid_position.x + size / 2)
    y_start = int(centroid_position.y - size / 2)
    y_end = int(centroid_position.y + size / 2)

    img_slice = img[centroid_position.z]

    return img_slice[y_start:y_end, x_start:x_end]


def plot_h5_slice(h5_path, slice=0):
    """ Plots a slice from a h5 image file """
    img = h5_to_img(h5_path)

    pyplot.imshow(img[slice], cmap='gray')
    pyplot.show()


if __name__ == "__main__":
    """ EXAMPLE USAGE: """
    h5_path = "C:/Users/Jeftha/Downloads/DOI/ProstateX-0000/h5/img0.h5"
    # plot_h5_slice(h5_path, slice=9)

    img = h5_to_img(h5_path)

    centroid_position = Centroid(167, 224, 9)

    lesion = extract_lesion_2d(img, centroid_position, size=40)

    pyplot.imshow(lesion, cmap='gray')
    pyplot.show()
