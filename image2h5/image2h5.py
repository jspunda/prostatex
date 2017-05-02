import h5py
from loaders.seriesloader import load_dicom_series


H5_DATASET_NAME = 'img_h5'


def img_to_h5(img_dir, h5_output_name):
    """ Saves all slices from an image to a h5 file. """
    img = load_dicom_series(img_dir)

    with h5py.File(h5_output_name, 'w') as f:
        f.create_dataset(H5_DATASET_NAME, data=img)

        print("Saved")


def h5_to_img(h5_input_name):
    """ Loads image data from a h5 file. """
    with h5py.File(h5_input_name, 'r') as f:
        return f[H5_DATASET_NAME][:]


if __name__ == "__main__":
    """ EXAMPLE USAGE: """
    # path to save h5 data to:
    save_path = "C:/Users/Jeftha/Downloads/DOI/ProstateX-0000/h5/img0.h5"

    # save image to 'save_path':
    img_to_h5(
        "C:/Users/Jeftha/Downloads/DOI/ProstateX-0000/"
        "1.3.6.1.4.1.14519.5.2.1.7311.5101.158323547117540061132729905711/"
        "1.3.6.1.4.1.14519.5.2.1.7311.5101.160028252338004527274326500702/",
        save_path)

    # load image from 'save_path':
    i = h5_to_img(save_path)

    print(i.shape)
