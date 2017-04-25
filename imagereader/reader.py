import SimpleITK
import glob
import os


def load_dicom_series(input_dir):
    """Reads an entire DICOM series of slices from 'input_dir' and returns its pixel data as an array."""

    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    return SimpleITK.GetArrayFromImage(dicom_series)


def find_dicom_series(root_dir, key, value):
    """Returns all DICOM series as pixel data in 'root_dir' and its sub directories in which
    the metadata key 'key' has value 'value'. Since DICOM metadata is lost when reading
    all slices at once, we first check a single .dcm slice to determine whether
    this series is relevant. If so, load entire directory, if not, move to next directory"""

    dicom_series = []
    sub_dirs = [x[0] for x in os.walk(root_dir)]  # Gather all subdirectories in 'root_dir'
    for directory in sub_dirs:
        file_list = glob.glob(directory + '/*.dcm')  # Look for .dcm files
        if file_list:  # If we find a dir with a .dcm series, process it
            dcm_file = file_list[0]  # Checking just one .dcm file is sufficient
            img = SimpleITK.ReadImage(dcm_file)  # Read single .dcm file to obtain metadata
            if value in img.GetMetaData(key):  # Check whether metadata key contains the right value
                dicom_series.append(load_dicom_series(directory))  # Loads all slices at once
    return dicom_series

# Example usage
# rootdir = 'C:/Users/Jeftha/Downloads/DOI/'
# key = '0008|103e'
# value = 'ADC'
#
# find_dicom_series(rootdir, key, value)
