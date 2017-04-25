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
    case_dirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
    
    for case_dir in case_dirs:
        if len(os.listdir(case_dir)) != 1:
            print("Unexpectedly found multiple folders for case", case_dir)
            print("Skipping this case for now.")
            continue
        else:
            # the 298318301283.129392381.1392892.1923891 folder name
            case_dir = os.path.join(case_dir, os.listdir(case_dir)[0])
        
        file = glob.glob(case_dir + '/*.dcm')[0]  # Checking just one .dcm file is sufficient
        img = SimpleITK.ReadImage(file)  # Read single .dcm file to obtain metadata
        if value in img.GetMetaData(key):  # Check whether metadata key contains the right value
            dicom_series.append(load_dicom_series(case_dir))  # Loads all slices at once
    return dicom_series
