import csv
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
        if size == None:
            sizecal = math.ceil(realsize / 2)
    else:
        sizecal = size
    x_start = int(centroid_position.x - sizecal / 2)
    x_end = int(centroid_position.x + sizecal / 2)
    y_start = int(centroid_position.y - sizecal / 2)
    y_end = int(centroid_position.y + sizecal / 2)

    # Quick try-except fix for possibly miss-annotated centroid coords (Known cases: ProstateX-0154)
    try:
        img_slice = img[centroid_position.z]
    except IndexError:
        img_slice = img[centroid_position.z - 1]

    return img_slice[y_start:y_end, x_start:x_end]


def parse_centroid(ijk):
    coordinates = ijk.split(" ")
    return Centroid(int(coordinates[0]), int(coordinates[1]), int(coordinates[2]))


def get_adc_centroids_from_csv(filename):
    centroids = {}
    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        reader.next()  # Skipping first row (i.e. column names)
        for row in reader:
            patient_id = row[0]
            name = row[1]
            ijk_coords = row[5]
            if patient_id not in ignore_list:
                if 'ADC' in name:  # Check whether name contains 'ADC'
                    c = parse_centroid(ijk_coords)  # Fifth column contains ijk coordinates for lesion centroid
                    try:
                        # We need to add a list of lesions, since there can be multiple lesions per ADC file
                        centroids[patient_id].append(c)
                    except KeyError:
                        centroids[patient_id] = [c]  # If there is no entry for this patient yet, create it.
    return centroids


def get_lesions_from_imgs(centroids, imgs, lesion_size, real_leison_size, imagetype):
    lesions = {}
    for key in imgs:
        pixel_array = imgs[key][0]  # For now imgs[key] contains a list, might change later to single object
        lesion_pixel_array = []
        for centroid in centroids[key]:
            print('Extracting lesions on pos {} for {}'.format(centroid, key))
            lesion_pixel_array.append(extract_lesion_2d(pixel_array, centroid, lesion_size, real_leison_size, imagetype))
        lesions[key] = lesion_pixel_array
    return lesions


###### Modified old version of DICOM image reader below, since new one returns in different patientID order#######
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

    dicom_series = {}
    sub_dirs = [x[0] for x in os.walk(root_dir)]  # Gather all subdirectories in 'root_dir'
    for directory in sub_dirs:
        # print(directory)
        file_list = glob.glob(directory + '/*.dcm')  # Look for .dcm files
        if file_list:  # If we find a dir with a .dcm series, process it
            dcm_file = file_list[0]  # Checking just one .dcm file is sufficient
            img = SimpleITK.ReadImage(dcm_file)  # Read single .dcm file to obtain metadata
            if value in img.GetMetaData(key):  # Check whether metadata key contains the right value
                patient_id = directory.split('\\')[5]
                if patient_id not in ignore_list:
                    print('Loading images for {}'.format(patient_id))
                    series = load_dicom_series(directory)  # Loads all slices at once
                    try:
                        dicom_series[patient_id].append(series)
                    except KeyError:
                        dicom_series[patient_id] = [series]

    return dicom_series

# 'Abnormal' cases (i.e cases with multiple ADC series)
ignore_list = ['ProstateX-0025', 'ProstateX-0031', 'ProstateX-0190', 'ProstateX-0191']

# Example usage
rootdir = 'C:\Users\Jeftha\Downloads\DOI'
key = '0008|103e'  # Description tagname
value = 'ADC'

file = 'C:\Users\Jeftha\Downloads\ProstateX-TrainingLesionInformationv2' \
       '\ProstateX-TrainingLesionInformationv2\ProstateX-Images-Train.csv'

images = find_dicom_series(rootdir, key, value)
print(len(images))
centroids = get_adc_centroids_from_csv(file)
print(len(centroids))
lesions = get_lesions_from_imgs(centroids, images, None, 20, 'ADC')
print(len(lesions))

# Plot one lesion
plt.imshow(lesions['ProstateX-0000'][0], cmap='gray')
plt.show()
