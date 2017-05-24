import SimpleITK as sitk
import os
import glob
import array
from lesion_extraction_2d.lesion_extractor_2d import extract_lesion_2d
import csv

import matplotlib.pyplot as plt


# set the directory of the ktrans images
scan_directory = 'ProstateXKtrains-train-fixed'

# find all the mhd file paths and put them in the list
def find_ktrans_series_path(root_dir):
    series_paths = []
    case_dirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]

    for case_dir in case_dirs:
        subdirs = os.listdir(case_dir)
        case_dir = os.path.join(case_dir, subdirs[0])
    #     zraw_lst = os.path.join(case_dir, subdirs[1])
        series_paths.append(case_dir)
    return series_paths

# load all the ktrans images and transfer them to a list with numpy array
def load_ktrans_series(inputdir):
    series_patch = find_ktrans_series_path(inputdir)
    im_lst = []
    for item in series_patch:
        raw_img = sitk.ReadImage(item)
        raw_np = sitk.GetArrayFromImage(raw_img)
        im_lst.append(raw_np)
    return im_lst

ktrans_lst_np = load_ktrans_series(scan_directory)

'''lesion centroid loading part'''

class Centroid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)

def parse_centroid(ijk):
    coordinates = ijk.split(b" ")
    return Centroid(int(coordinates[0]), int(coordinates[1]), int(coordinates[2]))

def get_ktrans_centroids_from_csv(filename):
    centroids = {}
    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        reader.next()  # Skipping first row (i.e. column names)
        for row in reader:
            patient_id = row[0]
            name = row[1]
            ijk_coords = row[4]
            c = parse_centroid(ijk_coords)  # Fifth column contains ijk coordinates for lesion centroid
            try:
                # We need to add a list of lesions, since there can be multiple lesions per ADC file
                centroids[patient_id].append(c)
            except KeyError:
                centroids[patient_id] = [c]  # Fifth column contains ijk coordinates for lesion centroid
    return centroids


filename ='ProstateX-TrainingLesionInformationv2/ProstateX-Images-KTrans-Train.csv'
lesion_dic = get_ktrans_centroids_from_csv(filename)


'''lesions extraction part'''

ct_lst = []
for item in lesion_dic:
    ct = lesion_dic[item][0]
    ct_lst.append(ct)

def extract_lesion_2d(img, centroid_position, size):

    sizecal = size
    x_start = int(centroid_position.x - sizecal / 1.5)
    x_end = int(centroid_position.x + sizecal / 1.5)
    y_start = int(centroid_position.y - sizecal / 1.5)
    y_end = int(centroid_position.y + sizecal / 1.5)

    # Quick try-except fix for possibly miss-annotated centroid coords (Known cases: ProstateX-0154)
    try:
        img_slice = img[centroid_position.z]
    except IndexError:
        img_slice = img[centroid_position.z - 1]

    return img_slice[y_start:y_end, x_start:x_end]

'''Here you can input the patient id to check the exact patient'''
patient_num = 27


'''ATTENTION HERE!! As a tool to get the general view about ktrans images. 
Here I only subtract the first lesion in the list for each patient. 
So the total number is 204, not 384.'''

'''Here you can input the size by milimeter to check the lesion square'''
lesion_ktrans = extract_lesion_2d(ktrans_lst_np[patient_num], centroid_position=ct_lst[patient_num] , size=16)

# show the lesion
plt.interactive(False)
plt.imshow(lesion_ktrans, cmap='gray', interpolation='none')
plt.show()
