""" Nothing important going on here, just somewhere to try stuff"""

import SimpleITK as sitk
import glob
import os
import numpy as np
import csv
import h5py
from loaders.seriesloader import load_dicom_series

def get_mhd_dir_lst(main_path):
    sub_dirs = [x[0] for x in os.walk(main_path)]  # Gather all subdirectories in 'root_dir'
    #print 'len = ', len(sub_dirs)
    dirs = sub_dirs
    for i in dirs:
        file_list = glob.glob(i + '/*.mhd')  # Look for .dcm files
        if not file_list:
            sub_dirs.remove(i)
    return sub_dirs

def get_mhd_patient_id(mhd_dir):
    return mhd_dir.split(os.sep)[-2]

def get_mhd_voxel_spacing(mhd_dir):
    img = sitk.ReadImage(mhd_dir)
    return img.GetSpacing()

root_dir = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Training\\Ktrans'

lst = get_mhd_dir_lst(root_dir)
my_dir = np.random.choice(lst)
patient_id = get_mhd_patient_id(my_dir)

print len(lst), my_dir
print patient_id

#mhd_filename = file_list[0]  # Checking just one .dcm file is sufficient
#img = SimpleITK.ReadImage(mhd_filename)


#patient_age = img.GetMetaData('0010|1010').strip()
#series_number = img.GetMetaData('0020|0011').strip()
#series_description = img.GetMetaData('0008|103e').strip()
#print patient_id



def get_unique_case_id(main_path):
    sub_dirs = [x[0] for x in os.walk(main_path)]  # Gather all subdirectories in 'root_dir'
    case_ids = []
    ages = {}
    for directory in sub_dirs:
        # print(directory)
        #print directory.split(os.sep)[-1] == 'Ktrans', directory.split(os.sep)[-1]
        if directory.split(os.sep)[-1] == 'Ktrans':
            file_list = glob.glob(directory + '/*.mhd')
            if not file_list:  # If we find a dir with a .dcm series, process it
                continue
            patient_id = get_mhd_patient_id(directory)
            case_ids.append(patient_id + '/Ktrans' + '_' + ages[patient_id])
        else:
            file_list = glob.glob(directory + '/*.dcm')  # Look for .dcm files
            if not file_list:  # If we find a dir with a .dcm series, process it
                continue

            dcm_filename = file_list[0]  # Checking just one .dcm file is sufficient
            img = sitk.ReadImage(dcm_filename)  # Read single .dcm file to obtain metadata

            # Extract some metadata that we want to keep
            patient_id = img.GetMetaData('0010|0020').strip()
            patient_age = img.GetMetaData('0010|1010').strip()
            series_number = img.GetMetaData('0020|0011').strip()
            series_description = img.GetMetaData('0008|103e').strip()

            case_ids.append(patient_id + '/' + series_description + '_' + series_number + '_' + patient_age)
            ages[patient_id] = patient_age
    return case_ids

root_dir_2 = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Training\\Own'
dir_lst = get_unique_case_id(root_dir_2)


root_dir = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Training\\Own'
sub_dirs = [x[0] for x in os.walk(root_dir) if x[0].split(os.sep)[-1] == 'Ktrans']
dir =glob.glob(np.random.choice(sub_dirs) + '/*.mhd')

print dir

img = sitk.ReadImage(dir[0])
VoxelSpacing = img.GetSpacing()
Dimen = str(img.GetSize()).replace(', ','x').replace('(','').replace(')','')
SpacingBetweenSlices = 'NaN'
TopLevel = 1
DCMSerDescr = 'Ktrans'
DCMSerNum = 0
print VoxelSpacing, Dimen