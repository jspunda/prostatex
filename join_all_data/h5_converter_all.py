import SimpleITK as sitk
import glob
import os
import csv
import h5py
from loaders.seriesloader import load_dicom_series

"""Script that creates a full hdf5 dataset from a folder All_training_data that contains, 
    for each directory \\ProstateX-AAAA, all DICOm and Ktrans data"""

"""Such directory can be created running Unify_train_data.py"""

""" The csv file used is the one created by csv_fix_all.py, that combines the info from all 3 csv files."""

""" The final h5 file is basically the same as before but in each group ProstateX-AAAA there's a new dataset Ktrans_0."""

def dicom_to_h5(root_dir, h5):
    sub_dirs = [x[0] for x in os.walk(root_dir)]  # Gather all subdirectories in 'root_dir'
    ages = {}
    for directory in sub_dirs:
        # print(directory)
        if directory.split(os.sep)[-1] == 'Ktrans': #if the file is from Ktrans, look for the .mhd
            Im_Ktrans = True
            file_list = glob.glob(directory + '/*.mhd')
        else:
            Im_Ktrans = False
            file_list = glob.glob(directory + '/*.dcm')  # Look for .dcm files
        if not file_list:  # If we find a dir with a .dcm or a .mhd series, process it
            continue

        series_filename = file_list[0]  # Checking just one .dcm file is sufficient
        img = sitk.ReadImage(series_filename)  # Read single .dcm file to obtain metadata

        # Extract some metadata that we want to keep
        if Im_Ktrans:
            patient_id = series_filename.split(os.sep)[-3]
            patient_age = ages[patient_id]
            series_number = '0'
            # Since there's only one ktrans per patient, the name will be unique by just adding 'Ktrans'
            data_path = patient_id + '/Ktrans' + '_' + series_number
        else:
            patient_id = img.GetMetaData('0010|0020').strip()
            patient_age = img.GetMetaData('0010|1010').strip()
            series_number = img.GetMetaData('0020|0011').strip()
            series_description = img.GetMetaData('0008|103e').strip()
            # Combine series description and series number to create a unique identifier for this DICOM series.
            # Should be unique for each patient. Only exception is ProstateX-0025, hence the try: except: approach.
            data_path = patient_id + '/' + series_description + '_' + series_number

            # Add the age info in the dictionary for Ktrans
            ages[patient_id] = patient_age
        try:
            print(patient_id)
            group = h5.create_group(data_path)
            if Im_Ktrans:
                data = sitk.GetArrayFromImage(img)
                pixeldata = group.create_dataset('pixel_array', data=data)
            else:
                pixeldata = group.create_dataset('pixel_array', data=load_dicom_series(directory))
            pixeldata.attrs.create('Age', patient_age)
            pixeldata.attrs.create('SeriesNr', series_number)
        except ValueError:
            print('Skipping duplicate {}'.format(data_path))


def train_csv_to_h5(csv_file, h5):
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        reader.next()  # Skip column names
        for row in reader:
            patient_id = row[0]
            name = row[1]
            finding_id = row[2]
            pos = row[3]
            world_matrix = row[4]
            ijk = row[5]
            top_level = row[6]
            spacing_between = row[7]
            voxel_spacing = row[8]
            dim = row[9]
            dcm_descr = row[10]
            dcm_sernum = row[11]
            zone = row[12]
            clin_sig = row[13]

            # csv file contains redundant information. For example row 4 and 5 add no more information when
            # row 3 has already been seen. Hence the try: except: approach.
            pathname = patient_id + '/' + dcm_descr + '_' + dcm_sernum + '/lesions/' + finding_id
            try:
                group = h5.create_group(pathname)
                group.attrs.create('Name', name)
                group.attrs.create('Pos', pos, dtype='S10')
                group.attrs.create('WorldMatrix', world_matrix, dtype='S10')
                group.attrs.create('ijk', ijk, dtype='S10')
                group.attrs.create('TopLevel', top_level, dtype='S10')
                group.attrs.create('SpacingBetween', spacing_between, dtype='S10')
                group.attrs.create('VoxelSpacing', voxel_spacing, dtype='S10')
                group.attrs.create('Dim', dim, dtype='S10')
                group.attrs.create('Zone', zone, dtype='S10')
                group.attrs.create('ClinSig', clin_sig, dtype='S10')
            except ValueError:
                print('Skipping duplicate {}'.format(pathname))


if __name__ == "__main__":
    # Example usage
    h5file = h5py.File('prostatex-train-ALL.hdf5', 'w')
    main_folder = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Training'
    data_folder = main_folder + '\\All_training_data'
    images_train_csv = main_folder + '\\ProstateX-TrainingLesionInformationv2\ProstateX-Images-Train-ALL.csv'

    dicom_to_h5(data_folder, h5file)
    train_csv_to_h5(images_train_csv, h5file)
