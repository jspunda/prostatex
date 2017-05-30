import SimpleITK
import glob
import os
import csv
import h5py
from loaders.seriesloader import load_dicom_series

"""
Script used to turn raw DICOM files and their .csv information into one HDF5 dataset. HDF5 allows
for a great speed-up when querying the data. Requires the DOI data folder and ProstateX-Images-Train-NEW.csv 
obtained by running csv_fix.py. (We only have to run this once to obtain the HDF5 dataset file, unless we want to 
structure the set in a different way or keep more info from the DICOM files. Then we have to rebuild this set and we all
have to download the new version. Rebuilding entire train set takes around 10-15 minutes)

The HDF5 set will be of the following form:

[ProstateX-0000]/
|
|---[DICOM_Series_name {1}]/  (For example ep2d_diff_tra_DYNDIST_ADC)
|   |
|   |---'pixel_array'/
|   |   |
|   |   |--- Attributes kept from DICOM header: For now just patient age and DICOM_series_Num
|   |   |--- Raw pixel array (i.e. all DICOM slices for this series)
|   |
|   |---'lesions'/
|   |   |
|   |   |--- [Finding_ID {1}]
|   |   |   |
|   |   |   |--- All lesion attributes from .csv: (e.g. ijk, voxelspacing, zone, etc.)
|   |   |
|   |   |--- [Finding_ID {2}]
|   |   |   |
|   |   |   |--- All lesion attributes from .csv: (e.g. ijk, voxelspacing, zone, etc.)
|   |   |
|   |   |--- ...
|   |   .
|   |   .
|   |   .
|   |
|---[DICOM_Series_name {2}]/ (For example t2_tse_cor)
|   |
|   |---'pixel_array'/
|   |   |
|   |   |--- Attributes kept from DICOM header: For now just patient age and DICOM_series_Num
|   |   |--- Raw pixel array (i.e. all DICOM slices for this series)
|   |
|   |---'lesions'/
|   |   |
|   |   |--- [Finding_ID {1}]
|   |   |   |
|   |   |   |--- All lesion attributes from .csv: (e.g. ijk, voxelspacing, zone, etc.)
|   |   |
|   |   |--- ...
|   |   .
|   |   .
|   |   .
|   |
|--- ...
|   .
|   .
|   .
[ProstateX-0001]/
|
|--- ...
.
.
.

"""


def dicom_to_h5(root_dir, h5):
    sub_dirs = [x[0] for x in os.walk(root_dir)]  # Gather all subdirectories in 'root_dir'
    for directory in sub_dirs:
        # print(directory)
        file_list = glob.glob(directory + '/*.dcm')  # Look for .dcm files
        if not file_list:  # If we find a dir with a .dcm series, process it
            continue

        dcm_filename = file_list[0]  # Checking just one .dcm file is sufficient
        img = SimpleITK.ReadImage(dcm_filename)  # Read single .dcm file to obtain metadata

        # Extract some metadata that we want to keep
        patient_id = img.GetMetaData('0010|0020').strip()
        patient_age = img.GetMetaData('0010|1010').strip()
        series_number = int(img.GetMetaData('0020|0011').strip())
        series_description = img.GetMetaData('0008|103e').strip()

        data_path = patient_id + '/' + series_description
        print('Converting: {}'.format(data_path))

        # If we find a DICOM series that already exists, we check if the series number is higher. If so, remove
        # series that is already present and add this one.
        create = False

        if data_path in h5:
            if h5[data_path]['pixel_array'].attrs.get('SeriesNr') < series_number:
                del h5[data_path]
                print('New series has higher series number, so adding.')
                create = True
            else:
                print('New series has lower series number, so not adding.')
        else:
            create = True

        if create:
            group = h5.create_group(data_path)
            pixeldata = group.create_dataset('pixel_array', data=load_dicom_series(directory))
            pixeldata.attrs.create('Age', patient_age)
            pixeldata.attrs.create('SeriesNr', series_number)


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

            # train csv file contains redundant information. For example row 4 and 5 add no more information when
            # row 3 has already been seen.
            pathname = patient_id + '/' + dcm_descr + '/lesions/' + finding_id

            if pathname in h5:
                print('Skipping duplicate {}'.format(pathname))
                continue
            else:
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

            # try:
            #     group = h5.create_group(pathname)
            #     group.attrs.create('Name', name)
            #     group.attrs.create('Pos', pos, dtype='S10')
            #     group.attrs.create('WorldMatrix', world_matrix, dtype='S10')
            #     group.attrs.create('ijk', ijk, dtype='S10')
            #     group.attrs.create('TopLevel', top_level, dtype='S10')
            #     group.attrs.create('SpacingBetween', spacing_between, dtype='S10')
            #     group.attrs.create('VoxelSpacing', voxel_spacing, dtype='S10')
            #     group.attrs.create('Dim', dim, dtype='S10')
            #     group.attrs.create('Zone', zone, dtype='S10')
            #     group.attrs.create('ClinSig', clin_sig, dtype='S10')
            # except ValueError:
            #     print('Skipping duplicate {}'.format(pathname))


if __name__ == "__main__":
    """Example usage: """
    # Example usage for train set
    h5file = h5py.File('prostatex-train.hdf5', 'w')
    dcm_folder = 'C:\Users\Jeftha\Downloads\DOI'
    images_train_csv = 'C:\Users\Jeftha\Downloads\ProstateX-TrainingLesionInformationv2' \
                       '\ProstateX-TrainingLesionInformationv2\ProstateX-Images-Train-NEW.csv'

    dicom_to_h5(dcm_folder, h5file)
    train_csv_to_h5(images_train_csv, h5file)

    # Example usage for test set
    # h5file = h5py.File('prostatex-test.hdf5', 'w')
    # dcm_folder = 'C:\Users\Jeftha\Downloads\\test_set\DOI'
    # images_test_csv = 'C:\Users\Jeftha\Downloads\ProstateX-TestLesionInformation' \
    #                    '\ProstateX-TestLesionInformation\ProstateX-Images-Test-NEW.csv'
    #
    # dicom_to_h5(dcm_folder, h5file)
    # train_csv_to_h5(images_test_csv, h5file)
