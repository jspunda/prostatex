import h5py

"""Script that contains functionality to query our HDF5 dataset, using list comprehensions"""


def dicom_series_query(h5_file, query_words):
    """Returns a list of HDF5 groups of DICOM series that match words in query_words."""
    query_result = [
        h5_file[patient_id][dcm_series]  # We want patients with DICOM series such that:
        for patient_id in h5_file.keys()  # For all patients
        for dcm_series in h5_file[patient_id].keys()  # For all DICOM series
        for word in query_words  # For every word in query words
        if word in dcm_series  # The word is present in DICOM series name
        ]
    return query_result


def filename_to_patient_id(name):
    return name[11:15]


def get_lesion_info(h5_file, query_words):
    query = dicom_series_query(h5_file, query_words)

    # list of attributes to include in the lesion info
    include_attrs = ['ijk', 'VoxelSpacing', 'Zone', 'ClinSig']

    lesions_info = []
    for h5_group in query:
        if 'pixel_array' not in h5_group or 'lesions' not in h5_group:
            print('Warning in {}: No pixel array or lesions found for {}. Skipping...'
                  .format(get_lesion_info, h5_group))
            continue

        pixel_array = h5_group['pixel_array'][:]  # The actual DICOM pixel data
        patient_age = h5_group['pixel_array'].attrs.get('Age')

        lesion_info = []
        for finding_id in h5_group['lesions'].keys():
            lesion_dict = {
                'name': h5_group.name,
                'patient_id': filename_to_patient_id(h5_group.name)
            }
            for attr in include_attrs:
                # Per lesion finding, gather the attributes necessary for actual lesion extraction from DICOM image
                lesion_dict[attr] = h5_group['lesions'][finding_id].attrs.get(attr)
            lesion_dict['fid'] = finding_id
            lesion_dict['Age'] = patient_age
            lesion_info.append(lesion_dict)

        lesions_info.append([lesion_info, pixel_array])

    return lesions_info


if __name__ == '__main__':
    """Example usage"""
    # Some basic examples using list comprehension on our HDF5 set:

    h5_file = h5py.File('C:\\Users\Jeftha\Downloads\prostatex-test.hdf5', 'r')

    # Selecting all patients
    patients = [h5_file[patient_id] for patient_id in h5_file.keys()]
    print(len(patients))

    # Selecting all DICOM series
    # Note that this would take quite some time using our old approach
    # Now it's almost instant
    series = [h5_file[patient_id][dcm_series]
              for patient_id in h5_file.keys()
              for dcm_series in h5_file[patient_id].keys()]
    print(len(series))

    # Selecting all 'ADC' DICOM series
    adc_series = [h5_file[patient_id][dcm_series]
                  for patient_id in h5_file.keys()
                  for dcm_series in h5_file[patient_id].keys()
                  if '_ADC' in dcm_series]
    print(len(adc_series))

    lesions_info = get_lesion_info(h5_file, ['ADC'])
    for lesion_info, pixel_array in lesions_info:
        current_patient = lesion_info[0]['name'].split('/')[1]
        print('{} with {} lesion(s): {}'.format(pixel_array.shape, len(lesion_info), lesion_info))
