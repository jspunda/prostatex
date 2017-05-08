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


if __name__ == '__main__':
    """Example usage"""
    # Some basic examples using list comprehension on our HDF5 set:

    h5 = h5py.File('C:\Users\Jeftha\stack\Rommel\ISMI\prostatex-train.hdf5', 'r')

    # Selecting all patients
    patients = [h5[patient_id] for patient_id in h5.keys()]
    print(len(patients))

    # Selecting all DICOM series
    # Note that this would take quite some time using our old approach
    # Now it's almost instant
    series = [h5[patient_id][dcm_series]
              for patient_id in h5.keys()
              for dcm_series in h5[patient_id].keys()]
    print(len(series))

    # Selecting all 'ADC' DICOM series
    adc_series = [h5[patient_id][dcm_series]
                  for patient_id in h5.keys()
                  for dcm_series in h5[patient_id].keys()
                  if 'ADC' in dcm_series]
    print(len(adc_series))

    # Example of how to extract info from a dicom_series_query result
    words = ['ADC', 'cor']
    query = dicom_series_query(h5, words)

    for h5_group in query:
        pixel_array = h5_group['pixel_array'][:]  # The actual DICOM pixel data
        # patient_age = h5_group['pixel_array'].attrs.get('Age')
        lesion_info = [
            [
                # Per lesion finding, gather the attributes necessary for actual lesion extraction from DICOM image
                h5_group['lesions'][finding_id].attrs.get('ijk'),
                h5_group['lesions'][finding_id].attrs.get('VoxelSpacing'),
                h5_group['lesions'][finding_id].attrs.get('ClinSig')
            ]
            for finding_id in h5_group['lesions'].keys()
            ]
        print('{} with {} lesion(s): {}'.format(pixel_array.shape, len(lesion_info), lesion_info))
