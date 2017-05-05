import h5py
import collections


class Query:
    """
    This class represents a query to the hdf5 prostatex-train.hdf5 dataset. Returns a dict containing a subset
    of hdf5 dataset that matches the query words.
    """
    def __init__(self, h5_file, query_words):
        self.h5_file = h5_file
        self.query_words = query_words
        self.result = collections.defaultdict(dict)

    def run(self):
        """Returns a dict containing all DICOM series and the lesions within them
        that match self.query_words"""

        for patient in self.h5_file.keys():  # Traverse every patient_id
            for word in self.query_words:  # Query word by word
                for dcm_desc in self.h5_file[patient].keys():  # Per patient_id, traverse all DICOM series names
                    if word in dcm_desc:  # If query word is present in DICOM series name
                        # Extract group containing image data and belonging lesions
                        img = self.h5_file[patient][dcm_desc]
                        self.result[patient][dcm_desc] = {}
                        self.result[patient][dcm_desc] = img  # Append to query result
        return dict(self.result)

    def get_result(self):
        return dict(self.result)

    def print_result(self):
        for patient_id in self.result.keys():
            for dcm_desc in self.result[patient_id]:
                print('For DICOM series {} found {} lesion(s) at:'
                      .format(self.result[patient_id][dcm_desc].name,
                              len(self.result[patient_id][dcm_desc]['lesions'])))
                for finding_id in self.result[patient_id][dcm_desc]['lesions']:
                    print(self.result[patient_id][dcm_desc]['lesions'][finding_id].attrs.get('ijk'))

# Example usage, let's say we want pixel data and lesion attributes for all ADC and cor images
# words = ['ADC', 'cor']
# h5 = h5py.File('C:\Users\Jeftha\stack\Rommel\ISMI\prostatex-train.hdf5', 'r')
#
# q = Query(h5, words)
# result = q.run()
# q.print_result()




### Old Callback approach (can be ignored for now) ###
# def centroid_query_callback(name, obj):
#     global query_words
#     global query_result
#     for word in query_words:
#         if 'lesions/' in name and word in name:
#             split = name.split('/')
#             patient_id = split[0]
#             dcm_desc = split[1]
#             lesion = {
#                 'ijk': obj.attrs.get('ijk'),
#                 'VoxelSpacing': obj.attrs.get('VoxelSpacing'),
#                 'ClinSig': obj.attrs.get('ClinSig')
#             }
#             try:
#                 query_result[patient_id][dcm_desc]['lesions'].append(lesion)
#             except KeyError:
#                 query_result[patient_id][dcm_desc] = {}
#                 query_result[patient_id][dcm_desc]['lesions'] = [lesion]
#             print('Found {} with lesion {}'.format(name, obj.attrs.items()))
#     return None
#
#
# def query_callback(name, obj):
#     global query_words
#     global query_result
#     for word in query_words:
#         if 'pixel_array' in name and word in name:
#             split = name.split('/')
#             patient_id = split[0]
#             dcm_desc = split[1]
#             pixel_data = obj[:]
#             try:
#                 query_result[patient_id][dcm_desc]['pixels'].append(pixel_data)
#             except KeyError:
#                 query_result[patient_id][dcm_desc]['pixels'] = [pixel_data]
#             print('Found {} with shape {}'.format(name, pixel_data.shape))
#     return centroid_query_callback(name, obj)
#
# query_words = ['ADC', 'cor']
# query_result = collections.defaultdict(dict)
#
# h5 = h5py.File('C:\Users\Jeftha\stack\Rommel\ISMI\prostatex-train.hdf5', 'r')
# dict(query_result)
# # h5.visititems(query_callback)
# for key in query_result.keys():
#     for key1 in query_result[key]:
#         for key2 in query_result[key][key1]:
#             if key2 == 'lesions':
#                 print ('Patient {} with dcmdesc {} has lesions {}'.format(key, key1, query_result[key][key1][key2]))
