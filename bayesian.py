# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from collections import OrderedDict
from operator import itemgetter
import numpy as np
import csv
import os
import dicom
import h5py
import math

class Centroid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)

def extract_lesion_2d(img, centroid_position, size=None, realsize=16, imagetype='ADC'):
    if imagetype == 'ADC':
        if size is None:
            sizecal = math.ceil(realsize / 2)
        else:
            sizecal = size
    else:
        sizecal = size
    x_start = int(centroid_position.x - sizecal / 2)
    x_end = int(centroid_position.x + sizecal / 2)
    y_start = int(centroid_position.y - sizecal / 2)
    y_end = int(centroid_position.y + sizecal / 2)

    if centroid_position.z < 0 or centroid_position.z >= len(img):
        return None

    img_slice = img[centroid_position.z]

    return img_slice[y_start:y_end, x_start:x_end]

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

def parse_centroid(ijk):
    coordinates = ijk.split(b" ")
    return Centroid(int(coordinates[0]), int(coordinates[1]), int(coordinates[2]))

def get_train_data(h5_file, query_words, size_px=16):
    lesion_info = get_lesion_info(h5_file, query_words)

    X = []
    y = []
    lesion_attributes = []
    previous_patient = ''
    for infos, image in lesion_info:
        current_patient = infos[0]['name'].split('/')[1]
        if current_patient == previous_patient:
            print('Warning in {}: Found duplicate match for {}. Skipping...'
                  .format(get_train_data.__name__, current_patient))
            continue
        for lesion in infos:

            centroid = parse_centroid(lesion['ijk'])
            lesion_img = extract_lesion_2d(image, centroid, size=size_px)
            if lesion_img is None:
                print('Warning in {}: ijk out of bounds for {}. No lesion extracted'
                      .format(get_train_data.__name__, lesion))
                continue

            X.append(lesion_img)

            lesion_attributes.append(lesion)

            y.append(lesion['ClinSig'] == b"TRUE")

        previous_patient = current_patient

    return np.asarray(X), np.asarray(y), np.asarray(lesion_attributes)

def get_pixels_in_window(np_array, window):
    pixels = np_array[(window[0] < np_array) & (np_array < window[1])]
    if len(pixels) != 0:  # If no pixels within window remain (i.e. lesion does not show up with current window)
        return pixels
    else:
        return None
        
def get_adc_features(h5_file_path, window, lesion_size):
    """
    Gathers amount of pixels within window and the mean of those pixels for ADC images.
    Returns an ordered dictionary of the form:
    'ProstateX-0000-1'/
    |
    |---'mean'/
    |   |
    |   |--- Mean value of pixels in window
    |
    ---'count'/
    |   |
    |   |--- Amount of pixels within window
    |
    'ProstateX-0001-1'/
    |
    |--- ...
    .
    .
    .
    """
    h5_file = h5py.File(h5_file_path, 'r')
    x, _, attrs = get_train_data(h5_file, ['ADC'], size_px=lesion_size)

    feature_dict = OrderedDict()
    for i in range(len(attrs)):
        lesion = x[i]
        attr = attrs[i]
        prox_id = 'ProstateX-{}-{}'.format(attr['patient_id'], attr['fid'])
        pixels_in_window = get_pixels_in_window(lesion, window)
        feature_dict[prox_id] = {}
        if pixels_in_window is not None:
            feature_dict[prox_id]['count'] = len(pixels_in_window)
            feature_dict[prox_id]['mean'] = pixels_in_window.mean()
        else:
            feature_dict[prox_id]['count'] = 0
            feature_dict[prox_id]['mean'] = 0

    return feature_dict

h5_path = "C:\\Users\\Patrick\\Desktop\\ProstateX\\prostatex-train.hdf5"
featurestrain = get_adc_features(h5_path, (400, 1100), 8)

h5_path = "C:\\Users\\Patrick\\Desktop\\ProstateX\\prostatex-test.hdf5"
featurestest = get_adc_features(h5_path, (400, 1100), 8)

#Puts the extra features of the h5 test and training set in a dictionary for later use.
features = dict(featurestrain.items() + featurestest.items())

os.chdir("C:\\Users\\Patrick\\Desktop\\ProstateX\\DOI")

folders = os.listdir("C:\\Users\\Patrick\\Desktop\\ProstateX\\DOI")

data = []

#Goes through the folders to read the dicom header from the first file in the first folder
for i in range (0, len(folders)):
    dirname = "C:\\Users\\Patrick\\Desktop\\ProstateX\\DOI\\" + folders[i]
    os.chdir(dirname)
    subfolders = os.listdir(dirname)
    dirname = dirname + "\\" + subfolders[0]
    os.chdir(dirname)
    subsubfolders = os.listdir(dirname)
    dirname = dirname + "\\" + subsubfolders[0]
    os.chdir(dirname)
    target = dicom.read_file(os.listdir(dirname)[0])
    
    PatientData = []
    
    PatientData.append(target.PatientID)
    PatientData.append(target.PatientAge)
    
    
    if(hasattr(target, 'PatientSize')):
        PatientData.append(target.PatientSize)
    else:
        PatientData.append("unknown")
    
    PatientData.append(target.PatientWeight)
            
    data.append(PatientData)

os.chdir("C:\\Users\\Patrick\\Desktop\\ProstateX")

ldata = []

#Reads the findings of the training set so that zone information and labels can be tied to patient ID
with open('ProstateX-Findings-Train.csv') as csvfile:
    preader = csv.reader(csvfile, delimiter=';')
    for row in preader:
        ldata.append(row)

ldata.pop(0)
poptar =  len(ldata)

#Reads the findings of the test set to get the zone information
with open('ProstateX-Findings-Test.csv') as csvfile:
    preader = csv.reader(csvfile, delimiter=',')
    for row in preader:
        row.append("unknown")
        ldata.append(row)

#Establishes where the first row of the testset will be.
ldata.pop(poptar)

ldata2 = ldata

prostiCount = 0

#Keeps track of the number of times something goes wrong. Should be 0 after running this.
woepscount = 0

bdata = []

#Combines the findings with the data from the dicom headers based on patient ID
for i in range (0, len(ldata2)):
    btarget = []
    if ldata2[i][0] == data[prostiCount][0]:
        btarget = ldata2[i] + data[prostiCount][1:]
        bdata.append(btarget)
    else:
        while ldata2[i][0] != data[prostiCount][0]:
            prostiCount = prostiCount + 1
            
        if ldata2[i][0] == data[prostiCount][0]:
            btarget = ldata2[i] + data[prostiCount][1:]
            bdata.append(btarget)
        else:
            print(ldata2[i][0] + " " + data[prostiCount][0])
            woepscount += 1

#Some manipulations to organize the table
for row in bdata:
    row[0] = row[0] + "-" + row[1]
    if row[0] in features:
        row[1] = features[row[0]]["count"]
        row[2] = features[row[0]]["mean"]
    else:
        row[1] = "unknown"
        row[2] = "unknown"
    
    if (row[6] != "unknown" and row[6] > 3):
        row[6] = row[6] / 100
    row[5] = int(row[5][1:-1])
    if (row[6] != "unknown"):
        row.append(row[7] / pow(row[6], 2))
    else:
        row.append("unknown")
    row[4], row[6] = row[8], row[4]
    del row[7]
    del row[7]
    if (row[6] == "TRUE"):
        row[6] = True
    elif(row[6] == "FALSE"):
        row[6] = False
    else:
        row[6] = "unknown"


            
bprior = bdata[:330]
bpred = bdata[330:]

pTrue = 0
pFalse = 0
total = len(bprior) * 1.0

#Gets a generalized probability distribution.
for i in range (len(bprior)):
    if (bprior[i][6] == True):
        pTrue = pTrue + 1
    else:
        pFalse = pFalse + 1

if (pTrue + pFalse == len(bprior)):
    pTrueNum = pTrue
    pFalseNum = pFalse
    pTrue = pTrue / total
    pFalse = pFalse / total

bpriorcount = sorted(bprior, key=itemgetter(1))
while (bpriorcount[len(bpriorcount) - 1][1] == "unknown"):
    bpriorcount.pop()

countTable = np.zeros((4,2))

# There are 329 cases with known count. To get 4 bins of roughly equal size we need to cut at 29 45 64
# with 16 px and window(500,1100)

# with 8 px and window(400,1100) it is 8, 17 and 29

for i in range (len(bpriorcount)):
    if (bpriorcount[i][1] < 8):
        if (bpriorcount[i][6] == True):
            countTable[0][0] = countTable[0][0] + 1
        elif (bpriorcount[i][6] == False):
            countTable[0][1] = countTable[0][1] + 1
    if (bpriorcount[i][1] >= 8 and bpriorcount[i][1] < 17):
        if (bpriorcount[i][6] == True):
            countTable[1][0] = countTable[1][0] + 1
        elif (bpriorcount[i][6] == False):
            countTable[1][1] = countTable[1][1] + 1
    if (bpriorcount[i][1] >= 17 and bpriorcount[i][1] < 29):
        if (bpriorcount[i][6] == True):
            countTable[2][0] = countTable[2][0] + 1
        elif (bpriorcount[i][6] == False):
            countTable[2][1] = countTable[2][1] + 1
    if (bpriorcount[i][1] >= 29):
        if (bpriorcount[i][6] == True):
            countTable[3][0] = countTable[3][0] + 1
        elif (bpriorcount[i][6] == False):
            countTable[3][1] = countTable[3][1] + 1

count1 = countTable[0][0] / (countTable[0][0] + countTable[0][1])
count2 = countTable[1][0] / (countTable[1][0] + countTable[1][1])
count3 = countTable[2][0] / (countTable[2][0] + countTable[2][1])
count4 = countTable[3][0] / (countTable[3][0] + countTable[3][1])

bpriormean = bprior

for i in range (0, len(bpriormean)):
    if (bpriormean[i][2] != "unknown"):
        bpriormean[i][2] = float(bpriormean[i][2])

bpriormean = sorted(bpriormean, key=itemgetter(2))
while (bpriormean[len(bpriormean) - 1][1] == "unknown"):
    bpriormean.pop()

meanTable = np.zeros((4,2))

# There are 329 cases with known mean. To get 4 bins of roughly equal size we need to cut at 922, 963 and 989
# with 16 px and window(500,1100)

# with 8 px and window(400,1100) it is 898, 956, 966

for i in range (len(bpriormean)):
    if (bpriormean[i][2] < 898):
        if (bpriormean[i][6] == True):
            meanTable[0][0] = meanTable[0][0] + 1
        elif (bpriormean[i][6] == False):
            meanTable[0][1] = meanTable[0][1] + 1
    if (bpriormean[i][2] >= 898 and bpriormean[i][2] < 956):
        if (bpriormean[i][6] == True):
            meanTable[1][0] = meanTable[1][0] + 1
        elif (bpriormean[i][6] == False):
            meanTable[1][1] = meanTable[1][1] + 1
    if (bpriormean[i][2] >= 956 and bpriormean[i][2] < 966):
        if (bpriormean[i][6] == True):
            meanTable[2][0] = meanTable[2][0] + 1
        elif (bpriormean[i][6] == False):
            meanTable[2][1] = meanTable[2][1] + 1
    if (bpriormean[i][2] >= 966):
        if (bpriormean[i][6] == True):
            meanTable[3][0] = meanTable[3][0] + 1
        elif (bpriormean[i][6] == False):
            meanTable[3][1] = meanTable[3][1] + 1

mean1 = meanTable[0][0] / (meanTable[0][0] + meanTable[0][1])
mean2 = meanTable[1][0] / (meanTable[1][0] + meanTable[1][1])
mean3 = meanTable[2][0] / (meanTable[2][0] + meanTable[2][1])
mean4 = meanTable[3][0] / (meanTable[3][0] + meanTable[3][1])


bpriorbmi = sorted(bprior, key=itemgetter(4))
while (bpriorbmi[len(bpriorbmi) - 1][4] == "unknown"):
    bpriorbmi.pop()

bmiTable = np.zeros((4,2))

# There are 324 cases with known BMI. To get 4 bins of roughly equal size we need to cut at 24.5 26.25 28.00
for i in range (len(bpriorbmi)):
    if (bpriorbmi[i][4] < 24.5):
        if (bpriorbmi[i][6] == True):
            bmiTable[0][0] = bmiTable[0][0] + 1
        elif (bpriorbmi[i][6] == False):
            bmiTable[0][1] = bmiTable[0][1] + 1
    if (bpriorbmi[i][4] >= 24.5 and bpriorbmi[i][4] < 26.25):
        if (bpriorbmi[i][6] == True):
            bmiTable[1][0] = bmiTable[1][0] + 1
        elif (bpriorbmi[i][6] == False):
            bmiTable[1][1] = bmiTable[1][1] + 1
    if (bpriorbmi[i][4] >= 26.25 and bpriorbmi[i][4] < 28.00):
        if (bpriorbmi[i][6] == True):
            bmiTable[2][0] = bmiTable[2][0] + 1
        elif (bpriorbmi[i][6] == False):
            bmiTable[2][1] = bmiTable[2][1] + 1
    if (bpriorbmi[i][4] >= 28.00):
        if (bpriorbmi[i][6] == True):
            bmiTable[3][0] = bmiTable[3][0] + 1
        elif (bpriorbmi[i][6] == False):
            bmiTable[3][1] = bmiTable[3][1] + 1

BMI1 = bmiTable[0][0] / (bmiTable[0][0] + bmiTable[0][1])
BMI2 = bmiTable[1][0] / (bmiTable[1][0] + bmiTable[1][1])
BMI3 = bmiTable[2][0] / (bmiTable[2][0] + bmiTable[2][1])
BMI4 = bmiTable[3][0] / (bmiTable[3][0] + bmiTable[3][1])

bpriorage = sorted(bprior, key=itemgetter(5))
# There are 330 cases with known ages. To get 4 bins of roughly equal size we need to cut at 60, 65, 69

ageTable = np.zeros((4,2))

for i in range (len(bpriorage)):
    if (bpriorage[i][5] < 60):
        if (bpriorage[i][6] == True):
            ageTable[0][0] = ageTable[0][0] + 1
        elif (bpriorage[i][6] == False):
            ageTable[0][1] = ageTable[0][1] + 1
    if (bpriorage[i][5] >= 60 and bpriorage[i][5] < 65):
        if (bpriorage[i][6] == True):
            ageTable[1][0] = ageTable[1][0] + 1
        elif (bpriorage[i][6] == False):
            ageTable[1][1] = ageTable[1][1] + 1
    if (bpriorage[i][5] >= 65 and bpriorage[i][5] < 69):
        if (bpriorage[i][6] == True):
            ageTable[2][0] = ageTable[2][0] + 1
        elif (bpriorage[i][6] == False):
            ageTable[2][1] = ageTable[2][1] + 1
    if (bpriorage[i][5] >= 69):
        if (bpriorage[i][6] == True):
            ageTable[3][0] = ageTable[3][0] + 1
        elif (bpriorage[i][6] == False):
            ageTable[3][1] = ageTable[3][1] + 1

Age1 = ageTable[0][0] / (ageTable[0][0] + ageTable[0][1])
Age2 = ageTable[1][0] / (ageTable[1][0] + ageTable[1][1])
Age3 = ageTable[2][0] / (ageTable[2][0] + ageTable[2][1])
Age4 = ageTable[3][0] / (ageTable[3][0] + ageTable[3][1])

zoneTable = np.zeros((4,2))

for i in range (len(bprior)):
    if (bprior[i][3] == "AS"):
        if (bprior[i][6] == True):
            zoneTable[0][0] = zoneTable[0][0] + 1
        elif (bprior[i][6] == False):
            zoneTable[0][1] = zoneTable[0][1] + 1
    if (bprior[i][3] == "PZ"):
        if (bprior[i][6] == True):
            zoneTable[1][0] = zoneTable[1][0] + 1
        elif (bprior[i][6] == False):
            zoneTable[1][1] = zoneTable[1][1] + 1
    if (bprior[i][3] == "SV"):
        if (bprior[i][6] == True):
            zoneTable[2][0] = zoneTable[2][0] + 1
        elif (bprior[i][6] == False):
            zoneTable[2][1] = zoneTable[2][1] + 1
    if (bprior[i][3] == "TZ"):
        if (bprior[i][6] == True):
            zoneTable[3][0] = zoneTable[3][0] + 1
        elif (bprior[i][6] == False):
            zoneTable[3][1] = zoneTable[3][1] + 1

AS = zoneTable[0][0] / (zoneTable[0][0] + zoneTable[0][1])
PZ = zoneTable[1][0] / (zoneTable[1][0] + zoneTable[1][1])
SV = zoneTable[2][0] / (zoneTable[2][0] + zoneTable[2][1])
TZ = zoneTable[3][0] / (zoneTable[3][0] + zoneTable[3][1])

predictions = []

for i in range (0, len(bpred)):
    
    if (bpred[i][1] < 29):
        pcount = count1
    if (bpred[i][1] >= 29 and bpred[i][1] < 45):
        pcount = count2
    if (bpred[i][1] >= 45 and bpred[i][1] < 64):
        pcount = count3
    if (bpred[i][1] >= 64):
        pcount = count4
    if (bpred[i][1] == "unknown"):
        pcount = pTrue
    
    if (bpred[i][2] < 922):
        pmean = mean1
    if (bpred[i][2] >= 922 and bpred[i][2] < 963):
        pmean = mean2
    if (bpred[i][2] >= 963 and bpred[i][2] < 989):
        pmean = mean3
    if (bpred[i][2] >= 989):
        pmean = mean4
    if (bpred[i][2] == "unknown"):
        pmean = pTrue
    
    if (bpred[i][3] == "AS"):
        pZone = AS
    if (bpred[i][3] == "PZ"):
        pZone = PZ
    if (bpred[i][3] == "SV"):
        pZone = SV
    if (bpred[i][3] == "TZ"):
        pZone = TZ
    
    if (bpred[i][4] < 24.5):
        pBMI = BMI1
    if (bpred[i][4] >= 24.5 and bpred[i][2] < 26.25):
        pBMI = BMI2
    if (bpred[i][4] >= 26.25 and bpred[i][2] < 28.00):
        pBMI = BMI3
    if (bpred[i][4] >= 28.00):
        pBMI = BMI4
    if (bpred[i][4] == "unknown"):
        pBMI = pTrue

    if (bpred[i][5] < 60):
        pAge = Age1
    if (bpred[i][5] >= 60 and bpred[i][3] < 65):
        pAge = Age2
    if (bpred[i][5] >= 65 and bpred[i][3] < 69):
        pAge = Age3
    if (bpred[i][5] >= 69):
        pAge = Age4
    
    #This calculates the final prediction of the Bayesian classifier.
    p = pcount + pmean + pZone + pBMI + pAge
    - (pcount * pmean) - (pcount * pZone) - (pcount * pBMI) - (pcount * pAge)
    - (pmean * pZone) - (pmean * pBMI) - (pmean * pAge)
    - (pZone * pBMI) - (pZone * pAge)
    - (pBMI * pAge)
    + (pcount * pmean * pZone) + (pcount * pmean * pBMI) + (pcount * pmean * pAge)
    + (pcount * pZone * pBMI) + (pcount * pZone * pAge)
    + (pcount * pBMI * pAge)
    + (pmean * pZone * pBMI) + (pmean * pZone * pAge)
    + (pmean * pBMI * pAge)
    + (pZone * pBMI * pAge)
    - (pcount * pmean * pZone * pBMI)
    - (pcount * pmean * pZone * pAge)
    - (pcount * pmean * pBMI * pAge)
    - (pcount * pZone * pBMI * pAge)
    - (pmean * pZone * pBMI * pAge)
    + (pcount * pmean * pZone * pBMI * pAge)
    
    predictions.append((bpred[i][0], p))

#Writes it into a file.
with open('bayesianpredictions.csv', 'wb') as csvfile:
    bwriter = csv.writer(csvfile)
    bwriter.writerow(['proxid', 'clinsig'])
    for row in predictions:
        bwriter.writerow(row)



