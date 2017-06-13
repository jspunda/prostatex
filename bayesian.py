# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import dicom
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
    preader = csv.reader(csvfile, delimiter=',')
    for row in preader:
        ldata.append(row)

ldata.pop(0)

#Reads the findings of the test set to get the zone information
with open('ProstateX-Findings-Test.csv') as csvfile:
    preader = csv.reader(csvfile, delimiter=',')
    for row in preader:
        row.append("unknown")
        ldata.append(row)

ldata.pop(330)

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
    del row[1]
    del row[1]
    if (row[4] != "unknown" and row[4] > 3):
        row[4] = row[4] / 100
    row[3] = int(row[3][1:-1])
    if (row[4] != "unknown"):
        row.append(row[5] / pow(row[4], 2))
    else:
        row.append("unknown")
    row[2], row[4] = row[6], row[2]
    del row[5]
    del row[5]
    if (row[4] == "TRUE"):
        row[4] = True
    elif(row[4] == "FALSE"):
        row[4] = False
    else:
        row[4] = "unknown"

mark = 1
bdata[0][0] = bdata[0][0] + "-1"

#This is so that lesions in the same patient are numbered correctly.
for i in range (1, len(bdata)):
    if (bdata[i][0][13] == bdata[i - 1][0][13]):
        mark = mark + 1
        bdata[i][0] = bdata[i][0] + "-" + str(mark)
    else:
        mark = 1
        bdata[i][0] = bdata[i][0] + "-" + str(mark)

bprior = bdata[:330]
bpred = bdata[330:]

pTrue = 0
pFalse = 0
total = len(bprior) * 1.0

#Gets a generalized probability distribution.
for i in range (len(bprior)):
    if (bprior[i][4] == True):
        pTrue = pTrue + 1
    else:
        pFalse = pFalse + 1

if (pTrue + pFalse == len(bprior)):
    pTrueNum = pTrue
    pFalseNum = pFalse
    pTrue = pTrue / total
    pFalse = pFalse / total

bpriorbmi = sorted(bprior, key=itemgetter(2))
while (bpriorbmi[len(bpriorbmi) - 1][2] == "unknown"):
    bpriorbmi.pop()

bmiTable = np.zeros((4,2))

# There are 324 cases with known BMI. To get 4 bins of roughly equal size we need to cut at 24.5 26.25 28.00
for i in range (len(bpriorbmi)):
    if (bpriorbmi[i][2] < 24.5):
        if (bpriorbmi[i][4] == True):
            bmiTable[0][0] = bmiTable[0][0] + 1
        elif (bpriorbmi[i][4] == False):
            bmiTable[0][1] = bmiTable[0][1] + 1
    if (bpriorbmi[i][2] >= 24.5 and bpriorbmi[i][2] < 26.25):
        if (bpriorbmi[i][4] == True):
            bmiTable[1][0] = bmiTable[1][0] + 1
        elif (bpriorbmi[i][4] == False):
            bmiTable[1][1] = bmiTable[1][1] + 1
    if (bpriorbmi[i][2] >= 26.25 and bpriorbmi[i][2] < 28.00):
        if (bpriorbmi[i][4] == True):
            bmiTable[2][0] = bmiTable[2][0] + 1
        elif (bpriorbmi[i][4] == False):
            bmiTable[2][1] = bmiTable[2][1] + 1
    if (bpriorbmi[i][2] >= 28.00):
        if (bpriorbmi[i][4] == True):
            bmiTable[3][0] = bmiTable[3][0] + 1
        elif (bpriorbmi[i][4] == False):
            bmiTable[3][1] = bmiTable[3][1] + 1

BMI1 = bmiTable[0][0] / (bmiTable[0][0] + bmiTable[0][1])
BMI2 = bmiTable[1][0] / (bmiTable[1][0] + bmiTable[1][1])
BMI3 = bmiTable[2][0] / (bmiTable[2][0] + bmiTable[2][1])
BMI4 = bmiTable[3][0] / (bmiTable[3][0] + bmiTable[3][1])

bpriorage = sorted(bprior, key=itemgetter(3))
# There are 330 cases with known ages. To get 4 bins of roughly equal size we need to cut at 60, 65, 69

ageTable = np.zeros((4,2))

for i in range (len(bpriorage)):
    if (bpriorage[i][3] < 60):
        if (bpriorage[i][4] == True):
            ageTable[0][0] = ageTable[0][0] + 1
        elif (bpriorage[i][4] == False):
            ageTable[0][1] = ageTable[0][1] + 1
    if (bpriorage[i][3] >= 60 and bpriorage[i][3] < 65):
        if (bpriorage[i][4] == True):
            ageTable[1][0] = ageTable[1][0] + 1
        elif (bpriorage[i][4] == False):
            ageTable[1][1] = ageTable[1][1] + 1
    if (bpriorage[i][3] >= 65 and bpriorage[i][3] < 69):
        if (bpriorage[i][4] == True):
            ageTable[2][0] = ageTable[2][0] + 1
        elif (bpriorage[i][4] == False):
            ageTable[2][1] = ageTable[2][1] + 1
    if (bpriorage[i][3] >= 69):
        if (bpriorage[i][4] == True):
            ageTable[3][0] = ageTable[3][0] + 1
        elif (bpriorage[i][4] == False):
            ageTable[3][1] = ageTable[3][1] + 1

Age1 = ageTable[0][0] / (ageTable[0][0] + ageTable[0][1])
Age2 = ageTable[1][0] / (ageTable[1][0] + ageTable[1][1])
Age3 = ageTable[2][0] / (ageTable[2][0] + ageTable[2][1])
Age4 = ageTable[3][0] / (ageTable[3][0] + ageTable[3][1])

zoneTable = np.zeros((4,2))

for i in range (len(bprior)):
    if (bprior[i][1] == "AS"):
        if (bprior[i][4] == True):
            zoneTable[0][0] = zoneTable[0][0] + 1
        elif (bprior[i][4] == False):
            zoneTable[0][1] = zoneTable[0][1] + 1
    if (bprior[i][1] == "PZ"):
        if (bprior[i][4] == True):
            zoneTable[1][0] = zoneTable[1][0] + 1
        elif (bprior[i][4] == False):
            zoneTable[1][1] = zoneTable[1][1] + 1
    if (bprior[i][1] == "SV"):
        if (bprior[i][4] == True):
            zoneTable[2][0] = zoneTable[2][0] + 1
        elif (bprior[i][4] == False):
            zoneTable[2][1] = zoneTable[2][1] + 1
    if (bprior[i][1] == "TZ"):
        if (bprior[i][4] == True):
            zoneTable[3][0] = zoneTable[3][0] + 1
        elif (bprior[i][4] == False):
            zoneTable[3][1] = zoneTable[3][1] + 1

AS = zoneTable[0][0] / (zoneTable[0][0] + zoneTable[0][1])
PZ = zoneTable[1][0] / (zoneTable[1][0] + zoneTable[1][1])
SV = zoneTable[2][0] / (zoneTable[2][0] + zoneTable[2][1])
TZ = zoneTable[3][0] / (zoneTable[3][0] + zoneTable[3][1])

predictions = []

for i in range (0, len(bpred)):
    if (bpred[i][1] == "AS"):
        pZone = AS
    if (bpred[i][1] == "PZ"):
        pZone = PZ
    if (bpred[i][1] == "SV"):
        pZone = SV
    if (bpred[i][1] == "TZ"):
        pZone = TZ
    
    if (bpred[i][2] < 24.5):
        pBMI = BMI1
    if (bpred[i][2] >= 24.5 and bpred[i][2] < 26.25):
        pBMI = BMI2
    if (bpred[i][2] >= 26.25 and bpred[i][2] < 28.00):
        pBMI = BMI3
    if (bpred[i][2] >= 28.00):
        pBMI = BMI4
    if (bpred[i][2] == "unknown"):
        pBMI = pTrue

    if (bpred[i][3] < 60):
        pAge = Age1
    if (bpred[i][3] >= 60 and bpred[i][3] < 65):
        pAge = Age2
    if (bpred[i][3] >= 65 and bpred[i][3] < 69):
        pAge = Age3
    if (bpred[i][3] >= 69):
        pAge = Age4
    
    #This calculates the final prediction of the Bayesian classifier.
    p = pZone + pBMI + pAge - (pZone * pBMI) - (pZone * pAge) - (pBMI * pAge) + (pZone * pBMI * pAge)
    
    predictions.append((bpred[i][0], p))

#Writes it into a file.
with open('bayesianpredictions.csv', 'wb') as csvfile:
    bwriter = csv.writer(csvfile)
    bwriter.writerow(['proxid', 'clinsig'])
    for row in predictions:
        bwriter.writerow(row)