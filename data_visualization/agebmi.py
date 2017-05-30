# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import dicom
os.chdir("C:\\Users\\Patrick\\Desktop\\ProstateX\\DOI")
#plan = dicom.read_file("000000.dcm")
#print plan.PatientWeight
#print plan.PatientSize
#print plan.PatientAge

folders = os.listdir("C:\\Users\\Patrick\\Desktop\\ProstateX\\DOI")

data = []

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
with open('ProstateX-Findings-Train.csv') as csvfile:
    preader = csv.reader(csvfile, delimiter=',')
    for row in preader:
       ldata.append(row)

ldata.pop(0)


fldata = []
fldata.append(ldata[0])

for i in range (1, len(ldata)):
    if ldata[i][0] != ldata[i-1][0]:
        fldata.append(ldata[i])
    


fdata = []
for i in range (0, len(fldata)):
    target = data[i]
    target.append(fldata[i][4])
    fdata.append(target)
                
for i in range (0, len(fdata)):
    fdata[i][1] = int(fdata[i][1][1] + fdata[i][1][2])
    fdata[i][4] = fdata[i][4] == 'TRUE'
    
for i in range (0, len(fdata)):
    if fdata[i][2] != 'unknown':
        if fdata[i][2] > 10.0:
            fdata[i][2] = float(fdata[i][2]) * 0.010

agelist = set([i[1] for i in fdata])
agetable = np.zeros((100,5))

for i in range(0, len(agetable)):
    agetable[i][0] = i * 1.000
    
for i in range(0, len(fdata)):
    agetable[fdata[i][1]][(int(fdata[i][4])+1)%2 + 1] = agetable[fdata[i][1]][(int(fdata[0][4])+1)%2 + 1] + 1
 
for i in range(0, len(agetable)):
    if (float(agetable[i][1]) != 0.0) & (float(agetable[i][2]) != 0.0):
        agetable[i][3] = agetable[i][1] + agetable[i][2]
    
    if (float(agetable[i][3]) != 0.0):
        agetable[i][4] = agetable[i][0] / agetable[i][3]
        
modagetable = np.zeros((42,5))

for i in range (0, len(modagetable)):
    for j in range (0, 5):
        modagetable[i][j] = agetable[i+37][j]
    
    
#[column[0] for column in agetable]

#plt.plot([column[0] for column in agetable], [column[1] for column in agetable])

#plt.plot([column[0] for column in modagetable], [column[1] for column in modagetable])
#plt.plot([column[0] for column in modagetable], [column[2] for column in modagetable])
#plt.plot([column[0] for column in modagetable], [column[3] for column in modagetable])

bmidata = []

for i in range (0, len(fdata)):
    if fdata[i][2] != 'unknown':
        bmidata.append(fdata[i])
        
bmitable = np.zeros((len(bmidata), 4))

for i in range (0, len(bmitable)):
        bmitable[i][0] = bmidata[i][2]
        bmitable[i][1] = bmidata[i][3]
        bmitable[i][2] = bmitable[i][1] / pow(bmitable[i][0], 2)
        bmitable[i][3] = bmidata[i][4]


bmiftable = np.zeros((5,4))

for i in range(0, len(bmitable)):
    if bmitable[i][2] < 20.0:
        bmiftable[0][int(bmitable[i][3])] = bmiftable[0][int(bmitable[i][3])] + 1
    elif (bmitable[i][2] >= 20.0) & (bmitable[i][2] < 25.0):
        bmiftable[1][int(bmitable[i][3])] = bmiftable[1][int(bmitable[i][3])] + 1
    elif (bmitable[i][2] >= 25.0) & (bmitable[i][2] < 30.0):
        bmiftable[2][int(bmitable[i][3])] = bmiftable[2][int(bmitable[i][3])] + 1
    elif (bmitable[i][2] >= 30.0) & (bmitable[i][2] < 35.0):
        bmiftable[3][int(bmitable[i][3])] = bmiftable[3][int(bmitable[i][3])] + 1
    elif bmitable[i][2] >= 35.0:
        bmiftable[4][int(bmitable[i][3])] = bmiftable[4][int(bmitable[i][3])] + 1

for i in range(0, len(bmiftable)):
    bmiftable[i][2] = bmiftable[i][0] + bmiftable[i][1]
    bmiftable[i][3] = bmiftable[i][1] / bmiftable[i][2]
    
finalagetable = np.zeros((7,4))

for i in range(0, len(modagetable)):
    if modagetable[i][0] < 40.0:
        finalagetable[0][0] = finalagetable[0][0] + modagetable[i][1]
        finalagetable[0][1] = finalagetable[0][1] + modagetable[i][2]
    elif (modagetable[i][0] >= 40.0) & (modagetable[i][0] < 50.0):
        finalagetable[1][0] = finalagetable[1][0] + modagetable[i][1]
        finalagetable[1][1] = finalagetable[1][1] + modagetable[i][2]
    elif (modagetable[i][0] >= 50.0) & (modagetable[i][0] < 55.0):
        finalagetable[2][0] = finalagetable[2][0] + modagetable[i][1]
        finalagetable[2][1] = finalagetable[2][1] + modagetable[i][2]
    elif (modagetable[i][0] >= 55.0) & (modagetable[i][0] < 60.0):
        finalagetable[3][0] = finalagetable[3][0] + modagetable[i][1]
        finalagetable[3][1] = finalagetable[3][1] + modagetable[i][2]        
    elif (modagetable[i][0] >= 60.0) & (modagetable[i][0] < 65.0):
        finalagetable[4][0] = finalagetable[4][0] + modagetable[i][1]
        finalagetable[4][1] = finalagetable[4][1] + modagetable[i][2]
    elif (modagetable[i][0] >= 65.0) & (modagetable[i][0] < 70.0):
        finalagetable[5][0] = finalagetable[5][0] + modagetable[i][1]
        finalagetable[5][1] = finalagetable[5][1] + modagetable[i][2]
    elif modagetable[i][0] >= 70.0:
        finalagetable[6][0] = finalagetable[6][0] + modagetable[i][1]
        finalagetable[6][1] = finalagetable[6][1] + modagetable[i][2]

for i in range(0, len(finalagetable)):
    finalagetable[i][2] = finalagetable[i][0] + finalagetable[i][1]
    finalagetable[i][3] = finalagetable[i][0] / finalagetable[i][2]