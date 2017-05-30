# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:29:46 2017

@author: Patrick
"""

import os
import csv
import numpy as np

data=[]
with open('ProstateX-Findings-Train.csv') as csvfile:
    preader = csv.reader(csvfile, delimiter=',')
    for row in preader:
       data.append(row)
    
data_matrix = [['Zone', 'True', 'False'],
               ['AS', 0, 0],
               ['PZ', 0, 0],
               ['SV', 0, 0],
               ['TZ', 0, 0]]    
    
for i in range (1, len(data)):
    if (data[i][3] == 'AS'):
        zone = 1
    elif (data[i][3] == 'PZ'):
        zone = 2
    elif (data[i][3] == 'SV'):
        zone = 3
    elif (data[i][3] == 'TZ'):
        zone = 4
        
    if (data[i][4] == 'TRUE'):
        value = 1
    elif (data[i][4] == 'FALSE'):
        value = 2
        
    data_matrix[zone][value] = data_matrix[zone][value] + 1

for i in range(0, len(data_matrix)):
    line = data_matrix[i][0] + '\t' + str(data_matrix[i][1]) + '\t' + str(data_matrix[i][2])
    print line

print

line = data_matrix[i][0] + '\t' + (data_matrix[0][1]) + '\t\t\t' + (data_matrix[0][2])
print line

for i in range(1, len(data_matrix)):
    line = data_matrix[i][0] + '\t' + str(data_matrix[i][1] / 330.0) + '\t\t' + str(data_matrix[i][2] / 330.0)
    print line