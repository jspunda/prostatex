# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 21:32:24 2017

@author: Patrick
"""

from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import dicom
os.chdir("C:\\Users\\Patrick\\Desktop\\ProstateX\\")

preds = []

#reads the .csv files to be ensembled
with open('predictions.csv') as csvfile:
    preader = csv.reader(csvfile, delimiter=',')
    for row in preader:
        preds.append(row)

bayesian_preds = []

with open('bayesianpredictions.csv') as csvfile:
    preader = csv.reader(csvfile, delimiter=',')
    for row in preader:
        bayesian_preds.append(row)

ensemble = []

with open('empty.csv') as csvfile:
    preader = csv.reader(csvfile, delimiter=',')
    for row in preader:
        ensemble.append(row)

#assigns weights and normalizes it all
weight_preds = 0.5
weight_bayesian_preds = 0.5
sum_weights = weight_preds + weight_bayesian_preds
normalization_factor = 1.00 / sum_weights


for i in range (1, len(ensemble)):
    pred = float(preds[i][1]) * weight_preds * normalization_factor
    bpred = float(bayesian_preds[i][1]) * weight_bayesian_preds * normalization_factor
    ensembled = pred + bpred
    ensemble[i].append(ensembled)

#writes the ensembled predictions into a .csv file
with open('ensemble.csv', 'wb') as csvfile:
    bwriter = csv.writer(csvfile)
    for row in ensemble:
        bwriter.writerow(row)