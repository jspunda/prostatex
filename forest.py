# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestRegressor
from csv import DictReader

train_csv = '/nfs/home4/schellev/features/features_train.csv'
test_csv = '/nfs/home4/schellev/features/features_test.csv'
n_estimators = 400

def to1hot(zone):
    zones = ['AS', 'PZ', 'SV', 'TZ']
    
    result = [float(zone == e) for e in zones]
    return result
    

train_data = []
train_labels = []
train_proxIds = []

with open(train_csv) as csvfile:
    reader = DictReader(csvfile)
    fields = list(reader.fieldnames)
    fields.remove('proxid')
    fields.remove('clinsig')
    fields.remove('Age') # because convert to float.
    fields.remove('Zone') # categorical data -> 1-hot
    
    for row in reader:
        train_proxIds.append(row.pop('proxid'))
        train_labels.append(row.pop('clinsig'))
        data_item= []
        for field in fields:
            data_item.append(row[field])
        data_item.append(float(row['Age'][:-1]))
        data_item.extend(to1hot(row['Zone']))
        train_data.append(data_item)
        

test_data = []
test_proxIds = []

with open(test_csv) as csvfile:
    reader = DictReader(csvfile)
    fields = list(reader.fieldnames)
    fields.remove('proxid')
    fields.remove('Age') # because convert to float.
    fields.remove('Zone') # categorical data -> 1-hot
    
    for row in reader:
        test_proxIds.append(row.pop('proxid'))
        data_item= []
        for field in fields:
            data_item.append(row[field])
        data_item.append(float(row['Age'][:-1]))
        data_item.extend(to1hot(row['Zone']))
        test_data.append(data_item)
        

forest = RandomForestRegressor(n_estimators = n_estimators)
        
forest.fit(train_data, train_labels)

prediction = forest.predict(test_data)

filename = 'random_forest_age_zone_adc_features_'+str(n_estimators)+'estim.csv'
with open(filename, 'w') as f:
    f.write('proxid,clinsig\n')
    for pair in (zip(test_proxIds, prediction)):
        line = "%s,%f\n" % pair
        f.write(line)
        print(line)
