import h5py
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from lesion_extraction_2d.lesion_extractor_2d import get_train_data


h5_file = h5py.File('C:\\Users\\Jeftha\\stack\\Rommel\\ISMI\\prostatex-train.hdf5', 'r')
X, y = get_train_data(h5_file, ['ADC'])

X = np.asarray(X)
X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = svm.SVC()
clf.fit(X_train, y_train)

correct = 0

predictions = clf.predict(X_test)
for i in range(len(predictions)):
    print("Prediction for %s: %s Actual: %s" % ((i + 1), predictions[i], y_test[i]))

    if y_test[i] == predictions[i]:
        correct += 1

print("Correct: %d , aantal: %d" % (correct, len(X_test)))
