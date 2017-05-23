import h5py
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from lesion_extraction_2d.lesion_extractor_2d import get_train_data


h5_file = h5py.File('C:\\Users\\Jeftha\\stack\\Rommel\\ISMI\\prostatex-train.hdf5', 'r')
X, y = get_train_data(h5_file, ['ADC'])

X = np.asarray(X)
X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
y = np.asarray(y)

skf = StratifiedKFold(n_splits=10)

cumulative_acc = 0
i = 0
for train, test in skf.split(X, y):
    clf = svm.SVC()
    clf.fit(X[train], y[train])

    correct = 0

    predictions = clf.predict(X[test])
    for i in range(len(predictions)):
        print("Prediction for %s: %s Actual: %s" % ((i + 1), predictions[i], y[test][i]))

        if y[test][i] == predictions[i]:
            correct += 1

    acc = (correct / len(X[test])) * 100
    print("Correct: %d%% , aantal: %d" % (acc, len(X[test])))

    cumulative_acc += acc

    i += 1

print("Avg acc: %d" % (cumulative_acc / i))
