import h5py
from sklearn import svm
from lesion_extraction_2d.lesion_extractor_2d import get_train_data


h5_file = h5py.File('C:\\Users\\Jeftha\\stack\\Rommel\\ISMI\\prostatex-train.hdf5', 'r')
X, y = get_train_data(h5_file, ['ADC'])

clf = svm.SVC()
clf.fit(X, y)

for i in range(10):
    print("Prediction for %s: %s" % ((i + 1), clf.predict(X[i])))
