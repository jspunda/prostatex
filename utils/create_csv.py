from simplenet import get_model


def predict_to_file(filename):
    m = get_model()

    # Data
    h5_file_location = os.path.join('/scratch-shared/ISMI/prostatex', 'prostatex-test.hdf5')
    h5_file = h5py.File(h5_file_location, 'r')
    x, _, attr = get_train_data(h5_file, ['ADC'])

    predictions = m.predict(x, verbose=1)

    with open(filename, 'wb') as f:
        f.write('proxid,clinsig\n')

        for i in range(len(predictions)):
            patient_id = attr[i]['patient_id']
            fid = attr[i]['fid']

            line = "ProstateX-%s-%s,%f\n" % (patient_id, fid, predictions[i])
            f.write(line)
            print(line)


if __name__ == "__main__":
    predict_to_file('predictions.csv')
