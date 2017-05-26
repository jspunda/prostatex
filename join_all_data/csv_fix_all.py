import csv
import glob
import SimpleITK as sitk
import sys


#orig_stdout = sys.stdout
#f = open('new_csv_new_out.txt', 'w')
#sys.stdout = f

""" This script combines all info from the 3 csv files, either using the ProstateX-Images-Train-NEW.csv file created by
    h5_converter.csv_fix.py or, in case you don't have this file, using all the 3 original csv files (Ktrans, images and findings)
    
    That's why it uses try: except:"""

main_path = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Training\\All_training_data\\'
NEW_train_csv = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Training' \
                 '\\ProstateX-TrainingLesionInformationv2\\ProstateX-Images-Train-NEW.csv'
ktrans_train_csv = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Training' \
                   '\\ProstateX-TrainingLesionInformationv2\\ProstateX-Images-Ktrans-Train.csv'
#images_train_csv = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Training' \
#                   '\\ProstateX-TrainingLesionInformationv2\\ProstateX-Images-Train.csv'
#findings_train_csv = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Training' \
#                     '\ProstateX-TrainingLesionInformationv2\ProstateX-Findings-Train.csv'

def join_rows(reader,ktrans_lst):
    new_lst = [['ProstateX-0000', 'a', 'b','c']]

    for row in reader:
        try:
            last_row = reader.line_num == 3870
        except AttributeError:
            last_row = row == reader[-1]
        if new_lst[-1][0] != row[0] or last_row:
            aux_lst = [x for x in ktrans_lst if x[0] == new_lst[-1][0]]
            for x in aux_lst:
                ref = [a for a in new_lst if (a[0] == x[0] and a[2] == x[2] and a[3] == x[3])]
                x.insert(12, ref[0][-2])
                x.insert(13, ref[0][-1])
                new_lst.append(x)
        new_lst.append(row)

    new_lst = new_lst[1:]

    return new_lst

""" Let's add the missing information in the Ktrans csv file row by row (some of it has to be taken from the .mhd file)
And combine together the complete Ktrans csv info with the NEW csv file"""

with open(ktrans_train_csv, 'rb') as ktrans_train:
    reader_ktrans = csv.reader(ktrans_train, delimiter=',')
    reader_ktrans.next()

    """ Ktrans is missing some info with respect to the DICOM series, with I filled as good as I could"""
    ktrans_lst = []
    for row in reader_ktrans:
        case_dir = main_path + row[0] + '\\Ktrans'
        mhd_dir = glob.glob(main_path + row[0] + '\\Ktrans' + '/*.mhd')
        img = sitk.ReadImage(mhd_dir[0])
        VoxelSpacing = str(img.GetSpacing()).replace(', ', ',').replace('(', '').replace(')', '')
        Dimen = str(img.GetSize()).replace(', ', 'x').replace('(', '').replace(')', '')
        row.insert(1, 'Ktrans')  # add name
        row.insert(6, 1)  # add TopLevel (chosen 1 because seems to be the only value)
        row.insert(7, 'Nan')  # add SpacingBetweenSlides ('Nan' because thsi info is not provided for Ktrans)
        row.insert(8, VoxelSpacing)  # add VoxelSpacing, provided by .mhd
        row.insert(9, Dimen)  # add Dim, provided by .mhd
        row.insert(10, 'Ktrans')  # add DCMSerDescr as 'Ktrans'
        row.insert(11, 0)  # add DCMSerNum as 0 because is not used in any dcm series
        ktrans_lst.append(row)

    try:
        with open(NEW_train_csv, 'rb')as NEW_train:
            reader_new = csv.reader(NEW_train, delimiter=',')
            column_names = reader_new.next()

            new_lst = join_rows(reader_new,ktrans_lst)

    except (IOError, NameError):
        with open(images_train_csv, 'rb')as images_train:
            with open(findings_train_csv, 'rb') as findings_train:
                new_rows = []
                reader_images = csv.reader(images_train, delimiter=',')
                reader_findings = csv.reader(findings_train, delimiter=',')
                column_names = reader_images.next()
                column_names.append('Zone')
                column_names.append('ClinSig')
                reader_findings.next()
                all_findings_rows = []
                for row in reader_findings:
                    all_findings_rows.append(row)

                for images_row in reader_images:
                    for findings_row in all_findings_rows:
                        if images_row[3] in findings_row:  # if the positions match, these rows belong to the same lesion
                            #print(images_row[0], images_row[3], findings_row[2])
                            images_row.append(findings_row[3])
                            images_row.append(findings_row[4])
                            new_rows.append(images_row)

                new_lst = join_rows(new_rows, ktrans_lst)


#for i in new_lst:
#   print
#print len(new_lst)

with open('ProstateX-Images-Train-ALL.csv', 'wb') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=column_names)
    writer.writeheader()
    for row in new_lst:
        new_row = {}
        for i in range(len(row)):
            new_row[column_names[i]] = row[i]
        writer.writerow(new_row)


#sys.stdout = orig_stdout
#f.close()