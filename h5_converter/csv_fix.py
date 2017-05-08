import csv

"""Script that takes ProstateX-Findings-Train.csv and adds per lesion its zone and clinsig information
to the corrects rows in ProstateX-Images-Train.csv, so all lesion information is inside ProstateX-Images-Train.csv

Only has to be run once, so hdf5 conversion later on needs to draw from just one .csv file"""

images_train_csv = 'C:\Users\Jeftha\Downloads\ProstateX-TrainingLesionInformationv2' \
                   '\ProstateX-TrainingLesionInformationv2\ProstateX-Images-Train.csv'
findings_train_csv = 'C:\Users\Jeftha\Downloads\ProstateX-TrainingLesionInformationv2' \
                     '\ProstateX-TrainingLesionInformationv2\ProstateX-Findings-Train.csv'

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
                    print(images_row[0], images_row[3], findings_row[2])
                    images_row.append(findings_row[3])
                    images_row.append(findings_row[4])
                    new_rows.append(images_row)

# Write new .csv file
with open('ProstateX-Images-Train-NEW.csv', 'wb') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=column_names)
    writer.writeheader()
    for row in new_rows:
        new_row = {}
        for i in range(len(row)):
            new_row[column_names[i]] = row[i]
        writer.writerow(new_row)
