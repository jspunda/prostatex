"""Nothing relevant... just a script that check that if you run the csv_fix_all.py either with the NEW csv or not, 
the result is the same."""

import csv

path = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\prostatex\\master\\h5_converter\\'

with open(path + 'new_csv_new_out.txt', 'r') as file1:
    content1 = file1.read()
    with open(path + 'new_csv_tricky_out.txt', 'r') as file2:
        content2 = file2.read()

        print content1 == content2

with open(path + 'ProstateX-Images-Train-ALL1.csv', 'rb') as csv1:
    reader1 = csv.reader(csv1, delimiter=',')
    lst1 = []
    for row in reader1:
        lst1.append(row)
    with open('ProstateX-Images-Train-ALL2.csv', 'rb') as csv2:
        reader2 = csv.reader(csv2, delimiter=',')
        lst2 = []
        for row in reader2:
            lst2.append(row)

        print lst1 == lst2