"""Keeping track of the manual changes in the .csv file"""

def manual_fix(row_lst):
    for row in row_lst:
        if row[0] == 'ProstateX-0005':
            if row[3].split(' ')[0] == '-22.0892639160156':
                row[2] = 3
            elif row[3].split(' ')[0] == '-14.5174331665039':
                row[2] = 1
            elif row[3].split(' ')[0] == '-38.6276':
                row[2] = 2

        elif row[0] == 'ProstateX-0025':
            if row[3].split(' ')[0] == '21.9659':
                row[2] = 5
            elif row[3].split(' ')[0] == '23.6983':
                row[2] = 2
            elif row[3].split(' ')[0] == '7.79027':
                row[2] = 3
            elif row[3].split(' ')[0] == '-8.32488':
                row[2] = 4

        elif row[0] == 'ProstateX-0154':
            if row[5] == '37 70 19':
                row[5] = '37 70 13'

        elif row[0] == 'ProstateX-0159':
            if row[3].split(' ')[0] == '-29.1486':
                row[2] = 3
            elif row[3].split(' ')[0] == '-7.30205':
                row[2] = 2

    return row_lst