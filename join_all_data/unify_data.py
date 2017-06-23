import shutil
import os

""" Script that, from a directory with two folders \\DOI and \\Ktrans, creates a combined folder All_training_data
    with the following structure:
    
    \\All_training_data
            \\ProstateX-0000
                    \\[Contents of DOI\\ProstateX-0000]
                    \\[Contents of Ktrans\\ProstateX-0000]
        
            .
            .
            .
            .
            \\ProstateX-0203
                    \\[Contents of DOI\\ProstateX-0203]
                    \\[Contents of Ktrans\\ProstateX-0203]"""

""" This is done because it makes life easier to convert all to HDF5"""

train_set = False #Denotes whether we are joining the training or the test data

main_path = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Test'

"""" Get all directories from the Ktrans dir and also the ones from DOI, but only up to \ProstateX-aaaa """
sub_dirs_src1 = [x[0] for x in os.walk(main_path + '\\DOI') if x[0].split(os.sep)[-1].split('-')[0] == 'ProstateX']
sub_dirs_src2 = [x[0] for x in os.walk(main_path + '\\Ktrans') if x[0].split(os.sep)[-1].split('-')[0] == 'ProstateX']

#print len(sub_dirs_src1), len(sub_dirs_src2)

# Turns out that there is no Ktrans for the test cases 325 and 331 but for practical issues,
# it's easier to just create an empty directory in those cases:
aux_lst2 = [x.split(os.sep)[-1] for x in sub_dirs_src2]
aux_lst1 = [x.split(os.sep)[-1] for x in sub_dirs_src1 if x.split(os.sep)[-1] not in aux_lst2]

if aux_lst1:
    for case in aux_lst1:
        doi_dir = main_path + '\\DOI\\' + case
        ktrans_dir = main_path + '\\Ktrans\\' + case
        os.makedirs(ktrans_dir)
        sub_dirs_src2.insert(sub_dirs_src1.index(doi_dir), ktrans_dir)

print len(sub_dirs_src1), len(sub_dirs_src2)

#for i in range(len(sub_dirs_src1)):
#    print sub_dirs_src1[i].split(os.sep)[-1] == sub_dirs_src2[i].split(os.sep)[-1]

num_cases = len(sub_dirs_src1)
case_0 = int(sub_dirs_src1[0].split(os.sep)[-1].split('-')[-1])
#print case_0

i = 0
while i < num_cases:
    patient_id = 'ProstateX-' + format(i + case_0, '04d')
    print patient_id
    if train_set:
        shutil.copytree(sub_dirs_src1[i], main_path + '\\All_training_data\\' + patient_id)
        shutil.copytree(sub_dirs_src2[i], main_path + '\\All_training_data\\' + patient_id + '\\Ktrans')
    else:
        shutil.copytree(sub_dirs_src1[i], main_path + '\\All_test_data\\' + patient_id)
        shutil.copytree(sub_dirs_src2[i], main_path + '\\All_test_data\\' + patient_id + '\\Ktrans')
    #print i
    i += 1