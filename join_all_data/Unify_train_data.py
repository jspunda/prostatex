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

main_path = 'C:\\Users\\User\\Mis Documentos\\Mine\\Trabajo\\Uni\\RU\\2ndS-ISMI\\Project\\Data\\Training'

"""" Get all directories from the Ktrans dir and also the ones from DOI, but only up to \ProstateX-aaaa """
sub_dirs_src1 = [x[0] for x in os.walk(main_path + '\\DOI') if x[0].split(os.sep)[-1].split('-')[0] == 'ProstateX']
sub_dirs_src2 = [x[0] for x in os.walk(main_path + '\\Ktrans') if x[0].split(os.sep)[-1].split('-')[0] == 'ProstateX']


i = 0
while i < 204:
    patient_id = 'ProstateX-' + format(i, '04d')
    #print patient_id
    shutil.copytree(sub_dirs_src1[i], main_path + '\\All_training_data\\' + patient_id)
    shutil.copytree(sub_dirs_src2[i], main_path + '\\All_training_data\\' + patient_id + '\\Ktrans')
    print i
    i += 1