import loaders.seriesloader as seriesloader
import glob
import SimpleITK

DATA_DIR = "/scratch-shared/ISMI/prostatex/train-data/images/DOI"

def check_equal_voxel_size(series_paths):
    values = set()
    for directory in series_paths:
        scan_files = glob.glob(directory + '/*.dcm')
        
        # Read single .dcm file to obtain metadata
        img = SimpleITK.ReadImage(scan_files[0])
        
        values.add(img.GetSpacing())
    
    return len(values) == 1


def check_equal_voxel_size_scan_type(metadata):
    series_paths = seriesloader.find_dicom_series_paths(DATA_DIR, metadata)
    if check_equal_voxel_size(series_paths):
        print("The scans have the same voxel size")
    else:
        print("Found different voxel sizes for the scans with the following metadata:", metadata)

"""
t2_tse_tra
ep2d_diff_tra_DYNDIST
t2_tse_sag
ep2d_diff_tra_DYNDISTCALC_BVAL
ep2d_diff_tra_DYNDIST_ADC
t2_tse_cor
tfl_3d PD ref_tra_1.5x1.5_t3
"""    

check_equal_voxel_size_scan_type({'0008|103e':'ADC'})
check_equal_voxel_size_scan_type({'0008|103e':'t2_tse_tra'})
check_equal_voxel_size_scan_type({'0008|103e':'t2_tse_sag'})
check_equal_voxel_size_scan_type({'0008|103e':'BVAL'})
check_equal_voxel_size_scan_type({'0008|103e':'cor'})
check_equal_voxel_size_scan_type({'0008|103e':'tfl_3d'})
check_equal_voxel_size_scan_type({'0008|103e':'cor'})
