import SimpleITK
import glob
import os


def load_dicom_series(input_dir):
    """Reads an entire DICOM series of slices from 'input_dir' and returns its pixel data as an array."""

    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    return SimpleITK.GetArrayFromImage(dicom_series)

def check_scan_metadata(scan_directory, expected_metadata):
    scan_files = glob.glob(scan_directory + '/*.dcm')
    
    # Read single .dcm file to obtain metadata
    img = SimpleITK.ReadImage(scan_files[0])
    
    for (key, value) in expected_metadata.items():
        # Check whether metadata key contains the right value
        if value not in img.GetMetaData(key):
            return False
    return True
    
def find_ADC_dicom_series(dir):
    metadata = {'0008|103e':'ADC'}
    return find_dicom_series(dir, metadata)
    
def find_dicom_series(root_dir, expected_metadata):
    """Returns the DICOM series for scans that have the expected metadata
    for all the cases in 'root_dir' (i.e. the 'DOI' folder). 
    Since DICOM metadata is lost when reading all slices at once, 
    we first check a single .dcm slice to determine whether
    this series is relevant. If so, load entire directory, if not, move to next directory"""

    dicom_series = []

    case_dirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
    
    print("Found", len(case_dirs), "case(s).")
    
    for case_dir in case_dirs:
        # Expecting that each case folder (e.g. "ProstateX-0000") contains
        # one folder with a 'numbers' name (e.g. "1.3.6.1.4.1.14519.5.2.1.7311.5101.158323547117540061132729905711")
        subdirs = os.listdir(case_dir)
        if len(subdirs) != 1:
            print("Unexpectedly found multiple folders for case", case_dir)
            print("Skipping this case for now.")
            continue

        case_dir = os.path.join(case_dir, subdirs[0])
        scan_dirs = [os.path.join(case_dir, x) for x in os.listdir(case_dir)]
        
        scan_found = False
        
        for scan_dir in scan_dirs:
            if check_scan_metadata(scan_dir, expected_metadata):
                # Loads all slices at once
                dicom_series.append(load_dicom_series(scan_dir))  
                
                if scan_found:
                    print("Found another scan with matching metadata.", scan_dir)
                scan_found = True
        
        if not scan_found:
            print ("Could not find a scan for case", case_dir)

    return dicom_series

# Example usage
# rootdir = 'C:/Users/Jeftha/Downloads/DOI/'
# key = '0008|103e'
# value = 'ADC'
#
# find_dicom_series(rootdir, key, value)
