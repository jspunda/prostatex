import SimpleITK
import glob
import os


def check_scan_metadata(scan_directory, expected_metadata):
    scan_files = glob.glob(scan_directory + '/*.dcm')
    
    # Read single .dcm file to obtain metadata
    img = SimpleITK.ReadImage(scan_files[0])
    
    for (key, value) in expected_metadata.items():
        # Check whether metadata key contains the right value
        if value not in img.GetMetaData(key):
            return False
    return True
    
    
def find_dicom_series_paths(root_dir, expected_metadata):
    series_paths = []

    case_dirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
    
    print("Found", len(case_dirs), "case(s).")
    
    for case_dir in case_dirs:
        # Expecting that each case folder (e.g. "ProstateX-0000") contains
        # one folder with a 'numbers' name (e.g. "1.3.6.1.4.1.14519.5.2.1.7311.5101.158323547117540061132729905711")
        subdirs = os.listdir(case_dir)
        if len(subdirs) != 1:
            print("Unexpectedly found multiple or no folders for case", case_dir)
            print("Skipping this case for now.")
            continue

        case_dir = os.path.join(case_dir, subdirs[0])
        scan_dirs = [os.path.join(case_dir, x) for x in os.listdir(case_dir)]
        
        scan_found = False
        
        for scan_dir in scan_dirs:
            if check_scan_metadata(scan_dir, expected_metadata):
                series_paths.append(scan_dir)  
                
                if scan_found:
                    print("Found another scan with matching metadata.", scan_dir)
                scan_found = True
        
        if not scan_found:
            print ("Could not find a scan for case", case_dir)

    return series_paths
    
    
def load_dicom_series(input_dir):
    """Reads an entire DICOM series of slices from 'input_dir' and returns its pixel data as an array."""

    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    return SimpleITK.GetArrayFromImage(dicom_series)

    
def load_all_ADC_dicom_series(dir):
    metadata = {'0008|103e':'ADC'}
    return load_all_dicom_series(dir, metadata)
    
    
def load_all_dicom_series(root_dir, expected_metadata):
    series_paths = find_dicom_series_paths(root_dir, expected_metadata)
    dicom_series = [load_dicom_series(path) for path in series_paths]
    return dicom_series
