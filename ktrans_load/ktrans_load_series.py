import SimpleITK as sitk
import os
import glob
import array

# set the directory of the ktrans images
scan_directory = 'ProstateXKtrains-train-fixed'

# find all the mhd file paths and put them in the list
def find_ktrans_series_path(root_dir):
    series_paths = []
    case_dirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]

    for case_dir in case_dirs:
        subdirs = os.listdir(case_dir)
        case_dir = os.path.join(case_dir, subdirs[0])
    #     zraw_lst = os.path.join(case_dir, subdirs[1])
        series_paths.append(case_dir)
    return series_paths

# load all the ktrans images and transfer them to a list with numpy array
def load_ktrans_series(inputdir):
    series_patch = find_ktrans_series_path(inputdir)
    im_lst = []
    for item in series_patch:
        raw_img = sitk.ReadImage(item)
        raw_np = sitk.GetArrayFromImage(raw_img)
        im_lst.append(raw_np)
    return im_lst

ktrans_lst_np = load_ktrans_series(scan_directory)