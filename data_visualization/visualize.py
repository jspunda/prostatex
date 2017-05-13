import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from lesion_extraction_2d.lesion_extractor_2d import get_train_data


def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_lesions(lesions, references, lesion_info, save=False, window=(None, None)):
    """
    Visualizes lesion cutouts, their histograms and a bigger reference image. 
    If save=True, plots will instead be saved to disk to img/{TRUE, FALSE}, depending on the lesion truth label

    :param lesions: Small lesion cutout image
    :param references: Bigger reference image for the lesion cutout
    :param lesion_info: Lesion attributes (fid, zone, clinsig, etc.)
    :param save: Indicates whether plots should be shown on screen or saved to disk
    :param window: Min and max window values when plotting images, such that every pixel < min becomes min and 
    every pixel > max becomes max

    Note: For ADC images, saving all plots (334 in total) takes a few minutes.
    """
    # Flatten image for histogram plot
    lesions = np.asarray(lesions)
    lesions_flat = np.reshape(lesions, (lesions.shape[0], lesions.shape[1] * lesions.shape[2]))

    # Check for output directories
    if save:
        require_dir('img/TRUE')
        require_dir('img/FALSE')

    for i in range(lesions_flat.shape[0]):
        # Global title
        plt.suptitle('Lesion and histogram (window {} - {})'.format(window[0], window[1]), y=1)
        # Create axes on a plot grid
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        # First ax contains lesion cutout
        ax1.imshow(lesions[i], cmap='gray', vmin=window[0], vmax=window[1])

        # Extract lesion attributes to display in title
        parsed_name = lesion_info[i]['name'].split('/')
        patient_id = parsed_name[1]
        dcm_series = parsed_name[-1]
        clinsig = lesion_info[i]['ClinSig']
        zone = lesion_info[i]['Zone']
        age = lesion_info[i]['Age']
        fid = lesion_info[i]['fid']

        # Set title for lesion cutout
        ax1.set_title('Lesion cutout\n{}\n{}\nfid:{}, Zone:{}, Age:{}\nClinSig:{}'
                      .format(patient_id, dcm_series, fid, zone, age, clinsig))

        # Plot reference image for lesion
        ax2.imshow(references[i], cmap='gray', vmin=window[0], vmax=window[1])
        ax2.set_title('Reference image')

        # Create histogram for lesion cutout
        n, b, p = ax3.hist(lesions_flat[i])
        ax3.set_title('Pixel value histogram')
        ax3.set_xlabel('Pixel value')
        ax3.set_ylabel('Pixels')
        plt.sca(ax3)  # Focus just on this ax in subplot
        plt.xticks(b)  # Set values on x axis in plot to match histogram bin values
        plt.tight_layout()  # Fix layout

        if save:
            # Save image to disk and clear the figure for next loop
            print('Saving plot {} of {}'.format(i + 1, lesions_flat.shape[0]))
            plt.savefig('img/{}/{}_{}_fid_{}.png'.format(clinsig, patient_id, dcm_series, fid))
            plt.clf()
        else:
            # Just display the plot on screen. Loop is paused until user closes the plot window
            print('Showing plot {} of {}'.format(i + 1, lesions_flat.shape[0]))
            plt.show()

if __name__ == "__main__":
    """ Example usage: """
    h5_file = h5py.File('C:\\Users\\Jeftha\\stack\\Rommel\\ISMI\\prostatex-train.hdf5', 'r')
    query_words = ['ADC']
    X, y = get_train_data(h5_file, query_words, keep_lesion_data=True, size_px=6)
    X_big, y = get_train_data(h5_file, query_words, keep_lesion_data=True, size_px=40)
    visualize_lesions(X, X_big, y, save=False, window=(300, 800))
