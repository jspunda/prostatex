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
    :param window: Min and max window values when plotting images

    Note: For ADC images, saving all plots (334 in total) takes a few minutes.
    """
    # Flatten image for histogram plot
    # lesions = np.asarray(lesions)
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
        bins = [x * 100 for x in range(0, 22)]
        n, b, p = ax3.hist(lesions_flat[i], bins=bins)
        ax3.set_title('Pixel value histogram')
        ax3.set_xlabel('Pixel value')
        ax3.set_ylabel('Pixels')
        plt.sca(ax3)  # Focus just on this ax in subplot
        plt.xticks([0, 400, 700, 1000, 1500, 2000])  # Set values on x axis in plot to match histogram bin values
        plt.tight_layout()  # Fix layout

        if save:
            # Save image to disk and clear the figure for next loop
            print('Saving plot {} of {}'.format(i + 1, lesions_flat.shape[0]))
            plt.savefig('img/{}/{}_{}_fid_{}.pdf'.format(clinsig, patient_id, dcm_series, fid))
            plt.clf()
        else:
            # Just display the plot on screen. Loop is paused until user closes the plot window
            print('Showing plot {} of {}'.format(i + 1, lesions_flat.shape[0]))
            plt.show()


def get_pixels_in_window(np_array, window):
    pixels = np_array[(window[0] < np_array) & (np_array < window[1])]
    if len(pixels) != 0:  # If no pixels within window remain (i.e. lesion does not show up with current window)
        return pixels
    else:
        return None


def plot_size_vs_value(lesions, window, marker='o'):
    """
    Returns a plot for the actual lesion size vs. the lowest value within the window

    The actual lesion size is defined as the amount of pixels in the cutout for which window[min] < pixel < window[max].
    This helps get rid of black 'background' and other irrelevant values.
    However, for tight windows, this can leave zero pixels. These lesions are then dismissed.
    """
    # Gather relevant lesion pixels (i.e. pixels in lesion that fall within the window)
    pixels_inside = []
    for lesion in lesions:
        pixels = get_pixels_in_window(lesion, window)
        if pixels is not None:
            pixels_inside.append(pixels)

    # Gather lowest values in lesions
    mean_lesion_value = [lesion_pixels.mean() for lesion_pixels in pixels_inside]

    # Gather lesion sizes
    lesion_sizes = [len(lesion_pixels) for lesion_pixels in pixels_inside]

    plt.xlabel('Mean lesion value')
    plt.ylabel('Pixel count within window')
    return plt.scatter(mean_lesion_value, lesion_sizes, marker=marker)


def size_vs_value_comparison(lesions, labels, window):
    """
    Gets size_vs_value plots for true and false lesions separately and plots them together in one figure, 
    in order to visualize potential clusters within our data.
    """
    lesions_true = lesions[np.where(labels)[0]]  # Gather all true lesions
    lesions_false = lesions[np.where(labels == False)[0]]  # Gather all false lesions

    false = plot_size_vs_value(lesions_false, window)
    true = plot_size_vs_value(lesions_true, window, marker='x')

    plt.legend((false, true), ('False', 'True'))
    plt.title('Mean lesion value vs. pixel count in window\n(Lesion cutout size: {}x{}, window: {}-{})'.
              format(lesions[0].shape[0], lesions[0].shape[0], window[0], window[1]))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """ Example usage: """
    h5_file = h5py.File('C:\\Users\\Jeftha\\Downloads\\prostatex-train.hdf5', 'r')
    query_words = ['ADC']
    X, y_labels, attr = get_train_data(h5_file, query_words, size_px=8)
    X_big, y, attr = get_train_data(h5_file, query_words, size_px=40)

    # Lesions often show up between these two values.
    # Effects of different windows values can be checked using visualize_lesions with a window
    ADC_window = (300, 1200)

    size_vs_value_comparison(X, y_labels, ADC_window)
    visualize_lesions(X, X_big, attr, save=False, window=ADC_window)
