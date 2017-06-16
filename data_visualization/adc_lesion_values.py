import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
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

def apply_window(np_array, window):
    np_array[np_array < window[0]] = window[0]
    np_array[np_array > window[1]] = window[1]
    return np_array


def get_pixels_in_window(np_array, window):
    pixels = np_array[(window[0] < np_array) & (np_array < window[1])]
    if len(pixels) != 0:  # If no pixels within window remain (i.e. lesion does not show up with current window)
        return pixels
    else:
        return None


def size_vs_value(lesions, window):
    """
    Returns lists for the actual lesion size vs. the mean value within the window

    The actual lesion size is defined as the amount of pixels in the cutout for which window[min]<pixel<window[max].
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

    return mean_lesion_value, lesion_sizes


def size_vs_value_scatter(lesions, labels, window):
    """
    Gets size_vs_value lists for true and false lesions separately and plots them together in one figure,
    in order to visualize potential clusters within our data.
    """
    lesions_true = lesions[np.where(labels)[0]]  # Gather all true lesions
    lesions_false = lesions[np.where(labels == False)[0]]  # Gather all false lesions

    false_features = size_vs_value(lesions_false, window)
    true_features = size_vs_value(lesions_true, window)

    false_plot = plt.scatter(false_features[0], false_features[1])
    true_plot = plt.scatter(true_features[0], true_features[1], marker='x')

    plt.xlabel('Mean lesion value')
    plt.ylabel('Pixel count within window')
    plt.legend((false_plot, true_plot), ('False', 'True'))
    plt.title('Mean lesion value vs. pixel count in window\n(Lesion cutout size: {}x{}, window: {}-{})\n'
              'Silhouette score: {}'
              .format(lesions[0].shape[0], lesions[0].shape[0], window[0], window[1],
                      size_vs_value_score(lesions, labels, window)))
    plt.tight_layout()
    plt.show()


def size_vs_value_score(lesions, labels, window):
    """"Computes silhouette score for a given clustering"""
    lesions_true = lesions[np.where(labels)[0]]  # Gather all true lesions
    lesions_false = lesions[np.where(labels == False)[0]]  # Gather all false lesions

    false_features = size_vs_value(lesions_false, window)
    true_features = size_vs_value(lesions_true, window)

    false_combined = list(zip(false_features[0], false_features[1]))
    false_labels = [0 for item in false_combined]

    true_combined = list(zip(true_features[0], true_features[1]))
    true_labels = [1 for item in true_combined]

    # Enforce that we have enough data points
    if len(false_labels) >= int(0.75 * len(lesions_false)) and len(true_labels) >= int(0.75 * len(lesions_true)):
        return metrics.silhouette_score(false_combined + true_combined, false_labels + true_labels)
    else:
        return -1


def find_best_window(lesions, labels):
    all_windows = [(start, end) for start in range(100, 4000, 100) for end in range(start+100, 4000, 100)]
    scores = [(size_vs_value_score(lesions, labels, window), window) for window in all_windows]
    best_window = max(scores)
    return best_window


if __name__ == "__main__":
    """ Example usage: """
    h5_file = h5py.File('C:\\Users\\Jeftha\\stack\\Rommel\\ISMI\\data\\prostatex-train.hdf5', 'r')
    query_words = ['ADC']
    X_big, y, attr = get_train_data(h5_file, query_words, size_px=40)

    X, y_labels, attr = get_train_data(h5_file, query_words, size_px=16)
    size_vs_value_scatter(X, y_labels, find_best_window(X, y_labels)[1])

    # zones = ['AS', 'PZ', 'TZ']
    # for i in range(4, 18, 2):
    #     X, y_labels, attr = get_train_data(h5_file, query_words, size_px=i)
    #     # for zone in zones:
    #     #     zoneX = np.asarray([X[np.where(attr == el)][0] for el in attr if el['Zone'] == zone])
    #     #     zoneY = np.asarray([y_labels[np.where(attr == el)][0] for el in attr if el['Zone'] == zone])
    #     #     # zoneA = np.asarray([el for el in attr if el['Zone'] == zone])
    #     #     size_vs_value_scatter(zoneX, zoneY, find_best_window(zoneX, zoneY)[1])
    #     size_vs_value_scatter(X, y_labels, find_best_window(X, y_labels)[1])

    # Lesions often show up between these two values.
    # Effects of different windows values can be checked using visualize_lesions with a window
    # ADC_window = (0, 100)
    # print size_vs_value_score(X, y_labels, ADC_window)
    # size_vs_value_scatter(X, y_labels, ADC_window)

    # visualize_lesions(pzX, pzX, pzA, save=False, window=ADC_window)
