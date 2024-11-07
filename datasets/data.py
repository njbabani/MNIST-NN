# -*- coding: utf-8 -*-
'''
Module for loading, normalising and flattening MNIST dataset
'''


def normalise_data(data):
    '''
    Normalise dataset (i.e. for pixels: [0, 255] to [0, 1])

    Args:
        data (np.ndarray): Dataset

    Returns:
        data_norm (np.ndarray): Normalised data
    '''

    # Find the maximum value in data
    data_max = data.max()

    # Find the minimum value in data
    data_min = data.min()

    # Normalise the data
    data_norm = (data - data_max) / (data - data_min)

    return data_norm


def flatten_data(data):
    '''
    Flatten the dataset into (28x28, m) where m is training examples

    Args:
        data (np.ndarray): Dataset

    Returns:
        data_flat (np.ndarray) = Flattened dataset
    '''

    # Obtain the number of training examples
    training_examples = data.shape[0]

    # MNIST image is square so width = height
    image_width = data.shape[1]

    # Compute total number of pixels in image
    total_pixels = image_width**2

    # Flatten the image
    data.reshape(total_pixels, training_examples)
