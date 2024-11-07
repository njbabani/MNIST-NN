# -*- coding: utf-8 -*-
'''
Module for loading, normalising and flattening MNIST dataset
'''


def normalise_data(data):
    '''
    Normalise dataset (i.e. for pixels: [0, 255] to [0, 1])

    Args:
        data (np.ndarray): Data to be normalised

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
