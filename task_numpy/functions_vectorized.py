import numpy as np


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """

    diagonal = np.diagonal(np.asarray(x))
    return np.prod(diagonal[diagonal != 0])


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """

    return np.array_equal(np.sort(np.asarray(x)), np.sort(np.asarray(y)))


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """

    values = np.asarray(x)[1:][np.asarray(x)[:-1] == 0]
    if values.size == 0:
        raise ValueError("No element follows zero")
    return np.max(values)


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """

    return np.sum(np.asarray(img) * np.asarray(coefs), axis=2)


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """

    values = np.asarray(x)
    if values.size == 0:
        return np.array([], dtype=values.dtype), np.array([], dtype=int)
    starts = np.r_[True, values[1:] != values[:-1]]
    indices = np.flatnonzero(starts)
    elements = values[indices]
    counters = np.diff(np.r_[indices, values.size])
    return elements, counters


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """

    x_array = np.asarray(x)
    y_array = np.asarray(y)
    differences = x_array[:, None, :] - y_array[None, :, :]
    return np.sqrt(np.sum(differences ** 2, axis=2))
