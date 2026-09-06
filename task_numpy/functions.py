def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """

    diagonal_length = min(len(x), len(x[0])) if len(x) else 0
    product = 1
    for i in range(diagonal_length):
        value = x[i][i]
        if value != 0:
            product *= value
    return product


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """

    if len(x) != len(y):
        return False
    remaining = list(y)
    for value in x:
        try:
            remaining.remove(value)
        except ValueError:
            return False
    return True


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """

    candidates = []
    for i in range(1, len(x)):
        if x[i - 1] == 0:
            candidates.append(x[i])
    if not candidates:
        raise ValueError("No element follows zero")
    return max(candidates)


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """

    height = len(img)
    width = len(img[0]) if height else 0
    result = []
    for i in range(height):
        row = []
        for j in range(width):
            value = 0
            for channel in range(len(coefs)):
                value += img[i][j][channel] * coefs[channel]
            row.append(value)
        result.append(row)
    return result


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """

    if len(x) == 0:
        return [], []
    elements = [x[0]]
    counters = [1]
    for value in x[1:]:
        if value == elements[-1]:
            counters[-1] += 1
        else:
            elements.append(value)
            counters.append(1)
    return elements, counters


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """

    distances = []
    for x_row in x:
        row = []
        for y_row in y:
            squared_sum = 0
            for x_value, y_value in zip(x_row, y_row):
                difference = x_value - y_value
                squared_sum += difference * difference
            row.append(squared_sum ** 0.5)
        distances.append(row)
    return distances
