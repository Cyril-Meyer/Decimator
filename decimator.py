import numpy as np


def mse(x, y):
    return np.square(np.subtract(x, y)).mean()


def decimate(sequence, metric=mse, threshold=0.20):
    """
    :param sequence: a numpy array of shape (N_IMG, X, Y, C)
    :param metric: a function F(A, B) which take numpy array and return a float
    :param threshold: a float value
    :return: the sequence with
    """
    out = []
    last_taken = sequence[0]
    out.append(last_taken)

    for i in range(1, sequence.shape[0]):
        v = metric(last_taken, sequence[i])
        if v > threshold:
            last_taken = sequence[i]
            out.append(last_taken)

    return np.array(out, dtype=sequence.dtype)
