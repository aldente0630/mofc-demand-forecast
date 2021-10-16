import pickle
import numpy as np
from numba import jit


def dump_pickle(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


@jit(nopython=True)
def remove_outlier(series, window_size, fill_na=True, n_sigmas=3):
    n = len(series)
    copied = series.copy()
    indices = []
    k = 1.4826

    for i in range(window_size, (n - window_size)):
        median = np.nanmedian(series[(i - window_size) : (i + window_size)])
        stat = k * np.nanmedian(
            np.abs(series[(i - window_size) : (i + window_size)] - median)
        )
        if np.abs(series[i] - median) > n_sigmas * stat:
            copied[i] = median if fill_na else np.nan
            indices.append(i)

    return copied, indices
