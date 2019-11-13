import numpy as np


def max_min_scaler(data):
    max_value = np.max(data)
    min_value = np.min(data)
    denominator = max_value - min_value
    return(data - min_value)/denominator
