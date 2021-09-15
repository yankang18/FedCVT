import time

import numpy as np
import torch


def convert_to_1d_labels(y_prob):
    y_1d = np.argmax(y_prob, axis=1)
    return y_1d


def f_score(precision, recall):
    return 2 / (1 / precision + 1 / recall)


def transpose(x):
    return x if x is None else torch.transpose(x, 0, 1)


def get_timestamp():
    local_time = time.localtime(time.time())
    timestamp = time.strftime("%Y%m%d%H%M%S", local_time)
    return timestamp
