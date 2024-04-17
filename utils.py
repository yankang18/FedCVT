import random
import time

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve


def convert_to_1d_labels(y_prob):
    y_1d = np.argmax(y_prob, axis=1)
    return y_1d


def convert_to_1d_prob(y_prob):
    # y_1d = np.argmax(y_prob, axis=1)
    # print("convert_to_1d_prob, y_prob:", y_prob)
    y_prob_1d = np.max(y_prob, axis=1)
    # print("y_1d:", y_prob_1d)
    return y_prob_1d


def f_score(precision, recall):
    return 2 / (1 / precision + 1 / recall)


def f_score_v2(y_gt, y_pred):
    # print("f_score_v2, y_gt:", y_gt)
    # print("f_score_v2, y_pred:", y_pred)
    precision, recall, thresholds = precision_recall_curve(y_gt, y_pred)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    score = np.max(f1_scores)
    return score


def transpose(x):
    return x if x is None else torch.transpose(x, 0, 1)


def get_timestamp():
    local_time = time.localtime(time.time())
    timestamp = time.strftime("%Y%m%d%H%M%S", local_time)
    return timestamp


def set_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
