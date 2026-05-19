"""Metrics for evaluation."""

import numpy as np


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0


def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm