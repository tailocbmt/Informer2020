import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)


def get_direction_value(value: float) -> int:
    if value > 0.0:
        return 1
    elif value < 0.0:
        return -1
    return 0


def directional_accuracy(pred, true, data_y):
    if np.all(true) > 1.0:
        previous_closes = true + np.multiply(data_y[:, :, -2], true)
        pred_pctChgs = pred - previous_closes

    vectorized_func = np.vectorize(get_direction_value)

    pred_value = vectorized_func(pred_pctChgs)
    true_value = vectorized_func(pred_pctChgs)

    acc = accuracy_score(true_value, pred_value)
    prec = precision_score(true_value, pred_value,
                           average="macro", zero_division=0)
    rec = recall_score(true_value, pred_value,
                       average="macro", zero_division=0)
    f1 = f1_score(true_value, pred_value,
                  average="macro", zero_division=0)

    return acc, prec, rec, f1


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))


def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0)
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred-true))


def MSE(pred, true):
    return np.mean((pred-true)**2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true, data_y):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    acc, prec, rec, f1 = directional_accuracy(pred, true, data_y)

    return mae, mse, rmse, mape, mspe, acc, prec, rec, f1
