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


def directional_accuracy(pred, true, data_y, target):
    results = {}
    vectorized_func = np.vectorize(get_direction_value)

    currency_pairs = ['AUD', 'CAD', 'CHF', 'EUR',
                      'GBP', 'JPY', 'NZD', 'USD', 'TOTAL']
    news_mask = {}
    for i, pair in enumerate(currency_pairs):
        if pair == 'TOTAL':
            total_sum = np.sum(data_y[:, :, 0:8], axis=2)
            news_mask[pair] = total_sum > 0.0
        else:
            news_mask[pair] = data_y[:, :, i] > 0.0

    if target == 'close':
        true_pctChgs = np.expand_dims(data_y[:, :, -2], axis=2)
        np.savetxt("pctChg.csv", true_pctChgs[:, :, 0], delimiter=",")
        previous_closes = true / (1 + (true_pctChgs / 100))
        pred_pctChgs = (pred - previous_closes) * 100

        pred_value = vectorized_func(pred_pctChgs)
        true_value = vectorized_func(true_pctChgs)
    else:
        pred_value = vectorized_func(pred)
        true_value = vectorized_func(true)

    for i, pair in enumerate(currency_pairs):
        pair_pred_value = pred_value[news_mask[pair]]
        pair_true_value = true_value[news_mask[pair]]

        pair_pred_value = pair_pred_value.flatten().astype(int)
        pair_true_value = pair_true_value.flatten().astype(int)

        pair_acc = accuracy_score(pair_true_value, pair_pred_value)
        pair_prec = precision_score(pair_true_value, pair_pred_value,
                                    average="macro", zero_division=0)
        pair_rec = recall_score(pair_true_value, pair_pred_value,
                                average="macro", zero_division=0)
        pair_f1 = f1_score(pair_true_value, pair_pred_value,
                           average="macro", zero_division=0)

        results[pair] = {
            'acc': pair_acc,
            'prec': pair_prec,
            'rec': pair_rec,
            'f1': pair_f1
        }

    pred_value = pred_value.flatten().astype(int)
    true_value = true_value.flatten().astype(int)

    acc = accuracy_score(true_value, pred_value)
    prec = precision_score(true_value, pred_value,
                           average="macro", zero_division=0)
    rec = recall_score(true_value, pred_value,
                       average="macro", zero_division=0)
    f1 = f1_score(true_value, pred_value,
                  average="macro", zero_division=0)

    results['NAIVE'] = {
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1
    }

    return results


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


def metric(pred, true, data_y, target, inverse_func):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    pair_results = directional_accuracy(inverse_func(
        pred), inverse_func(true), inverse_func(data_y), target)

    return mae, mse, rmse, mape, mspe, pair_results
