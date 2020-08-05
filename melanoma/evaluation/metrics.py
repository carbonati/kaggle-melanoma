import torch
from sklearn.metrics import roc_auc_score


def compute_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return None


def fpr_score(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    fp = sum((y_pred == 1) & (y_true == 0))
    tn = sum((y_pred == 0) & (y_true == 0))
    den = fp + tn
    if den == 0:
        return 0
    else:
        return fp / den


def specificity_score(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    fp = sum((y_pred == 1) & (y_true == 0))
    tn = sum((y_pred == 0) & (y_true == 0))
    den = fp + tn
    if den == 0:
        return 0
    else:
        return tn / den

def pos_rate(y_true, y_pred):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    return y_pred.ravel().sum() / len(y_pred.ravel())
