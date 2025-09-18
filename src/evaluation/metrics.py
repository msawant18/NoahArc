"""
Evaluation metrics.
"""
import numpy as np

def iou_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if inter==0 else 0.0
    return inter/union

def precision_score(y_true, y_pred):
    y_true = y_true.astype(bool); y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    return tp / (tp + fp + 1e-9)

def recall_score(y_true, y_pred):
    y_true = y_true.astype(bool); y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()
    return tp / (tp + fn + 1e-9)

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2*p*r/(p+r+1e-9)

def dice_coef(y_true, y_pred):
    y_true = y_true.astype(bool); y_pred = y_pred.astype(bool)
    inter = (y_true & y_pred).sum()
    return (2*inter) / (y_true.sum() + y_pred.sum() + 1e-9)

def brier_score(y_true, proba):
    y_true = y_true.astype(float)
    return np.mean((proba - y_true)**2)

def ece_score(y_true, proba, n_bins=15):
    y_true = y_true.astype(int)
    proba = proba.flatten()
    y_true = y_true.flatten()
    bins = np.linspace(0,1,n_bins+1)
    binids = np.digitize(proba, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = binids==b
        if mask.sum()==0: continue
        acc = y_true[mask].mean()
        conf = proba[mask].mean()
        ece += (mask.sum()/len(proba)) * abs(acc - conf)
    return ece
