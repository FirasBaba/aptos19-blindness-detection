from __future__ import division
from sklearn.metrics import cohen_kappa_score
import numpy as np
//lol


def kappa_metric(target, preds,coef = [0.5, 1.5, 2.5, 3.5]):
    test_preds = np.zeros(len(preds))
    for i, pred in enumerate(preds):
        if pred < coef[0]:
            test_preds[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            test_preds[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            test_preds[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            test_preds[i] = 3
        else:
            test_preds[i] = 4
    return cohen_kappa_score(target, test_preds, weights='quadratic')
    
