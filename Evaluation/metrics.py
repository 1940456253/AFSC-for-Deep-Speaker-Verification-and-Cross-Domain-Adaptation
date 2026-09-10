"""Tie-aware empirical ROC, interpolated EER, normalized minimum DCF.
A score >= threshold is accepted. Equal scores share one operating point.
"""
import numpy as np


def compute_metrics(scores, labels, p_target=0.01, c_miss=1.0, c_fa=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape or not len(scores):
        raise ValueError('Expected equally sized nonempty one-dimensional scores/labels')
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Scores must be finite and labels must be 0/1')
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    if not positives or not negatives:
        raise ValueError('Evaluation requires both target and nontarget trials')
    if not 0 < p_target < 1 or min(c_miss, c_fa) <= 0:
        raise ValueError('Invalid DCF parameters')
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tpr = np.r_[0.0, np.cumsum(y)[ends] / positives]
    fpr = np.r_[0.0, np.cumsum(1 - y)[ends] / negatives]
    fnr = 1 - tpr
    delta = fnr - fpr
    crossing = int(np.flatnonzero(delta <= 0)[0])
    if delta[crossing] == 0:
        eer = fpr[crossing]
    else:
        before = crossing - 1
        weight = delta[before] / (delta[before] - delta[crossing])
        eer = fpr[before] + weight * (fpr[crossing] - fpr[before])
    cost = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    return {'eer_percent': float(100 * eer),
            'min_dcf': float(cost.min() / min(c_miss * p_target, c_fa * (1 - p_target))),
            'p_target': p_target, 'c_miss': c_miss, 'c_fa': c_fa,
            'target_trials': positives, 'nontarget_trials': negatives}
