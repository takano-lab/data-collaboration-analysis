from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def _true_label_confidence(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classes: Optional[Iterable] = None,
) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score row sizes do not match.")

    if classes is None:
        # Assume encoded labels [0..C-1].
        idx = y_true.astype(int, copy=False)
        out = np.full((y_true.shape[0],), np.nan, dtype=float)
        valid = (idx >= 0) & (idx < y_score.shape[1])
        out[valid] = y_score[np.where(valid)[0], idx[valid]]
        return out

    cls_arr = np.asarray(list(classes))
    pos = {v: i for i, v in enumerate(cls_arr)}
    out = np.full((y_true.shape[0],), np.nan, dtype=float)
    for i, y in enumerate(y_true):
        j = pos.get(y, None)
        if j is None:
            continue
        out[i] = float(y_score[i, j])
    return out


def compute_membership_auc(
    *,
    y_member: np.ndarray,
    score_member: np.ndarray,
    y_nonmember: np.ndarray,
    score_nonmember: np.ndarray,
    classes: Optional[Iterable] = None,
) -> float:
    """
    Compute black-box membership inference AUC using true-label confidence.

    Parameters
    ----------
    y_member, y_nonmember:
        True labels for member/non-member sets.
    score_member, score_nonmember:
        Predicted class probabilities (shape: n x C) for each set.
    classes:
        Class order corresponding to probability columns.
        If None, labels are treated as encoded indices.
    """
    conf_m = _true_label_confidence(y_member, score_member, classes=classes)
    conf_n = _true_label_confidence(y_nonmember, score_nonmember, classes=classes)

    y_attack = np.concatenate(
        [np.ones(conf_m.shape[0], dtype=int), np.zeros(conf_n.shape[0], dtype=int)]
    )
    s_attack = np.concatenate([conf_m, conf_n]).astype(float)

    valid = np.isfinite(s_attack)
    if int(np.sum(valid)) < 2:
        return float("nan")
    yv = y_attack[valid]
    sv = s_attack[valid]
    if len(np.unique(yv)) < 2:
        return float("nan")

    try:
        return float(roc_auc_score(yv, sv))
    except ValueError:
        return float("nan")

