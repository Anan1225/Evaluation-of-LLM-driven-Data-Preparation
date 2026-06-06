from __future__ import annotations

from typing import Iterable


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    t = [int(v) for v in y_true]
    p = [int(v) for v in y_pred]
    if len(t) != len(p):
        raise ValueError("y_true and y_pred lengths do not match")

    tp = tn = fp = fn = 0
    for yt, yp in zip(t, p):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1

    total = len(t)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "n": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
