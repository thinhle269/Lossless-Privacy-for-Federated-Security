"""Classification metrics shared by every method."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score)


def compute_metrics(y_true, y_pred, y_prob=None, task="multiclass"):
    """Return a flat dict of metrics. y_prob is P(class=1) for binary tasks."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if task == "binary" and y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass
    return out
