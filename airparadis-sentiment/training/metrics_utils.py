from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def compute_binary_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        out["roc_auc"] = float("nan")
    return out
