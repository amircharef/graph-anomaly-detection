from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate(y_true, scores):
    return roc_auc_score(y_true, scores), average_precision_score(y_true, scores)