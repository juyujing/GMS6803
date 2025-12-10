from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import torch

class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, logits, target):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self.preds.extend(probs)
        self.targets.extend(target.detach().cpu().numpy())

    def compute(self):
        preds = np.array(self.preds)
        targets = np.array(self.targets)
        pred_labels = (preds > 0.5).astype(int)

        return {
            "AUC": roc_auc_score(targets, preds),
            "F1": f1_score(targets, pred_labels, zero_division=0),
            "Precision": precision_score(targets, pred_labels, zero_division=0),
            "Recall": recall_score(targets, pred_labels, zero_division=0)
        }