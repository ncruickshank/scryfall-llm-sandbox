import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_recall_fscore_support


class MultiLabelEvaluator:
    """
    Provided by ChatGPT
    """
    def __init__(self, thresholds=None, k_values=[3, 5]):
        """
        thresholds: list or np.array of thresholds to sweep
        k_values: list of k for precision@k
        """
        self.thresholds = thresholds if thresholds is not None else np.linspace(0.1, 0.9, 9)
        self.k_values = k_values

    def evaluate(self, logits, labels):
        logits = np.array(logits)
        logits = np.nan_to_num(logits)
        labels = np.array(labels)

        probs = expit(logits)

        results = {}

        # =========================
        # 1. Cardinality (Ground Truth)
        # =========================
        true_cardinality = labels.sum(axis=1).mean()
        results["avg_true_labels"] = true_cardinality

        # =========================
        # 2. Threshold Sweep
        # =========================
        best_macro_f1 = -1
        best_threshold = None

        for t in self.thresholds:
            preds = (probs > t).astype(int)

            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                labels, preds, average="micro", zero_division=0
            )

            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )

            avg_pred_labels = preds.sum(axis=1).mean()

            results[f"t_{t:.2f}_micro_f1"] = f1_micro
            results[f"t_{t:.2f}_macro_f1"] = f1_macro
            results[f"t_{t:.2f}_avg_pred_labels"] = avg_pred_labels

            # track best threshold (macro F1)
            if f1_macro > best_macro_f1:
                best_macro_f1 = f1_macro
                best_threshold = t

        results["best_threshold"] = best_threshold
        results["best_macro_f1"] = best_macro_f1

        # =========================
        # 3. Precision@K
        # =========================
        for k in self.k_values:
            topk_indices = np.argsort(-probs, axis=1)[:, :k]

            correct = 0
            total = labels.shape[0] * k

            for i in range(labels.shape[0]):
                correct += labels[i, topk_indices[i]].sum()

            precision_at_k = correct / total
            results[f"precision@{k}"] = precision_at_k

        # =========================
        # 4. Top-K Only Metrics
        # =========================
        for k in self.k_values:
            preds = np.zeros_like(labels)

            topk_indices = np.argsort(-probs, axis=1)[:, :k]
            for i in range(labels.shape[0]):
                preds[i, topk_indices[i]] = 1

            _, _, f1_micro, _ = precision_recall_fscore_support(
                labels, preds, average="micro", zero_division=0
            )

            _, _, f1_macro, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )

            results[f"top{k}_micro_f1"] = f1_micro
            results[f"top{k}_macro_f1"] = f1_macro

        # =========================
        # 5. Hybrid (Threshold + Top-K Cap)
        # =========================
        for t in self.thresholds:
            for k in self.k_values:
                preds = np.zeros_like(labels)

                for i in range(labels.shape[0]):
                    idx = np.where(probs[i] > t)[0]

                    if len(idx) == 0:
                        idx = np.argsort(-probs[i])[:k]
                    elif len(idx) > k:
                        idx = np.argsort(-probs[i])[:k]

                    preds[i, idx] = 1

                _, _, f1_micro, _ = precision_recall_fscore_support(
                    labels, preds, average="micro", zero_division=0
                )

                _, _, f1_macro, _ = precision_recall_fscore_support(
                    labels, preds, average="macro", zero_division=0
                )

                results[f"hybrid_t{t:.2f}_k{k}_micro_f1"] = f1_micro
                results[f"hybrid_t{t:.2f}_k{k}_macro_f1"] = f1_macro

        return results