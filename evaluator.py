import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report
) 


class ModelEvaluator:
    def __init__(self, predictions, ground_truth, train_log_path=None):
        self.predictions = np.array(predictions)
        self.ground_truth = np.array(ground_truth)

        self.train_log_path = train_log_path

    def plot_loss_curve(self):
        if not self.train_log_path:
            print("❌ train_log_path not provided.")
            return
        
        elif "kfold" in self.train_log_path:
            with open(self.train_log_path) as f:
                results = json.load(f)

            for fold_info in results["fold_details"]:
                history = fold_info["history"]
                epochs = [h["epoch"] for h in history]
                train_losses = [h["train_loss"] for h in history]
                val_losses = [h["val_loss"] for h in history]
                plt.plot(epochs, train_losses, label=f"Fold {fold_info['fold']} Train")
                plt.plot(epochs, val_losses, linestyle='--', label=f"Fold {fold_info['fold']} Val")


        else:
            with open(self.train_log_path) as f:
                logs = json.load(f)["logs"]

            epochs = [log["epoch"] for log in logs]
            train_loss = [log["train_loss"] for log in logs]
            val_loss = [log["val_loss"] for log in logs]
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs. Validation Loss")
        plt.legend()
        plt.show()   


    def evaluate_all_metrics(self):
        # 計算 Accuracy、Kappa、QWK
        acc = accuracy_score(self.ground_truth, self.predictions)
        kappa = cohen_kappa_score(self.ground_truth, self.predictions)
        qwk = quadratic_weighted_kappa(self.ground_truth, self.predictions)

        print("=== 📊 Overall Metrics ===")
        print(f"Accuracy       : {acc:.4f}")
        print(f"Cohen's Kappa  : {kappa:.4f}")
        print(f"QWK (Ordinal)  : {qwk:.4f}")

        # 建立表格：Precision / Recall / F1 for macro, weighted, micro
        rows = []
        for avg in ["macro", "weighted", "micro"]:
            precision = precision_score(self.ground_truth, self.predictions, average=avg, zero_division=0)
            recall = recall_score(self.ground_truth, self.predictions, average=avg, zero_division=0)
            f1 = f1_score(self.ground_truth, self.predictions, average=avg, zero_division=0)

            rows.append({
                "Average Type": avg.capitalize(),
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

        df_metrics = pd.DataFrame(rows)
        print("\n=== 📋 Detailed Scores by Average Type ===")
        print(df_metrics.to_string(index=False))

        print("\n=== 📝 Classification Report ===")
        print(classification_report(self.ground_truth, self.predictions, zero_division=0))

        # 回傳所有資料
        return {
            "accuracy": acc,
            "cohen_kappa": kappa,
            "qwk": qwk,
            "table": df_metrics
        }




    
    def plot_confusion_matrix(self, normalize=False):
        cm = confusion_matrix(self.ground_truth, self.predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
        else:
            fmt = "d"

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", square=True, 
                    xticklabels=range(cm.shape[0]), yticklabels=range(cm.shape[0]))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        plt.tight_layout()
        plt.show()

    def plot_per_class_recall(self):
        """畫出每個類別的 Recall 條形圖"""
        from sklearn.metrics import recall_score

        labels = np.unique(self.ground_truth)
        recalls = recall_score(self.ground_truth, self.predictions, labels=labels, average=None, zero_division=0)

        plt.figure(figsize=(6, 4))
        sns.barplot(x=labels, y=recalls)
        plt.ylim(0, 1)
        plt.xlabel("Class")
        plt.ylabel("Recall")
        plt.title("Per-Class Recall")
        plt.tight_layout()
        plt.show()

    def plot_misclassification_heatmap(self):
        """視覺化每個類別的誤判去向（不含正確的主對角線）"""
        cm = confusion_matrix(self.ground_truth, self.predictions)
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)  # 去掉主對角線（正確預測）

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_no_diag, annot=True, fmt="d", cmap="Reds", square=True,
                    xticklabels=range(cm.shape[0]), yticklabels=range(cm.shape[0]))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Misclassification Heatmap (Errors Only)")
        plt.tight_layout()
        plt.show()


def quadratic_weighted_kappa(y_true, y_pred, min_rating=None, max_rating=None):
        assert len(y_true) == len(y_pred)
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        
        if min_rating is None:
            min_rating = min(np.min(y_true), np.min(y_pred))
            print(f"Minimum rating set to: {min_rating}")
        if max_rating is None:
            max_rating = max(np.max(y_true), np.max(y_pred))
            print(f"Maximum rating set to: {max_rating}")
        
        num_ratings = max_rating - min_rating + 1
        conf_mat = confusion_matrix(y_true, y_pred, labels=range(min_rating, max_rating + 1))
        print(f"Confusion Matrix:\n{conf_mat}")
        
        # Get marginal distributions
        hist_true = np.bincount(y_true - min_rating, minlength=num_ratings)
        hist_pred = np.bincount(y_pred - min_rating, minlength=num_ratings)
        
        # Create weight matrix
        weights = np.zeros((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                weights[i][j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
        
        # Calculate expected and observed agreement matrices
        N = len(y_true)  # Total number of samples
        expected = np.outer(hist_true, hist_pred) / N  # Expected under independence
        observed = conf_mat  # Observed counts
        
        # Calculate weighted agreements
        numerator = np.sum(weights * observed)
        denominator = np.sum(weights * expected)
        
        return 1.0 - (numerator / denominator)