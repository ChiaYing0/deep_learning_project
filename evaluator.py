import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import seaborn as sns

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

    def plot_prediction_vs_truth(self):
        plt.scatter(self.ground_truth, self.predictions, alpha=0.5)
        plt.plot([self.ground_truth.min(), self.ground_truth.max()],
                 [self.ground_truth.min(), self.ground_truth.max()],
                 color='red', linestyle='--')
        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        plt.title("Prediction vs. Ground Truth")
        plt.show()

    def plot_error_distribution(self):
        errors = self.predictions - self.ground_truth
        plt.hist(errors, bins=50)
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title("Distribution of Prediction Errors")
        plt.show()

    def evaluate_metrics(self):
        mae = mean_absolute_error(self.ground_truth, self.predictions)
        mse = mean_squared_error(self.ground_truth, self.predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.ground_truth, self.predictions)

        print(f"MAE : {mae:.4f}")
        print(f"MSE : {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²  : {r2:.4f}")
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
    
    def check_prediction_std(self):
        std = np.std(self.predictions)
        uniq = len(np.unique(np.round(self.predictions, 2)))
        print(f"📉 Std: {std:.4f} | Unique values (2 decimal): {uniq}")
        if std < 0.1 or uniq < 5:
            print("⚠️ Prediction Collapse Detected.")
        else:
            print("✅ Prediction distribution looks ok.")
    
    def plot_prediction_truth_density(self):
        sns.kdeplot(self.ground_truth, label='Ground Truth', fill=True, linewidth=2)
        sns.kdeplot(self.predictions, label='Prediction', fill=True, linewidth=2)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Prediction vs. Ground Truth Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def residual_plot(self):
        residuals = self.predictions - self.ground_truth
        plt.scatter(self.ground_truth, residuals, alpha=0.5)
        plt.xlabel("Ground Truth")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.axhline(0, color='red', linestyle='--')
        plt.show()