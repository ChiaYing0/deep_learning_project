import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

class ModelEvaluator:
    def __init__(self, predictions, ground_truth, train_log_path=None):
        self.predictions = np.array(predictions)
        self.ground_truth = np.array(ground_truth)
        self.train_log_path = train_log_path

    def plot_loss_curve(self):
        if not self.train_log_path:
            print("❌ train_log_path not provided.")
            return

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

    def plot_binned_error(self, bins=None):
        df = pd.DataFrame({"truth": self.ground_truth, "pred": self.predictions})
        if bins is None:
            bins = np.linspace(self.ground_truth.min(), self.ground_truth.max(), 6)
        df["bin"] = pd.cut(df["truth"], bins=bins)
        bin_mae = df.groupby("bin").apply(lambda x: np.mean(np.abs(x["truth"] - x["pred"])))

        bin_mae.plot(kind="bar")
        plt.ylabel("Mean Absolute Error")
        plt.title("Error by Ground Truth Bins")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
