from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import json
from Model import Model
import numpy as np
from Dataset import Dataset
from sklearn.utils.class_weight import compute_class_weight



class Trainer:

    # [Train]
    # include check point: save, load model parameters

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_type = config.get("loss_type", "CE")  # 預設是 CE
        self.huber_beta = config.get("huber_beta", 10.0)
        self.criterion = self.get_loss_function(self.loss_type)
        self.lr = config.get("lr", 1e-4)
        self.epochs = config.get("regular_epochs", 10)
        self.batch_size = config.get("batch_size", 4)
        self.weight_decay = config.get("weight_decay", 0.0)
        self.checkpoint_path = config.get("checkpoint_path", "model_ckpt.pt")
        self.best_val_loss = float("inf")
        self.patience = config.get("patience", 5)  # Early stopping patience
        self.freeze_img_net = config.get("freeze_img_net", True)
        self.freeze_img_layers = config.get("freeze_img_layers", [])
        self.train_logs = []

        self.training_meta = config

    def train(self, model, train_dataset, valid_dataset):
        model = model.to(self.device)
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        patience_counter = 0

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0

            print(f"Using loss: {self.loss_type}")
            print(f"\n▶️ Epoch {epoch+1}/{self.epochs}")

            for i, batch in enumerate(train_dataloader):
                print(f"\n🔁 Batch {i+1}/{len(train_dataloader)} - start")

                for key in batch:
                    batch[key] = batch[key].to(self.device)

                output = model(batch)

                target = batch["target"].long()  # 確保是 LongTensor
                loss = self.criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 梯度監控
                if i == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            print(f"{name}: grad_norm = {param.grad.norm().item():.6f}")
                        else:
                            print(f"{name}: NO GRADIENT")

                if epoch == 0 and i == 0:
                    print("=== GRADIENT CHECK (First batch only) ===")
                    total_grad_norm = 0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            total_grad_norm += grad_norm
                            if "mm_net" in name:  # Focus on your transformer
                                print(f"{name}: {grad_norm:.6f}")
                    print(f"Total gradient norm: {total_grad_norm:.6f}")

                    # Check output values
                    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
                    print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")

                optimizer.step()
                total_loss += loss.item()

                print(f"✅ Batch {i+1} done - loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_dataloader)

            print(f"\n📊 [Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

            if valid_dataset is not None:
                avg_val_loss = self.validate(model, valid_dataset)
            else:
                avg_val_loss = None

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Save Log
            self.train_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_loss": avg_val_loss if valid_dataset else None,
                }
            )

            if valid_dataset and avg_val_loss < self.best_val_loss:
                print(
                    f"🎯 Best model updated at epoch {epoch+1} (Val Loss: {avg_val_loss:.4f})"
                )
                self.best_val_loss = avg_val_loss

                # 儲存最佳模型
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "loss": avg_val_loss,
                    },
                    self.checkpoint_path,
                )
                print(f"💾 Model checkpoint saved to {self.checkpoint_path}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Save Log Json
        filepath = f"./train_logs/train_logs_{self.loss_type}_lr{self.lr}.json"
        with open(filepath, "w") as f:
            json.dump(
                {"meta": self.training_meta, "logs": self.train_logs}, f, indent=2
            )

    def validate(self, model, valid_dataset):
        model.eval()
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size)
        total_loss = 0.0
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for batch in valid_dataloader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)

                output = model(batch)  # shape: (B, num_classes)
                target = batch["target"].long()  # 確保是 LongTensor

                pred = output.argmax(dim=1)  # 預測類別

                predictions.extend(pred.cpu().numpy())
                ground_truth.extend(target.cpu().numpy())

                loss = self.criterion(output, target)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(valid_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        np.save("result/val_predictions.npy", predictions)
        np.save("result/val_ground_truth.npy", ground_truth)

        return avg_val_loss

    def test(self, model, test_dataset):
        model = model.to(self.device)
        model.eval()
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for batch in test_dataloader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)

                output = model(batch)  # (B, num_classes)
                pred = output.argmax(dim=1)  # (B,)

                predictions.extend(pred.cpu().numpy())
                ground_truth.extend(batch["target"].long().cpu().numpy())

        return predictions, ground_truth

    def compute_class_weights(self, dataset):
        labels = dataset.df["target"].values
        classes = np.unique(labels)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
        print(f"📊 Computed class weights: {weights}")
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        return weight_tensor
    
    def get_loss_function(self, loss_type):
        if loss_type == "CE":
            train_set = Dataset(self.config, mode="train")
            weight_tensor = self.compute_class_weights(train_set)
            return nn.CrossEntropyLoss(weight=weight_tensor)
            # return nn.CrossEntropyLoss()
        # elif loss_type == "mse":
        #     return nn.MSELoss()
        else:
            raise ValueError(f"❌ Unsupported loss type: {loss_type}")
