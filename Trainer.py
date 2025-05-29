from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import json
from Model import Model
import numpy as np


class Trainer:

    # [Train]
    # include check point: save, load model parameters


    def __init__(self, config):
      self.config = config
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.loss_type = config.get("loss_type", "mae")  # 預設是 MAE
      self.huber_beta = config.get("huber_beta", 10.0)
      self.criterion = self.get_loss_function(self.loss_type)
      self.lr = config.get("lr", 1e-4)
      self.epochs = config.get("individual_epochs", 10)
      self.batch_size = config.get("batch_size", 4)
      self.weight_decay = config.get("weight_decay", 0.0)
      self.checkpoint_path = config.get("checkpoint_path", "model_ckpt.pt")
      self.best_val_loss = float("inf")
      self.patience = config.get("patience", 5)  # Early stopping patience
      self.train_logs = []

      self.training_meta = {
        "loss_type": self.loss_type,
        "quantile_q": config.get("quantile_q", 0.9) if self.loss_type == "quantile" else None,
        "learning_rate": self.lr,
        "batch_size": self.batch_size,
        "num_epochs": self.epochs,
        "optimizer": "Adam" 
        
      }
 

    def train(self, model, train_dataset, valid_dataset):
      model = model.to(self.device)
      optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
      # Learning rate scheduler
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(
          optimizer, mode='min', factor=0.5, patience=3, verbose=True
      )
      patience_counter = 0

      train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

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
            
            target = batch["target"].float().view(-1)
            output_raw = torch.expm1(output)
            target_raw = torch.expm1(target)
            loss = self.criterion(output_raw.view(-1), target_raw.view(-1))
            # loss = self.criterion(output.view(-1), target)

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
                      if 'mm_net' in name:  # Focus on your transformer
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
        self.train_logs.append({
          'epoch': epoch + 1,
          'train_loss': avg_loss,
          'val_loss': avg_val_loss if valid_dataset else None
          })

        if valid_dataset and avg_val_loss < self.best_val_loss:
          print(f"🎯 Best model updated at epoch {epoch+1} (Val Loss: {avg_val_loss:.4f})")
          self.best_val_loss = avg_val_loss
    
          # 儲存最佳模型
          torch.save({
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'epoch': epoch,
              'loss': avg_val_loss,
          }, self.checkpoint_path)
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
        json.dump({
            "meta": self.training_meta,
            "logs": self.train_logs
        }, f, indent=2)




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

                output = model(batch)
                target = batch["target"].float().view(-1)
              
                output_raw = torch.expm1(output)
                target_raw = torch.expm1(target)

                predictions.extend(output_raw.view(-1).cpu().numpy())
                ground_truth.extend(batch["target"].view(-1).cpu().numpy())

                loss = self.criterion(output_raw.view(-1), target_raw)

                # predictions.extend(output.view(-1).cpu().numpy())
                # loss = self.criterion(output.view(-1), target)

                total_loss += loss.item()

        avg_val_loss = total_loss / len(valid_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        np.save("result/val_predictions.npy", predictions)
        np.save("result/val_ground_truth.npy", ground_truth)
        return avg_val_loss

    def test(self, model, test_dataset):
      model = model.to(self.device)
      model.eval()
      test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
      predictions = []
      ground_truth = []

      with torch.no_grad():
        for batch in test_dataloader:
          for key in batch:
            batch[key] = batch[key].to(self.device)

          output = model(batch)
          output_raw = torch.expm1(output)
          predictions.extend(output_raw.view(-1).cpu().numpy())
          # predictions.extend(output.view(-1).cpu().numpy())
          ground_truth.extend(batch["target"].view(-1).cpu().numpy())
      
      return predictions, ground_truth

    @staticmethod
    def quantile_loss(preds, target, q):
        diff = target - preds
        return torch.max((q - 1) * diff, q * diff).mean()
    
    @staticmethod
    def asymmetric_loss(preds, target, alpha=1.5):
        diff = preds - target
        return torch.where(diff < 0, alpha * diff.abs(), diff.abs()).mean()


    def get_loss_function(self, loss_type):
      if loss_type == "mae":
          return nn.L1Loss()
      elif loss_type == "mse":
          return nn.MSELoss()
      elif loss_type == "huber":
          return nn.SmoothL1Loss(self.huber_beta)
      elif loss_type == "quantile":
          q = self.config.get("quantile_q", 0.9)
          return lambda preds, target: self.quantile_loss(preds, target, q)
      elif loss_type == "asymmetric":
        alpha = self.config.get("asymmetric_alpha", 1.5)
        return lambda preds, target: self.asymmetric_loss(preds, target, alpha)
      else:
          raise ValueError(f"❌ Unsupported loss type: {loss_type}")