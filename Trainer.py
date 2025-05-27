from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import json
from Model import Model


class Trainer:

    # [Train]
    # include check point: save, load model parameters


    def __init__(self, config):
      self.config = config
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.loss_type = config.get("loss_type", "mae")  # 預設是 MAE
      self.criterion = self.get_loss_function(self.loss_type)
      self.lr = config.get("lr", 1e-4)
      self.epochs = config.get("num_epochs", 10)
      self.batch_size = config.get("batch_size", 4)
      self.checkpoint_path = config.get("checkpoint_path", "model_ckpt.pt")
      self.train_logs = []

      self.training_meta = {
        "loss_type": self.loss_type,
        "learning_rate": self.lr,
        "batch_size": self.batch_size,
        "num_epochs": self.epochs,
        "optimizer": "Adam" 
      }
 

    def train(self, model, train_dataset, valid_dataset):
      model = model.to(self.device)
      optimizer = optim.Adam(model.parameters(), lr=self.lr)

      train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # for batch in train_dataloader:

        #     print('do one batch')
        #     logit = model(batch)
        #     print('do one batch done')
            
        #     break
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
            loss = self.criterion(output.view(-1), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"✅ Batch {i+1} done - loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"\n📊 [Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

        if valid_dataset is not None:
          avg_val_loss = self.validate(model, valid_dataset)
        else:
          avg_val_loss = None

        # Save Log
        self.train_logs.append({
          'epoch': epoch + 1,
          'train_loss': avg_loss,
          'val_loss': avg_val_loss if valid_dataset else None
          })

        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_loss,
        }, self.checkpoint_path)

        print(f"💾 Model checkpoint saved to {self.checkpoint_path}")
    
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

        with torch.no_grad():
            for batch in valid_dataloader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)

                output = model(batch)
                target = batch["target"].view_as(output).float()
                loss = self.criterion(output.view(-1), target)

                total_loss += loss.item()

        avg_val_loss = total_loss / len(valid_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def test(self, model, test_dataset):
      model = model.to(self.device)
      model.eval()
      test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
      predictions = []
      ground_truth = []

      with torch.no_grad():
        for batch in test_dataloader:
          for key in batch:
            batch[key] = batch[key].to(self.device)

          output = model(batch)
          predictions.extend(output.view(-1).cpu().numpy())
          ground_truth.extend(batch["target"].view(-1).cpu().numpy())
      
      return predictions, ground_truth

    def get_loss_function(self, loss_type):
      if loss_type == "mae":
          return nn.L1Loss()
      elif loss_type == "mse":
          return nn.MSELoss()
      elif loss_type == "huber":
          return nn.SmoothL1Loss()
      else:
          raise ValueError(f"❌ Unsupported loss type: {loss_type}")