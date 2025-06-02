import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import json
import os
from copy import deepcopy

class KFoldTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_type = config.get("loss_type", "mae")
        self.huber_beta = config.get("huber_beta", 10.0)
        self.criterion = self.get_loss_function(self.loss_type)
        self.lr = config.get("lr", 1e-4)
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 4)
        self.weight_decay = config.get("weight_decay", 1e-3)
        self.n_splits = config.get("n_splits", 5)
        self.patience = config.get("patience", 5)
        
        # Create directory for saving models
        self.model_save_dir = "kfold_models"
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Results storage
        self.fold_results = []
        self.best_models = []
        self.model_paths = []  # Store paths to saved models
        self.fold_losses = []  # Store fold losses for ensemble
        self.weights = []  # Store weights for ensemble predictions
  
    @staticmethod
    def quantile_loss(preds, target, q):
        diff = target - preds
        return torch.max((q - 1) * diff, q * diff).mean()
    
    def get_loss_function(self, loss_type):
        if loss_type == "mae":
            return nn.L1Loss()
        elif loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "huber":
            return nn.SmoothL1Loss(beta=self.huber_beta)
        elif loss_type == "quantile":
            q = self.config.get("quantile_q", 0.9)
            return lambda preds, target: self.quantile_loss(preds, target, q)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def train_single_fold(self, model, train_loader, val_loader, fold_idx):
        """Train model for a single fold"""
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx + 1}/{self.n_splits}")
        print(f"{'='*50}")
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        fold_history = []
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                # Forward pass
                output = model(batch)
                target = batch["target"].float().view(-1)
                loss = self.criterion(output.view(-1), target)
                # output_raw = torch.expm1(output)
                # target_raw = torch.expm1(target)
                # loss = self.criterion(output_raw.view(-1), target_raw)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                # Debug first batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
                    print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            avg_val_loss = self.validate_fold(model, val_loader)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
                print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return best_val_loss, fold_history, best_model_state

    def validate_fold(self, model, val_loader):
        """Validate model on validation set"""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                output = model(batch)
                target = batch["target"].float().view(-1)
                loss = self.criterion(output.view(-1), target)
                # output_raw = torch.expm1(output)
                # target_raw = torch.expm1(target)
                # loss = self.criterion(output_raw.view(-1), target_raw)


                total_loss += loss.item()
        
        return total_loss / len(val_loader)

    def save_fold_model(self, model_state, fold_idx, val_loss):
        """Save individual fold model"""
        model_filename = f"fold_{fold_idx+1}_model_{val_loss:.4f}.pt"
        model_path = os.path.join(self.model_save_dir, model_filename)
        
        torch.save({
            'model_state_dict': model_state,
            'fold_idx': fold_idx,
            'val_loss': val_loss,
            'config': self.config
        }, model_path)
        
        return model_path

    def load_fold_models(self):
        """Load all saved fold models"""
        self.best_models = []
        self.model_paths = []

        # Load from saved results file
        results_file = f"train_logs/kfold_results_{self.loss_type}_lr{self.lr}_{self.n_splits}fold.json"

        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            try:
                self.fold_loss = (results['k_fold_results']['fold_losses'])
            except KeyError:
                print(f"Warning: 'fold_losses' not found in {results_file}. Using empty list.")

            # Get model paths from results
            if 'model_paths' in results:
                for model_path in results['model_paths']:
                    if os.path.exists(model_path):
                        checkpoint = torch.load(model_path, map_location=self.device)
                        self.best_models.append(checkpoint['model_state_dict'])
                        self.model_paths.append(model_path)
                        
                    else:
                        print(f"Warning: Model file {model_path} not found!")
        
            else:
                print(f"Warning: No model paths found in {results_file}.")
        else:
            print(f"Warning: Results file {results_file} not found. No models loaded.")

        return len(self.best_models) > 0

    def k_fold_train(self, model_class, dataset, config):
        """Main k-fold cross validation training"""
        print(f"Starting {self.n_splits}-Fold Cross Validation")
        print(f"Dataset size: {len(dataset)}")
        print(f"Folds: {self.n_splits}, Samples per fold: ~{len(dataset)//self.n_splits}")
        
        # Create k-fold splitter
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # Store results for each fold
        fold_val_losses = []
        
        # Get indices for splitting
        indices = list(range(len(dataset)))
        
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices)):
            print(f"\nFold {fold_idx + 1}: Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
            
            # Create subset datasets
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            
            # Create data loaders
            train_loader = DataLoader(
                train_subset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                drop_last=True
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
            
            # Initialize fresh model for this fold
            model = model_class(config).to(self.device)
            
            # Train the fold
            best_val_loss, fold_history, best_model_state = self.train_single_fold(
                model, train_loader, val_loader, fold_idx
            )
            
            # Save the fold model
            model_path = self.save_fold_model(best_model_state, fold_idx, best_val_loss)
            
            # Store results
            fold_val_losses.append(best_val_loss)
            self.fold_results.append({
                'fold': fold_idx + 1,
                'best_val_loss': best_val_loss,
                'history': fold_history,
                'model_path': model_path
            })
            self.best_models.append(best_model_state)
            self.model_paths.append(model_path)
            
            print(f"Fold {fold_idx + 1} completed. Best Val Loss: {best_val_loss:.4f}")
            print(f"Model saved to: {model_path}")
        
        # Calculate overall statistics
        mean_val_loss = np.mean(fold_val_losses)
        std_val_loss = np.std(fold_val_losses)
        
        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Mean Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
        print(f"Individual Fold Losses: {[f'{loss:.4f}' for loss in fold_val_losses]}")
        print(f"Best Single Fold: {min(fold_val_losses):.4f}")
        print(f"Worst Single Fold: {max(fold_val_losses):.4f}")
        
        # Save results
        self.save_results(mean_val_loss, std_val_loss, fold_val_losses)
        
        return mean_val_loss, std_val_loss, self.best_models

    def predict_ensemble(self, model_class, test_dataset, config):
        """Make predictions using ensemble of all fold models"""
        # First try to load models if not already loaded
        if not self.best_models:
            if not self.load_fold_models():
                raise ValueError("No trained models found! Please run k-fold training first.")
        
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        all_predictions = []
        
        # Get predictions from each fold model
        for fold_idx, model_state in enumerate(self.best_models):
            print(f"Getting predictions from fold {fold_idx + 1} model...")
            
            # Load model and weights
            model = model_class(config).to(self.device)
            model.load_state_dict(model_state)
            model.eval()
            
            fold_predictions = []
            
            with torch.no_grad():
                for batch in test_loader:
                    for key in batch:
                        batch[key] = batch[key].to(self.device)
                    
                    output = model(batch)
                    fold_predictions.extend(output.view(-1).cpu().numpy())
                    # output_raw = torch.expm1(output)  # Apply expm1 to get original scale
                    # fold_predictions.extend(output_raw.view(-1).cpu().numpy())
            
            all_predictions.append(np.array(fold_predictions))
        
        # Ensemble predictions (average)
        if not self.fold_loss:
            ensemble_predictions = np.mean(all_predictions, axis=0)
        else:
            inv_losses = [1 / loss for loss in self.fold_loss]
            total = sum(inv_losses)
            self.weights = [w / total for w in inv_losses]
            print(f"🧮 Computed weights: {self.weights}")
            if len(all_predictions) != len(self.weights):
                raise ValueError("❌ 預測結果數量與 fold 數不一致")
            ensemble_predictions = sum ( w * p for w, p in zip(self.weights, all_predictions))
    
        
        # Get ground truth
        ground_truth = []
        for batch in test_loader:
            ground_truth.extend(batch["target"].view(-1).numpy())
        
        return ensemble_predictions, ground_truth, all_predictions

    def save_results(self, mean_loss, std_loss, fold_losses):
        """Save k-fold results to file"""
        results = {
            'config': self.config,
            'k_fold_results': {
                'n_splits': self.n_splits,
                'mean_val_loss': float(mean_loss),
                'std_val_loss': float(std_loss),
                'fold_losses': [float(loss) for loss in fold_losses]
            },
            'fold_details': self.fold_results,
            'model_paths': self.model_paths  # Include model paths for loading later
        }
        
        filename = f"train_logs/kfold_results_{self.loss_type}_lr{self.lr}_{self.n_splits}fold.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filename}")
        print(f"Models saved in directory: {self.model_save_dir}")




# Usage example:
"""
# Training with k-fold
config = {
    'loss_type': 'huber',
    'lr': 1e-5,
    'epochs': 50,
    'batch_size': 4,
    'weight_decay': 1e-3,
    'n_splits': 5,
    'patience': 10
}

trainer = KFoldTrainer(config)

# Train with k-fold
mean_loss, std_loss, best_models = trainer.k_fold_train(
    model_class=Model,
    dataset=your_dataset,
    config=model_config
)

# Make ensemble predictions
ensemble_preds, ground_truth, individual_preds = trainer.predict_ensemble(
    model_class=Model,
    test_dataset=test_dataset,
    config=model_config
)
"""