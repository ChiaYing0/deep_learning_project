import argparse
from Model import Model
from Dataset import Dataset
from Trainer import Trainer
from Trainer_k_fold import KFoldTrainer
import yaml
import torch
import numpy as np
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="train", choices=["train", "inference"])
    parser.add_argument("--use_kfold", action="store_true", help="Use K-fold cross validation")
    
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Create result directory if it doesn't exist
    os.makedirs("result", exist_ok=True)

    if args.mode == "train":
        model_class = Model  # Pass the class, not instance
        train_set = Dataset(config, mode="train")
        
        if args.use_kfold:
            # Use K-Fold Cross Validation
            print("Using K-Fold Cross Validation...")
            trainer = KFoldTrainer(config)
            
            # Combine train and validation sets for k-fold
            valid_set = Dataset(config, mode="valid")
            all_sets = train_set + valid_set
 

            mean_loss, std_loss, best_models = trainer.k_fold_train(
                model_class=Model,
                dataset=all_sets,
                config=config
            )
            
            print(f"K-Fold Training completed!")
            print(f"Mean Validation Loss: {mean_loss:.4f} ± {std_loss:.4f}")
            
        else:
            # Use regular training
            print("Using regular training...")
            trainer = Trainer(config)
            model = Model(config)
            valid_set = Dataset(config, mode="valid")
            trainer.train(model, train_set, valid_set)

    elif args.mode == "inference":
        test_set = Dataset(config, mode="inference")
        
        if args.use_kfold:
            # Use ensemble prediction from k-fold models
            print("Using K-Fold ensemble for inference...")
            trainer = KFoldTrainer(config)
            
            # Load the saved k-fold results to get best models
            kfold_results_file = f"train_logs/kfold_results_{config.get('loss_type', 'mae')}_lr{config.get('lr', 1e-4)}_{config.get('n_splits', 5)}fold.json"
            
            if not os.path.exists(kfold_results_file):
                print(f"K-fold results file {kfold_results_file} not found!")
                print("Please run training with --use_kfold first.")
                return

            try:
                ensemble_preds, ground_truth, individual_preds = trainer.predict_ensemble(
                    model_class=Model,
                    test_dataset=test_set,
                    config=config
                )
                
                # Save ensemble results
                np.save("result/ensemble_predictions.npy", ensemble_preds)
                np.save("result/ground_truth.npy", ground_truth)
                np.save("result/individual_fold_predictions.npy", individual_preds)
                
                print("K-Fold ensemble inference completed!")
                print(f"Ensemble predictions saved to result/ensemble_predictions.npy")
                
            except Exception as e:
                print(f"Error during ensemble prediction: {e}")
                print("Make sure you have trained models using k-fold first.")
                
        else:
            # Use regular single model inference
            print("Using regular single model inference...")
            trainer = Trainer(config)
            model = Model(config)
            
            # Load trained checkpoint
            checkpoint_path = config.get("checkpoint_path", "model_ckpt.pt")
            print(f"Loading best model checkpoint {checkpoint_path}...")
            
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint {checkpoint_path} not found!")
                return
                
            checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            predictions, ground_truth = trainer.test(model, test_set)
            
            # Save results
            np.save("result/predictions.npy", predictions)
            np.save("result/ground_truth.npy", ground_truth)
            print("Regular inference completed!")


if __name__ == "__main__":
    args = get_args()
    main(args)