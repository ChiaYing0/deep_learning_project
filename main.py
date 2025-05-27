import argparse
from Model import Model
from Dataset import Dataset
from Trainer import Trainer
import yaml
import torch
import numpy as np


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="train")

    args = parser.parse_args()
    return args


def main(args):

    config = yaml.safe_load(open(args.config))

    trainer = Trainer(config)

    if args.mode == "train":

        model = Model(config)
        train_set = Dataset(config, mode="train")

        # 考慮 k-fold [Train]
        valid_set = Dataset(config, mode="valid")

        trainer.train(model, train_set, valid_set)

    elif args.mode == "inference":

        model = Model(config)
        test_set = Dataset(config, mode="inference")

        # 載入訓練好的 checkpoint
        checkpoint_path = config.get("checkpoint_path", "model_ckpt.pt")
        print(f"Loading checkpoint from {checkpoint_path} ...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=trainer.device))

        predictions, ground_truth = trainer.test(model, test_set)

        # 儲存結果
        np.save("predictions.npy", predictions)
        np.save("ground_truth.npy", ground_truth)
        print("Inference done.")


if __name__ == "__main__":

    args = get_args()
    main(args)
