import torch.utils.data as data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch import tensor


class Dataset(data.Dataset):

    def init_img_dataset(self, config):
        # [Image]
        # 準備資料集
        return None

    def init_meta_dataset(self, config):
        # [Meta]
        drop_cols = ["target", "target_log"]

        meta_cols = (
            self.df.select_dtypes(include=["number"]).drop(columns=drop_cols).columns
        )
        # extract and store as a float32 NumPy array
        self.meta = self.df[meta_cols].astype(np.float32).values

    def init_target_dataset(self, config):
        # [Target]
        if config["use_log"]:
            self.target = self.df["target_log"].astype(np.float32).values
        else:
            self.target = self.df["target"].astype(np.float32).values

        return None

    def __init__(self, config, mode="train"):

        self.mode = mode

        if mode == "train":
            self.df = pd.read_csv(config["train_csv"], index_col="id")
        elif mode == "valid":
            self.df = pd.read_csv(config["val_csv"], index_col="id")
        elif mode == "inference":
            self.df = pd.read_csv(config["test_csv"], index_col="id")

        self.init_img_dataset(config)
        self.init_meta_dataset(config)
        self.init_target_dataset(config)

    def __len__(self):
        # [Meta]
        # 資料集有多少筆資料
        return len(self.target)

    def get_img_data(self, index):
        # [Image]
        # 拿指定index的資料
        arr = np.zeros((3, 5, 5), dtype=np.float32)
        return tensor(arr)
        # shape = [channel, height, width]

    def get_meta_data(self, index):
        # [Meta]
        # 拿指定index的資料
        return tensor(self.meta[index])  # shape = [n_dim]

    def get_target_data(self, index):
        # [Target]
        # return fake target (e.g. 0) when mode = inference
        if self.mode != "inference":
            return tensor(self.target[index])
        else:
            return tensor(0.0)

    def __getitem__(self, index):
        return {
            "img": self.get_img_data(index),
            "meta": self.get_meta_data(index),
            "target": self.get_target_data(index),
        }
