import torch.utils.data as data
import numpy as np
import pandas as pd
import os
import torch
from torch import tensor
from PIL import Image
from torchvision import transforms


class Dataset(data.Dataset):

    def init_img_dataset(self, config):
        # [Image]
        # 準備資料集
        # 從 img 欄位取出圖片檔案名稱
        self.img_paths = self.df["img"].values
        self.folder = "./images"

        # transform
        if self.mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

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
        img_name = self.img_paths[index]
        img_path = os.path.join(self.folder, img_name)

        if not os.path.exists(img_path):
            print(f"！！圖片不存在：{img_path}\n")
            return torch.zeros(3, 224, 224)  # 回傳黑圖代替，避免 crash
        
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)  # (C, H, W)

        # arr = np.zeros((3, 5, 5), dtype=np.float32)
        # return tensor(arr)
        # shape = [channel, height, width]
        # ---------------------------------

    def get_meta_data(self, index):
        # [Meta]
        # 拿指定index的資料
        return tensor(self.meta[index])  # shape = [n_dim]

    def get_target_data(self, index):
        # [Target]
        # 無論是train, validation, test 都回傳target
        return tensor(self.target[index])

        # # [Target]
        # # return fake target (e.g. 0) when mode = inference
        # if self.mode != "inference":
        #     return tensor(self.target[index])
        # else:
        #     return tensor(0.0)
        

    def __getitem__(self, index):
        return {
            "img": self.get_img_data(index),
            "meta": self.get_meta_data(index),
            "target": self.get_target_data(index),
        }
