import torch.utils.data as data
import numpy as np
import pandas as pd
import os
import glob
import torch
from torch import tensor
from PIL import Image
from torchvision import transforms


class Dataset(data.Dataset):

    def init_img_dataset(self, config):
        # [Image]
        # 準備資料集
        # 從 img 欄位取出圖片檔案名稱
        self.img_paths = self.df.index.values
        self.folder = "./train_images"

        # transform
        if self.mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        return None

    def init_meta_dataset(self, config):
        # [Meta]
        drop_cols = ["AdoptionSpeed"]

        meta_cols = (
            self.df.select_dtypes(include=["number"]).drop(columns=drop_cols).columns
        )
        # extract and store as a float32 NumPy array

        self.meta = self.df[meta_cols].astype(np.float32).values

    def init_target_dataset(self, config):
        # [Target]

        self.target = self.df["AdoptionSpeed"].astype(np.int64).values

    def __init__(self, config, mode="train"):

        self.mode = mode

        if mode == "train":
            self.df = pd.read_csv(config["train_csv"], index_col="PetID")
        elif mode == "valid":
            self.df = pd.read_csv(config["val_csv"], index_col="PetID")
        elif mode == "inference":
            self.df = pd.read_csv(config["test_csv"], index_col="PetID")

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
        pet_id = self.img_paths[index]
        img_glob_pattern = os.path.join(self.folder, f"{pet_id}-*.jpg")

        # 找所有符合的圖片
        matched_imgs = glob.glob(img_glob_pattern)
        if not matched_imgs:
            print(f"！！沒有找到圖片：{pet_id}-*.jpg")
            return torch.zeros(3, 224, 224)  # 回傳黑圖

        # 優先使用編號最小的那張 (理論上只有一張圖片)
        matched_imgs.sort()  # 確保順序一致
        img_path = matched_imgs[0]

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
        return tensor(self.target[index], dtype=torch.long)

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
