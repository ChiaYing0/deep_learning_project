import torch.utils.data as data
import numpy as np

class Dataset(data.Dataset):

    def init_img_dataset(self, config):
        # [Image]
        # 準備資料集
        return None

    def init_meta_dataset(self, config):
        # [Meta]
        # 準備資料集
        return None

    def init_target_dataset(self, config):
        # [Target]
        return None

    def __init__(self, config, mode='train'):

        self.mode = mode
        self.init_img_dataset(config)
        self.init_meta_dataset(config)
        self.init_target_dataset(config)

    def __len__(self):
        #[Meta]
        # 資料集有多少筆資料
        return 64

    def get_img_data(self, index):
        # [Image]
        # 拿指定index的資料
        return np.zeros((3, 5, 5)).astype(np.float32) # shape = [channel, height, width]

    def get_meta_data(self, index):
        # [Meta]
        # 拿指定index的資料

        return np.zeros([6]).astype(np.float32) # shape = [n_dim]

    def get_target_data(self, index):
        # [Target]
        # return fake target (e.g. 0) when mode = inference
        return 0

    def __getitem__(self, index):

        img_data = self.get_img_data(index)
        meta_data = self.get_meta_data(index)
        target_data = self.get_target_data(index)

        return {
            'img': img_data,
            'meta': meta_data,
            'target': target_data
        }