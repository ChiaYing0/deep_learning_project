import torch
import torch.nn as nn
from MM_Model import MM_Model

# class MM_Model(nn.Module):

#     def __init__(self, config):

#         super().__init__()

#         pass

#     def forward(self, img_logit, meta_logit):
#         return torch.zeros(img_logit.shape[0], device=img_logit.device)


class Model(nn.Module):

    def get_img_net(self, config):
        # [Image]
        # 實作net
        return nn.Identity()

    def get_img_data_from_batch(self, batch):
        # [Image]
        # 從batch取得network輸入
        return batch["img"]

    def get_img_net_out_dim(self):
        # [Image]
        # img_net output channel 數
        return 3

    def get_meta_net(self, config):
        # [Meta]
        # 實作net
        layers = []
        in_dim = config["meta_input_dim"]

        # 中間層：Linear → BN → ReLU
        for h in config["meta_hidden_dims"]:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            in_dim = h

        layers.append(nn.Linear(in_dim, config["meta_embedding_dim"]))
        layers.append(nn.BatchNorm1d(config["meta_embedding_dim"]))

        return nn.Sequential(*layers)

    def get_meta_data_from_batch(self, batch):
        # [Meta]
        # 從batch取得network輸入
        return batch["meta"]

    def get_meta_net_out_dim(self, config):
        # [Image]
        # meta_net output channel 數
        return config["meta_embedding_dim"]

    def get_mm_net(self, config):
        # [MM]
        # 實作net
        return MM_Model(
            config, self.get_img_net_out_dim(), self.get_meta_net_out_dim(config)
        )

    def __init__(self, config):

        super().__init__()

        self.img_net = self.get_img_net(config)
        self.meta_net = self.get_meta_net(config)
        self.mm_net = self.get_mm_net(config)

    def forward(self, batch):

        img_data = self.get_img_data_from_batch(batch)
        img_logit = self.img_net(img_data)

        meta_data = self.get_meta_data_from_batch(batch)
        meta_logit = self.meta_net(meta_data)

        logit = self.mm_net(img_logit, meta_logit)

        return logit
