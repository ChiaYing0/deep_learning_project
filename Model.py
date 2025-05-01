import torch
import torch.nn as nn

class MM_Model(nn.Module):

    def __init__(self, config):

        super().__init__()

        pass

    def forward(self, img_logit, meta_logit):
        return torch.zeros(img_logit.shape[0], device=img_logit.device)

class Model(nn.Module):

    def get_img_net(self, config):
        # [Image]
        # 實作net
        return nn.Identity()

    def get_img_data_from_batch(self, batch):
        # [Image]
        # 從batch取得network輸入
        return batch['img']
    
    def get_meta_net(self, config):
        # [Meta]
        # 實作net
        return nn.Identity()

    def get_meta_data_from_batch(self, batch):
        # [Meta]
        # 從batch取得network輸入
        return batch['meta']

    def get_mm_net(self, config):
        # [MM]
        # 實作net
        return MM_Model(config)

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