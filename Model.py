import torch
import torch.nn as nn
from MM_Model import MM_Model
from torchvision import models
from meta.MetaNet import MetaNet
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# class MM_Model(nn.Module):

#     def __init__(self, config):

#         super().__init__()

#         pass

#     def forward(self, img_logit, meta_logit):
#         return torch.zeros(img_logit.shape[0], device=img_logit.device)


class Model(nn.Module):

    def get_img_net(self, config):
        # [Image]
        # 實作net，使用 ResNet50 提取特徵
        # base = models.resnet50(pretrained=True)

        # ❗❗❗❗剛剛改過
        weights = EfficientNet_B0_Weights.DEFAULT
        base = efficientnet_b0(weights=weights)
        # 凍結 從config 取得是否凍結 img_net
        # 以及凍結的 layer 名稱
        freeze_img_net = config.get("freeze_img_net", True)
        if freeze_img_net:
            freeze_layers = config.get("freeze_img_layers", [])
            if freeze_layers:
                for name, param in base.named_parameters():
                    if any(name.startswith(layer_name) for layer_name in freeze_layers):
                        param.requires_grad = False
        else:
            print("Not freezing any layers in the image network. Check your config.")
        backbone = nn.Sequential(*list(base.children())[:-1])
        
        return backbone
        # return nn.Identity()

    def get_img_data_from_batch(self, batch):
        # [Image]
        # 從batch取得network輸入
        return batch["img"]

    def get_img_net_out_dim(self):
        # [Image]
        # img_net output channel 數
        # ResNet50 輸出維度為 2048
        #return 2048

        # ❗❗❗❗
        # return 512  # ResNet18 輸出維度為 512
        return 1280 # EfficientNet-B0 輸出維度1280
        

    def get_meta_net(self, config):
        # [Meta] 實作net
        net = MetaNet(config)

        if config.get(
            "meta_pretrain", False
        ):  # meta_pretrain 為 True 則 load pretrained 的 參數
            ckpt_path = config.get("meta_encoder_ckpt")
            if ckpt_path:
                state_dict = torch.load(ckpt_path)
                net.encoder.load_state_dict(state_dict)
                print(f"Loaded pretrained meta encoder from {ckpt_path}")
            else:
                print(
                    "meta_pretrain is True but no 'meta_encoder_ckpt' path is provided."
                )

        else:
            print("Meta Net Loaded from scratch.")

        return net

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
