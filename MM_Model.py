import torch.nn as nn
import torch
from positional_encodings.torch_encodings import PositionalEncodingPermute2D


class FusionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, img, meta):
        gate = self.gate(torch.cat([img, meta], dim=1))
        return gate * img + (1 - gate) * meta


class MM_Model(nn.Module):
    def __init__(self, config, img_dim, meta_dim):
        super(MM_Model, self).__init__()

        self.dim = config["mm_model_dim"]
        self.pos_encoder = PositionalEncodingPermute2D(self.dim)
        self.img_align_mlp = nn.Conv2d(img_dim, self.dim, 1)
        self.meta_align_mlp = nn.Linear(meta_dim, self.dim)
        self.img_mode_emb = nn.Parameter(torch.rand(self.dim), requires_grad=True)
        self.meta_mode_emb = nn.Parameter(torch.rand(self.dim), requires_grad=True)
        self.num_layers = config.get("mm_model_num_layers", 3)
        self.num_heads = config.get("mm_model_num_heads", 4)

        # self.transformer = nn.TransformerEncoderLayer(
        #     d_model = self.dim,
        #     nhead = 4,
        #     batch_first=False
        # )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.dim,
                nhead=self.num_heads,
                batch_first=False,
                dropout=0.05,  # Add dropout
            ),
            num_layers=self.num_layers,  # Use multiple layers
        )

        # self.regression_head = nn.Linear(self.dim, 1)
        self.regression_head = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, config.get("mm_model_output_dim", 4)),  # <- 分類任務
        )
        self.fusion_gate = FusionGate(self.dim)

    def add_pos_encoding(self, img_logit):

        pos_enc = self.pos_encoder(img_logit)
        img_logit += pos_enc

        return img_logit

    def add_mode_emb(self, img_logit, meta_logit):

        img_logit += self.img_mode_emb.view(1, -1, 1)
        meta_logit += self.meta_mode_emb.view(1, -1, 1)

        return img_logit, meta_logit

    def forward(self, img_logit, meta_logit):

        img_logit = self.img_align_mlp(img_logit)
        img_logit = self.add_pos_encoding(img_logit)

        img_logit = torch.flatten(img_logit, -2, -1)

        meta_logit = self.meta_align_mlp(meta_logit)
        meta_logit = meta_logit.unsqueeze(2)

        img_logit, meta_logit = self.add_mode_emb(img_logit, meta_logit)

        # 融合 img + meta 成一個 token
        img_token = img_logit.mean(-1)       # [B, D]
        meta_token = meta_logit.squeeze(2)   # [B, D]
        fused_token = self.fusion_gate(img_token, meta_token)  # [B, D]
        fused_token = fused_token.unsqueeze(2)   # [B, D, 1]              # [B, D, 1]
        
        # 拼接 img patch + 融合 token（用來取代 meta token）
        concat_logit = torch.cat([img_logit, fused_token], 2)   # [B, D, H*W+1]
        concat_logit = concat_logit.permute(2, 0, 1)   

        # concat_logit = torch.cat([img_logit, meta_logit], 2)
        # concat_logit = concat_logit.permute(2, 0, 1)

        output = self.transformer(concat_logit)
        output = output.permute(1, 2, 0)
        output = self.classification_head(output.mean(-1))

        # # 測 MetaNet Only
        # # output = self.regression_head(meta_logit.squeeze(2))

        return output
