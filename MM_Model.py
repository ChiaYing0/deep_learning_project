import torch.nn as nn
import torch
from positional_encodings.torch_encodings import PositionalEncodingPermute2D


class MM_Model(nn.Module):
    def __init__(self, config, img_dim, meta_dim):
        super(MM_Model, self).__init__()

        self.dim = config['mm_model_dim']
        self.pos_encoder = PositionalEncodingPermute2D(self.dim)
        self.img_align_mlp = nn.Conv2d(img_dim, self.dim, 1)
        self.meta_align_mlp = nn.Linear(meta_dim, self.dim)
        self.img_mode_emb = nn.Parameter(torch.rand(self.dim), requires_grad=True)
        self.meta_mode_emb = nn.Parameter(torch.rand(self.dim), requires_grad=True)

        self.transformer = nn.TransformerEncoderLayer(
            d_model = self.dim,
            nhead = 4,
            batch_first=True
        )

        self.regression_head = nn.Linear(self.dim, 1)


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
        
        concat_logit = torch.cat([img_logit, meta_logit], 2)

        output = self.transformer(concat_logit)
        output = self.regression_head(output.mean(-1))

        return output

