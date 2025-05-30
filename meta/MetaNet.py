import torch
import torch.nn as nn


class MetaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # layers = []
        # in_dim = config["meta_input_dim"]

        # for h in config["meta_hidden_dims"]:
        #     layers += [
        #         nn.Linear(in_dim, h),
        #         nn.BatchNorm1d(h),
        #         nn.ReLU(),
        #         nn.Dropout(config["meta_dropout"]),
        #     ]
        #     in_dim = h

        # layers.append(nn.Linear(in_dim, config["meta_embedding_dim"]))
        # layers.append(nn.LayerNorm(config["meta_embedding_dim"]))

        # self.encoder = nn.Sequential(*layers)  # <-- 這行不能漏！

        embedding_dim = config.get("meta_embedding_dim", 128)

        # Use shape input dimension if specified, otherwise use all features
        if config.get("use_shap", False):
            input_dim = config.get("meta_input_dim_shap", 20)
        else:
            input_dim = config.get("meta_input_dim", 51)

        dropout = config.get("meta_dropout", 0.1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),   
            nn.BatchNorm1d(256),
            nn.ReLU(),  
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
