import torch
import torch.nn as nn


class MetaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        in_dim = config["meta_input_dim"]

        for h in config["meta_hidden_dims"]:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(config["meta_dropout"]),
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, config["meta_embedding_dim"]))
        layers.append(nn.LayerNorm(config["meta_embedding_dim"]))

        self.encoder = nn.Sequential(*layers)  # <-- 這行不能漏！

    def forward(self, x):
        return self.encoder(x)
