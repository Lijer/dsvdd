import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MLP_NET(BaseNet):

    def __init__(self,input_dim = 32, rep_dim = 128):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, rep_dim),
            nn.LeakyReLU(negative_slope=2e-1, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(rep_dim, rep_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # print(type(x))
        out = self.encoder(x)

        return out

