import torch
from torch import nn
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return self.alpha * x + self.beta

class ResMLP_Block(nn.Module):
    def __init__(self, dim, channel, dropout=0, act_layer=nn.GELU, layerscale=0.2):
        super(ResMLP_Block, self).__init__()
        self.aff1 = Affine(dim)
        self.aff2 = Affine(dim)
        self.lin = nn.Linear(channel, channel)
        # self.drop_path = DropPath(dropout) if dropout > 0 else nn.Identity()
        self.layerscale_1 = nn.Parameter(layerscale * torch.ones(dim))
        self.layerscale_2 = nn.Parameter(layerscale * torch.ones(dim))
        self.mlp = Mlp(in_features=dim, hidden_features=4*dim, act_layer=act_layer, drop=dropout)
    def forward(self, x):
        res1 = self.lin(self.aff1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.layerscale_1 * res1
        res2 = self.mlp(self.aff2(x))
        x = x + self.layerscale_2 * res2
        return x

if __name__ == '__main__':
    model = ResMLP_Block(20, 120)
    x = torch.randn(16, 120, 20)
    y = model(x)
    print(y.shape)