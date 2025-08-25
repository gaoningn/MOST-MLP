import torch
from torch import nn
from einops import rearrange
from net.ResMLP_Block import ResMLP_Block, Affine
from timm.models.vision_transformer import PatchEmbed

class ImageEncoder(nn.Module):
    def __init__(self, img_size:int=128, patch_size:int=16, slice:int=11, frame:int=20, in_chans:int=1, embed_dim:int=256, depth:int=6, dropout:float=0.1, activation=nn.ReLU):
        super(ImageEncoder, self).__init__()
        self.patch_embedding = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.dim = patch_size*patch_size
        num_patches = self.patch_embedding.num_patches
        self.blocks = nn.ModuleList([
            ResMLP_Block(dim=embed_dim, channel=num_patches, dropout=dropout, act_layer=activation)
            for _ in range(depth)
        ])
        self.affine = Affine(embed_dim)
        self.pooling = nn.AvgPool1d(kernel_size=num_patches)
        self.lin = nn.Linear(slice * frame * embed_dim, embed_dim)
        # self.pooling1 = nn.AvgPool1d(kernel_size=int(11 * 20))
    def forward(self, x): #x: [b, s, t, h, h]
        batch, slice, frame = x.shape[0], x.shape[1], x.shape[2]
        x = rearrange(x, 'b s t w h -> (b s t) 1 w h')
        x = self.patch_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.affine(x)
        x = self.pooling(x.permute(0, 2, 1)).squeeze(-1)
        x = rearrange(x, '(b s t) l -> b (s t l)', b=batch, s=slice, t=frame, l=self.dim)
        x = self.lin(x)
        return x

if __name__ == '__main__':
    net = ImageEncoder().cuda()
    x = torch.randn(2, 11, 20, 128, 128).cuda()
    y = net(x)

    print(y.shape)
