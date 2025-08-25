import torch
from torch import nn
from net.ResMLP_Block import ResMLP_Block, Affine
from einops import rearrange

class MotionInteraction(nn.Module):
    def __init__(self, channel:int=120, dim:int=22, depth:int=6, dropout=0.1, activation=nn.ReLU):
        super(MotionInteraction, self).__init__()
        self.blocks = nn.ModuleList([
            ResMLP_Block(dim=dim, channel=channel, dropout=dropout, act_layer=activation)
            for _ in range(depth)
        ])
        self.affine = Affine(dim)
        self.lin = nn.Linear(dim * channel, channel)
        # self.pooling = nn.AvgPool1d(kernel_size=channel)
        # self.pooling = nn.AvgPool1d(kernel_size=dim)
    def forward(self, x): #x:[B, S, T, Q, L]
        batch, slice, frame = x.shape[0], x.shape[1], x.shape[2]
        x = rearrange(x, 'b s t q l -> (b s t) q l')
        for block in self.blocks:
            x = block(x)
        x = self.affine(x)
        x = rearrange(x, 'm q l -> m (q l)')
        x = self.lin(x)
        # x = self.pooling(x.permute(0,2,1)).squeeze(-1)
        # x = self.pooling(x).squeeze(-1)
        x = rearrange(x, '(b s t) l -> b s t l', b=batch, s=slice, t=frame)
        return x

class MOST_scan(nn.Module):
    def __init__(self, dim:int=20, channel:int=300, depth:int=6, dropout=0.1, activation=nn.ReLU):
        super(MOST_scan, self).__init__()
        self.blockstf = nn.ModuleList([
            ResMLP_Block(dim=dim, channel=channel, dropout=dropout, act_layer=activation)
            for _ in range(depth)
        ])
        self.blockstr = nn.ModuleList([
            ResMLP_Block(dim=dim, channel=channel, dropout=dropout, act_layer=activation)
            for _ in range(depth)
        ])
        self.blockssf = nn.ModuleList([
            ResMLP_Block(dim=dim, channel=channel, dropout=dropout, act_layer=activation)
            for _ in range(depth)
        ])
        self.affine1 = Affine(dim)
        self.affine2 = Affine(dim)
        self.affine3 = Affine(dim)
        self.pooling1 = nn.AvgPool1d(kernel_size=dim)
        self.pooling2 = nn.AvgPool1d(kernel_size=dim)
        self.pooling3 = nn.AvgPool1d(kernel_size=dim)
    def forward(self, x):
        tf_x = temporal_forward(x)
        tr_x = temporal_reverse(x)
        sf_x = spatial_forward(x)
        # tr_x = temporal_sshape(x)
        for blk_tf, blk_tr, blk_sf in zip(self.blockstf, self.blockstr, self.blockssf):
            tf_x = blk_tf(tf_x)
            tr_x = blk_tr(tr_x)
            sf_x = blk_sf(sf_x)
        tf_x = self.affine1(tf_x)
        tr_x = self.affine2(tr_x)
        sf_x = self.affine3(sf_x)
        tf_x = self.pooling1(tf_x).squeeze(-1)
        tr_x = self.pooling2(tr_x).squeeze(-1)
        sf_x = self.pooling3(sf_x).squeeze(-1)
        MF = torch.cat([tf_x, tr_x, sf_x], dim=1)
        # MF = torch.cat([tr_x, tr_x, tr_x], dim=1)
        return MF

class MotionEncoder(nn.Module):
    def __init__(self, slice:int=11, frame:int=20, angle:int=120, dim:int=20,
                 depth:int=6, dropout=0.1, activation=nn.ReLU):
        super(MotionEncoder, self).__init__()
        self.encoder1 = MotionInteraction(channel=angle, dim=dim, depth=depth, dropout=dropout, activation=activation)
        self.encoder2 = MOST_scan(dim=angle, channel=slice * frame, depth=depth, dropout=dropout, activation=activation)
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        return x

def temporal_forward(x):
    batch, slice, frame, dim = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    device = x.device
    tf_x = torch.zeros((batch, slice*frame, dim), device=device)
    for s in range(slice):
        for t in range(frame):
            tf_x[:, s * frame + t, :] = x[:, s, t, :]
    return tf_x

def temporal_reverse(x):
    batch, slice, frame, dim = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    device = x.device
    tr_x = torch.zeros((batch, slice*frame, dim), device=device)
    for s in range(slice):
        for t in range(frame):
            tr_x[:, (slice - s) * frame - t - 1, :] = x[:, s, t, :]
    return tr_x

def temporal_sshape(x):
    batch, slice, frame, dim = x.shape
    device = x.device
    ts_x = torch.zeros((batch, slice*frame, dim), device=device)
    for s in range(slice):
        for t in range(frame):
            if s % 2 == 0:
                ts_x[:, s * frame + t, :] = x[:, s, t, :]
            else:
                ts_x[:, (slice - s) * frame - t - 1, :] = x[:, s, t, :]
    return ts_x

def spatial_forward(x):
    batch, slice, frame, dim = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    device = x.device
    sf_x = torch.zeros((batch, slice*frame, dim), device=device)
    for s in range(slice):
        for t in range(frame):
            sf_x[:, t * slice + s, :] = x[:, s, t, :]
    return sf_x

if __name__ == '__main__':
    net = MotionEncoder()
    x = torch.randn(16, 11, 20, 120, 20)
    y = net(x)
    print(y.shape)