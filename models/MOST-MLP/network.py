import torch
import torch.nn as nn
import net.Motion_Encoder as ME
import net.Image_Encoder as IE

class main_net(nn.Module):
    def __init__(self, slice:int=11, frame:int=30, angle:int=120, dim:int=20,
                 depth:int=4, dropout_m=0.1, dropout_i=0.1, activation=nn.ReLU, img_size:int=128,
                 patch_size:int=16, in_chans:int=1, embed_dim:int=256, num_classes:int=5):
        super(main_net, self).__init__()
        self.motionencoder = ME.MotionEncoder(slice=slice, frame=frame, angle=angle,
                                              dim=dim, depth=depth, dropout=dropout_m,
                                              activation=activation)
        self.imageencoder = IE.ImageEncoder(img_size=img_size, patch_size=patch_size,
                                            in_chans=in_chans, embed_dim=embed_dim, frame=frame,
                                            depth=depth, dropout=dropout_i, activation=activation)
        self.outlayer = nn.Linear(3 * frame * slice + embed_dim, num_classes)
        # self.outlayer = nn.Linear(3 * dim + embed_dim, num_classes)
    def forward(self, xm:torch.Tensor, xi:torch.Tensor):
        """
        xm: [batch_size, slice_size, frame_size, sample_size, dim]
        xi: [batch_size, slice_size, frame_size, img_size, img_size]
        """
        xm = self.motionencoder(xm)
        xi = self.imageencoder(xi)
        x = torch.cat([xm, xi], dim=1)
        x = self.outlayer(x)
        return x

if __name__ == '__main__':
    xm = torch.randn(2, 11, 20, 120, 20).cuda()
    xi = torch.randn(2, 11, 20, 128, 128).cuda()
    net = main_net().cuda()
    y = net(xm, xi)
