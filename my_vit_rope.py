import torch
import torch.nn as nn
import torch.nn.functional as F
import rope_vit_main.deit.models_v2 as rope_model
import rope_vit_main.deit.models_v2_rope as rope_model_v2

from functools import partial
from einops import rearrange
from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class My_RopeVit(nn.Module):
    def __init__(self, rope_theta=100.0, rope_mixed=False, use_ape=False, img_size=128, patch_size=16, in_chans=1, num_classes=5, embed_dim=192, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None, init_scale=1e-4, slice_num=11, frame_num=30, feature_num=20,
                 block_layers = rope_model.Block, Patch_layer=PatchEmbed,act_layer=nn.GELU, Attention_block = rope_model.Attention, Mlp_block=Mlp):
        super(My_RopeVit, self).__init__()
        self.dropout_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, feature_num))
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])
        self.blocks1 = nn.ModuleList([
            block_layers(
                dim=feature_num, num_heads=10, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.norm1 = norm_layer(feature_num)
        self.linear_image = nn.Linear(embed_dim * slice_num * frame_num, embed_dim)
        self.linear_motion = nn.Linear(feature_num * slice_num * frame_num, embed_dim)
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(2 * embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.use_ape = use_ape
        if not self.use_ape:
            self.pos_embed = None
        self.rope_mixed = rope_mixed
        self.num_heads = num_heads
        self.patch_size = patch_size

        if self.rope_mixed:
            self.compute_cis = partial(rope_model_v2.compute_mixed_cis, num_heads=self.num_heads)

            freqs = []
            for i, _ in enumerate(self.blocks):
                freqs.append(
                    rope_model_v2.init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

            t_x, t_y = rope_model_v2.init_t_xy(end_x=img_size // patch_size, end_y=img_size // patch_size)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(rope_model_v2.compute_axial_cis, dim=embed_dim // num_heads, theta=rope_theta)

            freqs_cis = self.compute_cis(end_x=img_size // patch_size, end_y=img_size // patch_size)
            self.freqs_cis = freqs_cis
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward_image_features(self, x):
        B, S, T, H, W = x.shape
        x = rearrange(x, 'b s t w h -> (b s t) 1 w h')
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B*S*T, -1, -1)
        if self.use_ape:
            pos_embed = self.pos_embed
            if pos_embed.shape[-2] != x.shape[-2]:
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                pos_embed = pos_embed.view(
                    1, (img_size[1] // patch_size[1]), (img_size[0] // patch_size[0]), self.embed_dim
                ).permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed, size=(H // patch_size[1], W // patch_size[0]), mode='bicubic', align_corners=False
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            x = x + pos_embed
        x = torch.cat((cls_tokens, x), dim=1)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]
        x = rearrange(x, '(b s t) l -> b (s t l)', b=B, s=S, t=T)
        x = self.linear_image(x)
        return x
    def forward_motion_features(self, x):
        B, S, T, Q, L = x.shape
        x = rearrange(x, 'b s t q l -> (b s t) q l')
        cls_tokens = self.cls_token1.expand(B * S * T, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for i, blk in enumerate(self.blocks1):
            x = blk(x)
        x = self.norm1(x)
        x = x[:, 0]
        x = rearrange(x, '(b s t) l -> b (s t l)', b=B, s=S, t=T)
        x = self.linear_motion(x)
        return x

    def forward(self, x, y):
        x = self.forward_motion_features(x)
        y = self.forward_image_features(y)
        x = torch.cat((x, y), dim=1)
        x = self.head(x)
        return x