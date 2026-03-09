import torch
import torch.nn as nn
from einops import rearrange
 # self.attn_l = AttentionLePEWithMultiscale(
        #     dim=in_channels,
        #     num_heads=num_heads,
        #     qkv_bias=True,      # Depending on your specific needs, can be True or False
        #     attn_drop=0.1,      # Example dropout rate for attention
        #     proj_drop=0.1,      # Example dropout rate for projection
        #     side_dwconv=5,      # Size of the dilated convolution kernel
        #     dilation=2          # Dilation factor for dilated convolutions
        # )
class AttentionLePEWithMultiscale(nn.Module):
    """
    Vanilla attention with Local Enhancement with Position Encoding (LEPE)
    + Multi-scale feature fusion
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5,
                 dilation=2, scales=[1, 2, 4]):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.scales = scales

        # 空洞卷积，使用 dilation 增强感受野，padding=4 保持尺寸
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=4, dilation=dilation, groups=dim)

        # 多尺度特征的卷积
        self.scale_convs = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=s, dilation=s) for s in self.scales])

        # 投影层将多尺度特征降维
        self.proj_multi_scale = nn.Linear(len(scales) * dim, dim)

    def forward(self, x):
        """
        Args:
            x: NCHW tensor
        Returns:
            NCHW tensor
        """
        _, _, H, W = x.size()
        x = rearrange(x, 'n c h w -> n (h w) c')

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 增强局部特征，保持尺寸一致
        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')

        # 多尺度特征
        multi_scale_features = []
        for scale_conv in self.scale_convs:
            scale_feat = scale_conv(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
            multi_scale_features.append(rearrange(scale_feat, 'n c h w -> n (h w) c'))
        multi_scale_feat = torch.cat(multi_scale_features, dim=-1)
        multi_scale_feat = self.proj_multi_scale(multi_scale_feat)

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe
        x = x + multi_scale_feat

        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'n (h w) c -> n c h w', h=H, w=W)
        return x