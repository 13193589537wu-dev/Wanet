import torch
import torch.nn as nn
import torch.nn.functional as F
# # from mmcv.cnn import Conv2d, kaiming_init, constant_init, trunc_normal_init
# # from mmcv.runner import BaseModule
# from mmcv.cnn.bricks import DropPath
# # 假设 DWT_HighAdapter, DWT_LowAdapter, DWT_2D, DAPPM_head, ARABlock 等已定义
# from your_module import DWT_HighAdapter, DWT_LowAdapter, DWT_2D, DAPPM_head, ARABlock, LocalEnhancementModule, DynamicConv

class GeometryAwareModule(nn.Module):
    """几何感知模块，增强沟渠的细长结构和纹理特征"""
    def __init__(self, in_channels, reduction=8):
        super(GeometryAwareModule, self).__init__()
        # 几何形状分支：细长核卷积捕捉线性结构
        self.geo_conv_h = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=(1, 3), padding=(0, 1))
        self.geo_conv_v = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=(3, 1), padding=(1, 0))
        self.geo_fuse = nn.Conv2d(in_channels // reduction * 2, in_channels, 1)
        self.geo_bn = nn.SyncBatchNorm(in_channels)
        self.geo_act = nn.Sigmoid()

        # 纹理分支：基于特征梯度增强边缘
        self.tex_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.SyncBatchNorm(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 通道注意力：动态调整通道权重
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 融合层
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.fusion_bn = nn.SyncBatchNorm(in_channels)
        self.fusion_act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 几何形状分支：捕捉细长结构
        geo_h = self.geo_conv_h(x)  # 水平细长核
        geo_v = self.geo_conv_v(x)  # 垂直细长核
        geo = torch.cat([geo_h, geo_v], dim=1)
        geo = self.geo_fuse(geo)
        geo = self.geo_bn(geo)
        geo_mask = self.geo_act(geo)  # 形状引导掩码

        # 纹理分支：基于特征梯度
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        grad = grad_x + grad_y
        tex_mask = self.tex_conv(grad)  # 纹理边缘掩码

        # 通道注意力
        channel_mask = self.channel_attn(x)

        # 融合几何和纹理特征
        enhanced = x * geo_mask * tex_mask * channel_mask
        fused = self.fusion(torch.cat([x, enhanced], dim=1))
        fused = self.fusion_bn(fused)
        fused = self.fusion_act(fused)

        # 残差连接
        out = x + fused
        return out
if __name__ == "__main__":
    # 假设通道为64，尺寸为64x64
    dummy_input = torch.randn(1, 64, 64, 64)
    module = GeometryAwareModule(in_channels=64)
    output = module(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    assert dummy_input.shape == output.shape, "输入输出尺寸不一致，不能即插即用"
    print("GeometryAwareModule 即插即用测试通过！")
