import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryRegularizationModule(nn.Module):
    """边界正则化模块，挖掘前景与背景关系，规范模糊细节"""
    def __init__(self, in_channels, reduction=8):
        super(BoundaryRegularizationModule, self).__init__()
        # 前景背景对比分支：通道注意力分离前景与背景
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_contrast = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 边界正则化分支：基于特征梯度规范模糊区域
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 3, padding=1),
            nn.SyncBatchNorm(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 融合层
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.fusion_bn = nn.SyncBatchNorm(in_channels)
        self.fusion_act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 前景背景对比分支
        contrast = self.global_pool(x)
        contrast_mask = self.conv_contrast(contrast)  # [B, C, 1, 1]
        contrast_mask = contrast_mask.expand_as(x)  # 广播到原尺寸

        # 边界正则化分支
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        grad = grad_x + grad_y
        boundary_mask = self.boundary_conv(grad)  # 边界掩码

        # 融合
        enhanced = x * contrast_mask * boundary_mask
        fused = self.fusion(torch.cat([x, enhanced], dim=1))
        fused = self.fusion_bn(fused)
        fused = self.fusion_act(fused)

        # 残差连接
        out = x + fused
        return out
