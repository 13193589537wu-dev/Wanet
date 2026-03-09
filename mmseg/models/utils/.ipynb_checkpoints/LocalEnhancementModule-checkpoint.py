import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv(nn.Module):
    """动态卷积模块，基于DyConv思想"""
    def __init__(self, in_channels, out_channels, kernel_size=3, num_kernels=4):
        super(DynamicConv, self).__init__()
        self.num_kernels = num_kernels
        self.conv_kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            for _ in range(num_kernels)
        ])
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_kernels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        weights = self.attention(x).view(b, self.num_kernels, 1, 1)  # [B, num_kernels, 1, 1]
        out = 0
        for i, conv in enumerate(self.conv_kernels):
            out += conv(x) * weights[:, i:i+1, :, :]  # 动态加权卷积
        return out

class LocalEnhancementModule(nn.Module):
    """局部增强模块，结合小核卷积、动态卷积和边界感知"""
    def __init__(self, in_channels, reduction=8):
        super(LocalEnhancementModule, self).__init__()

        # 小核卷积：深度可分离卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, in_channels, 1)

        # 动态卷积
        self.dynamic_conv = DynamicConv(in_channels, in_channels, kernel_size=3, num_kernels=4)

        # 边界感知分支
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 融合层
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 小核卷积
        local_features = self.depthwise(x)
        local_features = self.pointwise(local_features)
        local_features = self.relu(local_features)

        # 动态卷积
        dynamic_features = self.dynamic_conv(local_features)

        # 边界感知：基于特征图梯度
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        boundary_features = grad_x + grad_y  # [B, C, H, W]
        boundary_mask = self.boundary_conv(boundary_features)  # [B, C, H, W]
        boundary_guided_features = dynamic_features * boundary_mask

        # 融合
        fused_features = self.fusion(torch.cat([local_features, boundary_guided_features], dim=1))
        fused_features = self.bn(fused_features)
        fused_features = self.relu(fused_features)

        # 残差连接
        out = x + fused_features
        return out
if __name__ == "__main__":
    # 假设通道数为64，输入尺寸为64x64
    dummy_input = torch.randn(1, 64, 64, 64)  # [B, C, H, W]
    module = LocalEnhancementModule(in_channels=64)
    output = module(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    assert dummy_input.shape == output.shape, "输入输出尺寸不一致，不能即插即用"
    print("模块即插即用测试通过！")