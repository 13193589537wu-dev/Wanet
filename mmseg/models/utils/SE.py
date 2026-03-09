import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel_size=7):
        super(SE, self).__init__()

        # 通道注意力模块（Squeeze-Excitation with Avg + Max pooling）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力模块（如 CBAM 中的设计）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=spatial_kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # ---- 通道注意力部分 ----
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        pooled = avg_pool + max_pool  # 方向2：融合平均+最大池化结果

        channel_attn = self.fc(pooled).view(b, c, 1, 1)
        x_channel = x * channel_attn.expand_as(x)

        # ---- 空间注意力部分 ----
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_attn = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        x_out = x_channel * spatial_attn

        return x_out

# 示例用法
# if __name__ == "__main__":
#     # 生成一个随机张量，模拟输入：batch size = 4, channels = 64, height = width = 32
#     x = torch.randn(4, 64, 32, 32)
#     # 创建一个SELayer实例，通道数为64
#     se_layer = SELayer(channel=64)
#     # 通过SELayer调整输入特征
#     y = se_layer(x)
#     # 打印输出张量的形状，应该与输入相同
#     print(y.shape)  # 输出: torch.Size([4, 64, 32, 32])