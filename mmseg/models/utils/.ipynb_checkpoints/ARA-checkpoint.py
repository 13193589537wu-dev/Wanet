#CaraNet: Context Axial Reverse Attention Network for Small Medical Objects Segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()
        self.mode = mode
        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1), stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width
        view = (batch_size, -1, axis)
        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)
        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)
        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)
        out = self.gamma * out + x
        return out

class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1, stride=1, padding=0)
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3), stride=1, padding=1)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)
        return Wx

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_relu(output)
        return output

class CFPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super(CFPModule, self).__init__()
        # 每个分支输出通道数，确保总和为 out_channels
        num_branches = len(dilation_rates)
        branch_channels = out_channels // num_branches
        # 调整最后一个分支通道数以补齐余数
        self.convs = nn.ModuleList()
        for i, d in enumerate(dilation_rates):
            if i == num_branches - 1:
                # 最后一个分支补齐剩余通道
                branch_out = out_channels - (num_branches - 1) * branch_channels
            else:
                branch_out = branch_channels
            self.convs.append(Conv(in_channels, branch_out, kSize=3, stride=1,
                                   padding=d, dilation=d, bn_acti=True))
        self.conv_fuse = Conv(out_channels, out_channels, kSize=1, stride=1, padding=0, bn_acti=True)

    def forward(self, x):
        # 多尺度卷积
        features = [conv(x) for conv in self.convs]
        # 通道融合
        out = torch.cat(features, dim=1)  # [bs, out_channels, h, w]
        out = self.conv_fuse(out)
        return out

class ARABlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ARABlock, self).__init__()
        # 如果未指定 out_channels，设置为与 in_channels 相同
        out_channels = in_channels if out_channels is None else out_channels
        # 多尺度特征金字塔
        self.cfp = CFPModule(in_channels, in_channels, dilation_rates=[1, 2, 4])
        # 轴向注意力模块
        self.aa_kernel = AA_kernel(in_channels, in_channels)
        # 逆向注意力后处理的卷积层
        self.conv1 = Conv(in_channels, in_channels, kSize=3, stride=1, padding=1, bn_acti=True)
        self.conv2 = Conv(in_channels, in_channels, kSize=3, stride=1, padding=1, bn_acti=True)
        self.conv3 = Conv(in_channels, out_channels, kSize=3, stride=1, padding=1, bn_acti=True)

    def forward(self, feature):
        # 多尺度特征提取
        cfp_out = self.cfp(feature)  # [bs, in_channels, h, w]
        # 轴向注意力
        aa_atten = self.aa_kernel(cfp_out)  # [bs, in_channels, h, w]
        # 从 feature 生成逆向注意力掩码
        decoder_ra = -1 * torch.sigmoid(torch.mean(feature, dim=1, keepdim=True)) + 1  # [bs, 1, h, w]
        decoder_ra = decoder_ra.expand(-1, feature.size(1), -1, -1)  # 扩展到与特征图相同的通道数
        # 结合轴向注意力和逆向掩码
        aa_atten_out = decoder_ra * aa_atten  # 元素逐一相乘
        # 卷积处理
        ra_out = self.conv1(aa_atten_out)
        ra_out = self.conv2(ra_out)
        ra_out = self.conv3(ra_out)  # 输出 [bs, out_channels, h, w]
        # 融合逆向注意力输出与输入特征
        out = ra_out + feature  # 与 feature 融合，保持通道数一致
        return out

if __name__ == '__main__':
    # 使用 CPU 或 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 测试 ARABlock
    block = ARABlock(in_channels=32).to(device)  # 默认 out_channels=in_channels
    feature = torch.randn(1, 32, 44, 44).to(device)  # 修复为 32 通道
    out = block(feature)
    print(f"Output shape: {out.shape}")  # 预期输出 [1, 32, 44, 44]