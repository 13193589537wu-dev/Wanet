import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Conv2d,ConvModule)
from mmcv.runner import BaseModule
from ..builder import BACKBONES
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,trunc_normal_init,normal_init)
from timm.models.layers import DropPath
from pytorch_wavelets import DWTForward  
import pywt
from ..utils.PPA import PPA 
# from ..utils.Sobel import SobelInputEnhancer
from ..utils.AttentionLePE import AttentionLePEWithMultiscale
from ..utils.SE import SE
from ..utils.EGA import EGA
# from ..utils.ARA import ARABlock
# from ..utils.LocalEnhancementModule import LocalEnhancementModule
# from ..utils.GeometryAwareModule import GeometryAwareModule
# from ..utils.BoundaryRegularizationModule import BoundaryRegularizationModule
@BACKBONES.register_module()
class SCTNet(BaseModule):
    """
    The SCTNet implementation based on mmSegmentation.
    Args:
        layer_nums (List, optional): The layer nums of every stage. Default: [2, 2, 2, 2]
        base_channels (int, optional): The base channels. Default: 64
        spp_channels (int, optional): The channels of DAPPM. Defualt: 128
        in_channels (int, optional): The channels of input image. Default: 3
        num_heads (int, optional): The num of heads in CFBlock. Default: 8
        drop_rate (float, optional): The drop rate in CFBlock. Default:0.
        drop_path_rate (float, optional): The drop path rate in CFBlock. Default: 0.2
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """
    def __init__(self,
                 layer_nums=[2, 2, 2, 2],
                 base_channels=64,  #Slim32
                 spp_channels=128,  #Slim64
                 in_channels=3,  
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 pretrained=None,
                 init_cfg=None):
        super(SCTNet,self).__init__(init_cfg=init_cfg)
        self.base_channels = base_channels
        base_chs = base_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, base_chs, kernel_size=3, stride=2, padding=1),
            nn.SyncBatchNorm(base_chs),
            nn.ReLU(),
            nn.Conv2d(
                base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            nn.SyncBatchNorm(base_chs),
            nn.ReLU(), )
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs,
                                       layer_nums[0])
        #---------------------------------------新增
        # self.dwt_high = DWT_HighAdapter(in_channels, base_chs)        # 用于增强 layer2 输入
       
        #-------------------------------------------------------
        self.layer2 = self._make_layer(
            BasicBlock, base_chs, base_chs * 2, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, base_chs * 2, base_chs * 4, layer_nums[2], stride=2)

        self.layer3_2 = CFBlock(
            in_channels=base_chs * 4,
            out_channels=base_chs * 4,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate)
        
        # self.ppa = PPA(in_features=base_chs * 4, filters=base_chs * 4)  #增加
      
        self.convdown4 = nn.Sequential(
            nn.Conv2d(
                base_chs*4, base_chs*8, kernel_size=3, stride=2, padding=1),
            nn.SyncBatchNorm(base_chs*8),
            nn.ReLU(),)
        self.layer4 = CFBlock(
            in_channels=base_chs * 8,
            out_channels=base_chs * 8,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate)
          
        
        # self.hybrid_mixer_layer4 = HybridTokenMixer(
        #         dim=base_chs * 8,
        #         kernel_size=3,
        #         num_groups=2,
        #         num_heads=8,   # 增强全局关系建模
        #         sr_ratio=1,
        #         reduction_ratio=8
        #     )
        
        self.layer5 = CFBlock(
            in_channels=base_chs * 8,
            out_channels=base_chs * 8,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate)
        
        # self.hybrid_mixer_layer5 = HybridTokenMixer(
        #     dim=base_chs * 8,
        #     kernel_size=3,
        #     num_groups=2,
        #     num_heads=8,
        #     sr_ratio=1,
        #     reduction_ratio=8
        # )
        self.spp = DAPPM_head(
            base_chs * 8, spp_channels, base_chs * 2)

         # 低频增强：挪到 layer3_2 之后
        self.dwt_low = DWT_LowAdapter(in_channels, base_chs * 2)
        
        # self.brm = BoundaryRegularizationModule(in_channels=base_chs * 8, reduction=8)
        #self.gam = GeometryAwareModule(in_channels=base_chs * 8, reduction=8)
        # self.ema = EMA(base_chs * 2, factor=32)  # 因为 stage4 输出是 8c    #新增
        
        # if self.init_cfg.type == 'Pretrained':
        # # if self.init_cfg is not None and self.init_cfg.get('type') == 'Pretrained':
        #     super(SCTNet, self).init_weights()
        # else:
        #     self.init_weight()
        if self.init_cfg is not None and self.init_cfg.get('type') == 'Pretrained':
            super(SCTNet, self).init_weights()
        else:
            self.init_weight()

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def init_weight(self):
        self.conv1.apply(self._init_weights_kaiming)
        self.layer1.apply(self._init_weights_kaiming)
        self.layer2.apply(self._init_weights_kaiming)
        self.layer3.apply(self._init_weights_kaiming)
        self.convdown4.apply(self._init_weights_kaiming)
        self.spp.apply(self._init_weights_kaiming)



    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride),
                nn.SyncBatchNorm(out_channels))

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=True))
            else:
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        #------------原版本----------------
        x1 = self.layer1(self.conv1(x))  # c, 1/4
      
        x2 = self.layer2(self.relu(x1))  # 2c, 1/8

        #----------------小波变换版本-------------
        # x_conv1 = self.conv1(x)                   # conv1输出，1/4
        # x1 = self.layer1(x_conv1)                 # layer1输出，1/4
        
        # # 高频增强：放在 layer2 之前
        # x1_high = self.dwt_high(x, x1)            # 融合原图高频 + x1输出
        # x2 = self.layer2(self.relu(x1_high))      # 2c, 1/8
        
        #------------------------到此------------------------------------
        x3_1 = self.layer3(self.relu(x2))  # 4c, 1/16
        x3 = self.layer3_2(self.relu(x3_1))  # 4c, 1/16  
        #-----------------------增----------------------------------------
        # x3 = self.ppa(x3)
        
        x4_down=self.convdown4(x3)
        x4 = self.layer4(self.relu(x4_down))  # 8c, 1/32
        # x4 = x4 + self.hybrid_mixer_layer4(x4)  # 全局连续性增强
        x5 = self.layer5(self.relu(x4))  # 8c, 1/32
        # x5 = x5 + self.hybrid_mixer_layer5(x5)  # 可选：进一步全局一致性
        # x5 = self.gam(x5)  # 插入 GAM
        # x5 = self.brm(x5)  # 插入 BRM
        x6=self.spp(x5)   #2c, 1/32
        #-------------增-----------------
        x6 = self.dwt_low(x, x6)
    
        # x6 = self.ema(x6)               # EMA 加强细节空间关系
        ##############
        # 低频分量升维
       
        ###############
        x7 = F.interpolate(
            x6, size=x2.shape[2:], mode='bilinear')  # 2c, 1/8
        x_out = torch.cat([x2, x7], dim=1)  # 4c, 1/8
        logit_list = [x_out, x2,[x,[x_out,x5,x3]]]

        return logit_list

#conv->bn->relu->conv->bn->relu
class BasicBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super(BasicBlock,self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.SyncBatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.SyncBatchNorm(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out if self.no_relu else self.relu(out)


#BN->Conv->GELU->drop->Conv2->drop
class MLP(BaseModule):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP,self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.SyncBatchNorm(in_channels, eps=1e-06)  #TODO,1e-6?
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class ConvolutionalAttention(BaseModule):
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8):
        super(ConvolutionalAttention,self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv =nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 =nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        trunc_normal_init(self.kv, std=0.001)
        trunc_normal_init(self.kv3, std=0.001)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.)
            constant_init(m.bias, val=.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)


    def _act_dn(self, x):
        x_shape = x.shape  # n,c_inter,h,w
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,c_inter,h,w -> n,heads,c_inner//heads,hw
        x = F.softmax(x, dim=3)   
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)  
        x = x.reshape([x_shape[0], self.inter_channels, h, w]) 
        return x

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor. (n,c,h,w)
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        x1 = F.conv2d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=(3,0))  
        x1 = self._act_dn(x1)  
        x1 = F.conv2d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=(3,0))  
        x3 = F.conv2d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=(0,3)) 
        x3 = self._act_dn(x3)
        x3 = F.conv2d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=(0,3)) 
        x=x1+x3
        return x

class CFBlock(BaseModule):
    """
    The CFBlock implementation based on PaddlePaddle.
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        drop_rate (float, optional): The drop rate in MLP. Default:0.
        drop_path_rate (float, optional): The drop path rate in CFBlock. Default: 0.2
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.):
        super(CFBlock,self).__init__()
        in_channels_l = in_channels
        out_channels_l = out_channels
        self.attn_l = ConvolutionalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=64,
            num_heads=num_heads)
        
        # self.attn_l = AttentionLePEWithMultiscale(
        #         dim=in_channels,
        #         num_heads=num_heads,
        #         qkv_bias=True,      # Depending on your specific needs, can be True or False
        #         attn_drop=0.1,      # Example dropout rate for attention
        #         proj_drop=0.1,      # Example dropout rate for projection
        #         side_dwconv=5,      # Size of the dilated convolution kernel
        #         dilation=2          # Dilation factor for dilated convolutions
        #     )
      
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def forward(self, x):
        x_res = x
        x = x_res + self.drop_path(self.attn_l(x))
        x = x + self.drop_path(self.mlp_l(x)) 
        return x

class DAPPM_head(BaseModule):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(DAPPM_head,self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=5, stride=2, padding=2, count_include_pad =True),
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=9, stride=4, padding=4, count_include_pad =True),
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=17, stride=8, padding=8, count_include_pad =True),
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale0 = nn.Sequential(
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.process1 = nn.Sequential(
            nn.SyncBatchNorm(
                inter_channels),
            nn.ReLU(),
            Conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process2 = nn.Sequential(
            nn.SyncBatchNorm(
                inter_channels),
            nn.ReLU(),
            Conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process3 = nn.Sequential(
            nn.SyncBatchNorm(
                inter_channels),
            nn.ReLU(),
            Conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process4 = nn.Sequential(
            nn.SyncBatchNorm(
                inter_channels),
            nn.ReLU(),
            Conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.compression = nn.Sequential(
            nn.SyncBatchNorm(
                inter_channels * 5),
            nn.ReLU(),
            Conv2d(
                inter_channels * 5,
                out_channels,
                kernel_size=1))
        self.shortcut = nn.Sequential(
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x_shape = x.shape[2:]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1((F.interpolate(
                self.scale1(x), size=x_shape, mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(
            self.scale2(x), size=x_shape, mode='bilinear') + x_list[1]))))
        x_list.append(
            self.process3((F.interpolate(
                self.scale3(x), size=x_shape, mode='bilinear') + x_list[2])))
        x_list.append(
            self.process4((F.interpolate(
                self.scale4(x), size=x_shape, mode='bilinear') + x_list[3])))

        out = self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)
        return out
        
###自己添加 EMA注意力模块
class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels % self.groups == 0, "channels must be divisible by groups"
        self.group_channels = channels // self.groups
 
        # 空间注意力模块（保持不变）
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(1, self.group_channels)
        self.conv1x1 = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1)
 
        # 多尺度卷积分支（显式包含5×1和1×5卷积）
        self.multi_scale_conv = nn.ModuleList([
            # 原始设计中的3×1和1×3卷积
            nn.Sequential(
                nn.Conv2d(self.group_channels, self.group_channels, kernel_size=(3,1), padding=(1,0)),
                nn.BatchNorm2d(self.group_channels),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(self.group_channels, self.group_channels, kernel_size=(1,3), padding=(0,1)),
                nn.BatchNorm2d(self.group_channels),
                nn.SiLU(inplace=True)
            ),
            # 新增的5×1和1×5卷积（通过膨胀卷积模拟大核效果，减少参数量）
            nn.Sequential(
                nn.Conv2d(self.group_channels, self.group_channels, kernel_size=(5,1), padding=(2,0)),  # 5×1卷积
                nn.BatchNorm2d(self.group_channels),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(self.group_channels, self.group_channels, kernel_size=(1,5), padding=(0,2)),  # 1×5卷积
                nn.BatchNorm2d(self.group_channels),
                nn.SiLU(inplace=True)
            ),
            # 保留普通3×3卷积作为基础特征提取
            nn.Sequential(
                nn.Conv2d(self.group_channels, self.group_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.group_channels),
                nn.SiLU(inplace=True)
            )
        ])
 
        # 融合多分支（1×1卷积调整通道数）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.group_channels * 5, self.group_channels, kernel_size=1),  # 分支数从3→5
            nn.BatchNorm2d(self.group_channels),
            nn.SiLU(inplace=True)
        )
 
        # 最终输出处理（严格保持维度）
        self.proj = nn.Sequential(
            nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1),
            nn.BatchNorm2d(self.group_channels)
        )
 
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.view(b * self.groups, self.group_channels, h, w)  # [B*G, C//G, H, W]
 
        # ===== 空间注意力分支（保持不变）=====
        x_h = self.pool_h(group_x)  # [B*G, C//G, H, 1]
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # [B*G, C//G, W, 1]
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # [B*G, C//G, H+W, 1]
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        spatial_attn = x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()  # [B*G, C//G, H, W]
        x1 = self.gn(group_x * spatial_attn)  # 注意力增强
 
        # ===== 多尺度特征提取 =====
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            multi_scale_features.append(conv(x1))  # 基于注意力增强后的特征提取
        x2 = torch.cat(multi_scale_features, dim=1)  # [B*G, C//G*5, H, W]
        x2 = self.fusion_conv(x2)  # 融合多尺度特征 [B*G, C//G, H, W]
 
        # ===== 最终输出（无残差连接）=====
        out = self.proj(x2)
        out = out.view(b, c, h, w)  # 恢复原始形状
        
        return out  # 严格保持输入输出维度一致
 




class DWT_2D(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0).float())

    def forward(self, x):
        ll = F.conv2d(x, self.w_ll.repeat(x.shape[1],1,1,1), stride=2, groups=x.shape[1])
        lh = F.conv2d(x, self.w_lh.repeat(x.shape[1],1,1,1), stride=2, groups=x.shape[1])
        hl = F.conv2d(x, self.w_hl.repeat(x.shape[1],1,1,1), stride=2, groups=x.shape[1])
        hh = F.conv2d(x, self.w_hh.repeat(x.shape[1],1,1,1), stride=2, groups=x.shape[1])
        return ll, lh, hl, hh

# class DWT_Adapter(nn.Module):
#     def __init__(self, in_ch, out_ch, wave='haar', upsample=True):
#         super().__init__()
#         self.dwt = DWT_2D(wave)
#         self.conv1x1 = nn.Conv2d(in_ch * 4, out_ch, 1)
#         self.upsample = upsample

#     def forward(self, x):
#         h, w = x.shape[2], x.shape[3]
#         ll, lh, hl, hh = self.dwt(x)
#         x_dwt = torch.cat([ll, lh, hl, hh], dim=1)
#         x_dwt = self.conv1x1(x_dwt)
#         if self.upsample:
#             x_dwt = F.interpolate(x_dwt, size=(h, w), mode='bilinear', align_corners=False)
#         return x_dwt
class DWT_HighAdapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dwt = DWT_2D()
        self.fuse = nn.Conv2d(in_ch * 3, out_ch, 1)  # 高频通道融合
        self.out_conv = nn.Conv2d(out_ch * 2, out_ch, 1)  # concat后降维

    def forward(self, x_raw, x_feat):
        _, lh, hl, hh = self.dwt(x_raw)
        high = torch.cat([lh, hl, hh], dim=1)
        high = self.fuse(high)
        high = F.interpolate(high, size=x_feat.shape[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat([x_feat, high], dim=1)
        return self.out_conv(x_cat)

class DWT_LowAdapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dwt = DWT_2D()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.out_conv = nn.Conv2d(out_ch * 2, out_ch, 1)

    def forward(self, x_raw, x_feat):
        ll, _, _, _ = self.dwt(x_raw)
        ll = self.proj(ll)
        ll = F.interpolate(ll, size=x_feat.shape[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat([x_feat, ll], dim=1)
        return self.out_conv(x_cat)

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)
        
# def gauss_kernel(channels=3, cuda=True):
#     kernel = torch.tensor([[1., 4., 6., 4., 1],
#                            [4., 16., 24., 16., 4.],
#                            [6., 24., 36., 24., 6.],
#                            [4., 16., 24., 16., 4.],
#                            [1., 4., 6., 4., 1.]])
#     kernel /= 256.
#     kernel = kernel.repeat(channels, 1, 1, 1)
#     if cuda:
#         kernel = kernel.cuda()
#     return kernel


# def downsample(x):
#     return x[:, :, ::2, ::2]


# def conv_gauss(img, kernel):
#     img = F.pad(img, (2, 2, 2, 2), mode='reflect')
#     out = F.conv2d(img, kernel, groups=img.shape[1])
#     return out


# def upsample(x, channels):
#     cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
#     cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
#     cc = cc.permute(0, 1, 3, 2)
#     cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
#     cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
#     x_up = cc.permute(0, 1, 3, 2)
#     return conv_gauss(x_up, 4 * gauss_kernel(channels))


# def make_laplace(img, channels):
#     filtered = conv_gauss(img, gauss_kernel(channels))
#     down = downsample(filtered)
#     up = upsample(down, channels)
#     if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
#         up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
#     diff = img - up
#     return diff


# def make_laplace_pyramid(img, level, channels):
#     current = img
#     pyr = []
#     for _ in range(level):
#         filtered = conv_gauss(current, gauss_kernel(channels))
#         down = downsample(filtered)
#         up = upsample(down, channels)
#         if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
#             up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
#         diff = current - up
#         pyr.append(diff)
#         current = down
#     pyr.append(current)
#     return pyr


# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#         )

#     def forward(self, x):
#         avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
#         max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
#         channel_att_sum = avg_out + max_out

#         scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return x * scale


# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

#     def forward(self, x):
#         x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
#         x_out = self.spatial(x_compress)
#         scale = torch.sigmoid(x_out)  # broadcasting
#         return x * scale


# class CBAM(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16):
#         super(CBAM, self).__init__()
#         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
#         self.SpatialGate = SpatialGate()

#     def forward(self, x):
#         x_out = self.ChannelGate(x)
#         x_out = self.SpatialGate(x_out)
#         return x_out


# # Edge-Guided Attention Module
# class EGA(nn.Module):
#     def __init__(self, in_channels):
#         super(EGA, self).__init__()

#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True))

#         self.attention = nn.Sequential(
#             nn.Conv2d(in_channels, 1, 3, 1, 1),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid())

#         self.cbam = CBAM(in_channels)

#     def forward(self, edge_feature, x, pred):
#         residual = x
#         xsize = x.size()[2:]
#         pred = torch.sigmoid(pred)

#         # reverse attention
#         background_att = 1 - pred
#         background_x = x * background_att

#         # boudary attention
#         edge_pred = make_laplace(pred, 1)
#         pred_feature = x * edge_pred

#         # high-frequency feature
#         edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
#         input_feature = x * edge_input

#         fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
#         fusion_feature = self.fusion_conv(fusion_feature)

#         attention_map = self.attention(fusion_feature)
#         fusion_feature = fusion_feature * attention_map

#         out = fusion_feature + residual
#         out = self.cbam(out)
#         return out