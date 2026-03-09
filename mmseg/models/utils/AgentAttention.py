import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentAttention(nn.Module):
    def __init__(self, dim, window_size=(7, 7), num_heads=8, agent_num=49, drop_rate=0.):
        super(AgentAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.agent_num = agent_num
        self.scale = (dim // num_heads) ** -0.5

        # 归一化
        self.norm = nn.LayerNorm(dim)

        # 查询、键、值投影
        self.qkv = nn.Linear(dim, dim * 3, bias=True)

        # 代理令牌池化
        self.pool = nn.AdaptiveAvgPool2d(int(agent_num ** 0.5))

        # 位置偏置
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, window_size[0] * window_size[1]))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, window_size[0] * window_size[1], agent_num))
        nn.init.trunc_normal_(self.an_bias, std=0.02)
        nn.init.trunc_normal_(self.na_bias, std=0.02)

        # 深度可分离卷积
        self.dwc = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        B, C, H0, W0 = x.shape

        # 1. 如果大小不能被 window 整除，pad 补齐
        pad_h = (self.window_size[0] - H0 % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W0 % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H, W = x.shape[2:]

        # 2. 进入窗口 attention
        N = self.window_size[0] * self.window_size[1]
        x = x.view(B, C, H // self.window_size[0], self.window_size[0],
                   W // self.window_size[1], self.window_size[1])
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, N, C)

        # 3. 归一化
        x = self.norm(x)

        # 4. QKV 投影
        qkv = self.qkv(x).reshape(-1, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 5. 生成代理令牌
        x_pool = x.view(-1, self.window_size[0], self.window_size[1], C).permute(0, 3, 1, 2)
        agent_tokens = self.pool(x_pool).view(-1, C, self.agent_num).permute(0, 2, 1)

        k_agent = self.qkv(agent_tokens)[:, :, :C].reshape(-1, self.agent_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_agent = self.qkv(agent_tokens)[:, :, C:2*C].reshape(-1, self.agent_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 6. 代理注意力计算
        attn = (q @ k_agent.transpose(-2, -1)) * self.scale + self.an_bias
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v_agent).transpose(1, 2).reshape(-1, N, C)

        # 7. 反向注意力
        attn = (k_agent @ q.transpose(-2, -1)) * self.scale + self.na_bias
        attn = F.softmax(attn, dim=-1)
        x = x + (attn @ v).transpose(1, 2).reshape(-1, N, C)

        # 8. 恢复形状
        x = x.view(B, H // self.window_size[0], W // self.window_size[1],
                  self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)

        # 9. Crop 回原始大小（optional）
        x = x[:, :, :H0, :W0]

        # 10. 深度可分离卷积
        x = self.dwc(x)
        return x
