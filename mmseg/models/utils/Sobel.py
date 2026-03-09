import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelEdgeModule(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_kernel_x = torch.tensor([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_kernel_y = torch.tensor([[1, 2, 1],
                                       [0, 0, 0],
                                       [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        self.weight_x = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, x):  # x: [B, 1, H, W]
        edge_x = F.conv2d(x, self.weight_x, padding=1)
        edge_y = F.conv2d(x, self.weight_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edge
class SobelInputEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = SobelEdgeModule()

    def forward(self, x):  # x: [B, 3, H, W]
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)  # [B, 1, H, W]
        edge = self.sobel(gray)   # [B, 1, H, W]
        return torch.cat([x, edge], dim=1)  # [B, 4, H, W]