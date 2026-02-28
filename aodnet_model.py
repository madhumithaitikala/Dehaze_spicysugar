"""
AOD-Net: All-in-One Dehazing Network (ICCV 2017)
Paper: https://arxiv.org/abs/1707.06543

This model reformulates the atmospheric scattering model as:
    J(x) = K(x) * I(x) - K(x) + b
where K(x) is estimated by a lightweight CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AODNet(nn.Module):
    """
    AOD-Net architecture.
    Input: Hazy image (B, 3, H, W)
    Output: Clear image (B, 3, H, W)
    """
    def __init__(self):
        super(AODNet, self).__init__()
        # Multi-scale feature extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        # Fusion layer
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1  # bias constant

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)       # (B, 6, H, W)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)       # (B, 6, H, W)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)  # (B, 12, H, W)
        k = F.relu(self.conv5(cat3))

        # J(x) = K(x) * I(x) - K(x) + b
        output = k * x - k + self.b
        return F.relu(output)
