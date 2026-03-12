import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels=64, res_scale=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.res_scale = res_scale

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return residual + x * self.res_scale
    
class EDSR(nn.Module):

    def __init__(self, num_blocks=12, channels=64):

        super().__init__()

        # first feature extraction
        self.head = nn.Conv2d(1, channels, 3, padding=1)

        # residual body
        body = []
        for _ in range(num_blocks):
            body.append(ResidualBlock(channels))

        self.body = nn.Sequential(*body)

        # reconstruction
        self.tail = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):

        x_head = self.head(x)

        x_body = self.body(x_head)

        x = x_head + x_body

        x = self.tail(x)

        return x