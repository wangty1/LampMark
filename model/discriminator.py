import torch.nn as nn

from .basic_blocks import ConvBlock


class Discriminator(nn.Module):

    def __init__(self, dis_c, dis_blocks):
        super(Discriminator, self).__init__()
        self.dis_c = dis_c
        self.dis_blocks = dis_blocks
        self.conv_head = ConvBlock(3, dis_c)
        layers = []
        for _ in range(dis_blocks - 1):
            layers.append(ConvBlock(dis_c, dis_c))
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(dis_c, 1)

    def forward(self, x):
        x = self.conv_head(x)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        out = self.linear(x)
        return out
