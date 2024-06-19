import torch
import torch.nn as nn

from .basic_blocks import ConvBlock, SEResNet, DiffusionNet, SEResNetDecoder
from .common_manipulations import Manipulation


class Encoder(nn.Module):

    def __init__(self, img_size, en_c, en_blocks, wm_len, diffusion_length=256):
        """
        :param img_size: length of input image square.
        :param en_c: number of channels of input image.
        :param en_blocks: number of blocks.
        :param wm_len: length of input watermark.
        """
        super(Encoder, self).__init__()
        self.img_size = img_size

        self.conv_head_img = ConvBlock(3, en_c)
        self.conv_img = SEResNet(en_c, en_c, en_blocks)

        self.diffusion_length = diffusion_length
        self.expand_dim = int(self.diffusion_length ** 0.5)
        self.wm_exp = nn.Linear(wm_len, self.diffusion_length)
        self.conv_wm_head = nn.Sequential(
            ConvBlock(1, en_c),
            DiffusionNet(en_c, en_c, blocks=en_blocks),
            SEResNet(en_c, en_c, blocks=1)
        )
        self.conv_wm = SEResNet(en_c, en_c, blocks=en_blocks)

        self.conv_cat = ConvBlock(en_c * 2, en_c)
        self.conv_tail = nn.Conv2d(en_c + 3, 3, kernel_size=1)

    def forward(self, img, wm):
        img_encode = self.conv_head_img(img)  # [b, 3, 128, 128] -> [b, 16, 128, 128]
        img_encode = self.conv_img(img_encode)  # [b, 16, 128, 128] -> SE doesn't change dim

        wm_expand = self.wm_exp(wm)  # [b, 64]  -> Upsample the watermark

        wm_expand = wm_expand.view(-1, 1, self.expand_dim, self.expand_dim)  # [b, 1, 8, 8]
        wm_expand = self.conv_wm_head(wm_expand)  # [b, 16, 128, 128] from [b, 16, 8, 8], [b, 16, 16, 16], ...
        wm_expand = self.conv_wm(wm_expand)  # [b, 16, 128, 128]

        img_wm = torch.cat([img_encode, wm_expand], dim=1)  # [b, 32, 128, 128]
        img_wm = self.conv_cat(img_wm)  # [b, 16, 128, 128]
        img_wm = torch.cat([img_wm, img], dim=1)  # [b, 19, 128, 128]

        out = self.conv_tail(img_wm)
        return out


class Decoder(nn.Module):

    def __init__(self, img_size, de_c, de_blocks, wm_len, diffusion_length=256):
        super(Decoder, self).__init__()

        self.img_size = img_size
        self.wm_len = wm_len
        self.diffusion_length = diffusion_length

        self.conv_head = nn.Sequential(
            ConvBlock(3, de_c),
            SEResNetDecoder(de_c, de_c, de_blocks + 3),
            ConvBlock(de_c * (2 ** (de_blocks + 2)), de_c)
        )
        self.conv = SEResNet(de_c, de_c, blocks=de_blocks + 1, do_attn=False)
        self.conv_tail = ConvBlock(de_c, 1)
        self.wm_layer = nn.Linear(self.diffusion_length, self.wm_len)

    def forward(self, x):
        x = self.conv_head(x)  # [b, 3, 128, 128] -> [b, 16, 128, 128] ... [b, 256, 8, 8] -> [b, 16, 8, 8]
        x = self.conv(x)  # [b, 16, 8, 8]
        x = self.conv_tail(x)  # [b, 1, 8, 8]
        x = x.view(x.shape[0], -1)  # [b, 64]

        x = self.wm_layer(x)  # [b, 64]
        return x


class LPMark(nn.Module):

    def __init__(self, img_size, en_c, en_blocks, de_c, de_blocks, wm_len, device, noise_layers):
        super(LPMark, self).__init__()
        self.device = device
        self.encoder = Encoder(img_size, en_c, en_blocks, wm_len)
        self.manipulation = Manipulation(noise_layers)
        self.decoder = Decoder(img_size, de_c, de_blocks, wm_len)

    def forward(self, img, wm):
        encoded_img = self.encoder(img, wm)
        manipulated_img = self.manipulation([encoded_img, img, self.device])
        decoded_wm = self.decoder(manipulated_img)
        return encoded_img, manipulated_img, decoded_wm
