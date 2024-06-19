import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import random
import string
import os

from kornia.filters import GaussianBlur2d, MedianBlur


class Identity(nn.Module):
    """ Identity noise layer. Does nothing on the image. Useful in combinations of multi-manipulations. """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, img_wm_device):
        return img_wm_device[0]


class Dropout(nn.Module):
    """ Randomly drops pixels from the watermarked image and substitutes pixels from the original image. """

    def __init__(self, ratio):
        super(Dropout, self).__init__()
        self.ratio = ratio

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]
        img = img_wm_device[1]
        device = img_wm_device[2]

        mask = torch.rand(img.shape).to(device)
        img_wm = torch.where(mask > self.ratio * 1.0, img_wm, img)

        return img_wm


class Resize(nn.Module):
    """ Resize the watermarked image. """

    def __init__(self, ratio, interpolation='nearest'):
        super(Resize, self).__init__()
        self.ratio = ratio
        self.interpolation = interpolation

    def forward(self, img_wm_device):
        img = img_wm_device[1]
        img_wm = img_wm_device[0]
        img_wm = nn.functional.interpolate(img_wm, scale_factor=(self.ratio, self.ratio), mode=self.interpolation,
                                           recompute_scale_factor=True)
        img_wm = transforms.Resize((img.shape[-1], img.shape[-2]))(img_wm)
        return img_wm


class GaussianNoise(nn.Module):
    """ Add Gaussian noise with free adjustment of mean and variance. """

    def __init__(self, mean=0, std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]
        device = img_wm_device[2]
        noise_tensor = torch.randn(img_wm.shape) * self.std + self.mean
        img_wm = img_wm + noise_tensor.to(device)
        return img_wm


class GaussianBlur(nn.Module):
    """ Gaussian blur. """

    def __init__(self, sigma, kernel=3):
        super(GaussianBlur, self).__init__()
        self.sigma = sigma
        self.kernel = kernel
        self.gaussian_filter = GaussianBlur2d((self.kernel, self.kernel), (self.sigma, self.sigma))

    def forward(self, img_wm_device):
        return self.gaussian_filter(img_wm_device[0])


class SaltPepper(nn.Module):
    """ Salt & Pepper noise with noise ratio. Implementation adopted from MBRS. """

    def __init__(self, ratio):
        super(SaltPepper, self).__init__()
        self.ratio = ratio

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]
        device = img_wm_device[2]
        prob_0 = self.ratio / 2
        prob_1 = 1 - prob_0
        rdn = torch.rand(img_wm.shape).to(device)

        img_wm = torch.where(rdn > prob_1, torch.zeros_like(img_wm).to(device), img_wm)
        img_wm = torch.where(rdn < prob_0, torch.ones_like(img_wm).to(device), img_wm)

        return img_wm


class MedBlur(nn.Module):
    """ Median blur. """

    def __init__(self, kernel):
        super(MedBlur, self).__init__()
        self.kernel = kernel
        self.middle_filter = MedianBlur((self.kernel, self.kernel))

    def forward(self, img_wm_device):
        return self.middle_filter(img_wm_device[0])


class Brightness(nn.Module):
    """ Brightness of image. """

    def __init__(self, factor):
        super(Brightness, self).__init__()
        self.factor = factor

    def forward(self, img_wm_device):
        img_wm = transforms.functional.adjust_brightness(img_wm_device[0], self.factor)
        return img_wm


class Contrast(nn.Module):
    """ Contrast of image. """

    def __init__(self, factor):
        super(Contrast, self).__init__()
        self.factor = factor

    def forward(self, img_wm_device):
        img_wm = transforms.functional.adjust_contrast(img_wm_device[0], self.factor)
        return img_wm


class Saturation(nn.Module):
    """ Saturation of image. """

    def __init__(self, factor):
        super(Saturation, self).__init__()
        self.factor = factor

    def forward(self, img_wm_device):
        img_wm = transforms.functional.adjust_saturation(img_wm_device[0], self.factor)
        return img_wm


class Hue(nn.Module):
    """ Hue of image. """

    def __init__(self, factor):
        super(Hue, self).__init__()
        self.factor = factor

    def forward(self, img_wm_device):
        img_wm = transforms.functional.adjust_hue(img_wm_device[0], self.factor)
        return img_wm


""" The following JPEG implementations are directly borrowed from the MBRS project as they proposed a good approximation
    and our contribution has nothing to do with inventing a new JPEG compression method. 
    
    More details can be found at https://github.com/jzyustc/MBRS
"""


class JpegTest(nn.Module):
    def __init__(self, Q, subsample=2, path="temp/"):
        super(JpegTest, self).__init__()
        self.Q = Q
        self.subsample = subsample
        self.path = path
        if not os.path.exists(path): os.mkdir(path)

    def get_path(self):
        return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg"

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]

        shape = img_wm.shape
        noised_image = torch.zeros_like(img_wm)

        for i in range(shape[0]):
            single_image = ((img_wm[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)

            file = self.get_path()
            while os.path.exists(file):
                file = self.get_path()
            im.save(file, format="JPEG", quality=self.Q, subsampling=self.subsample)
            jpeg = np.array(Image.open(file), dtype=np.uint8)
            os.remove(file)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            noised_image[i] = transform(jpeg).unsqueeze(0).to(img_wm.device)

        return noised_image


class JpegBasic(nn.Module):
    def __init__(self):
        super(JpegBasic, self).__init__()

    def std_quantization(self, image_yuv_dct, scale_factor, round_func=torch.round):

        luminance_quant_tbl = (torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
            image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

        chrominance_quant_tbl = (torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
            image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

        q_image_yuv_dct = image_yuv_dct.clone()
        q_image_yuv_dct[:, :1, :, :] = image_yuv_dct[:, :1, :, :] / luminance_quant_tbl
        q_image_yuv_dct[:, 1:, :, :] = image_yuv_dct[:, 1:, :, :] / chrominance_quant_tbl
        q_image_yuv_dct_round = round_func(q_image_yuv_dct)
        return q_image_yuv_dct_round

    def std_reverse_quantization(self, q_image_yuv_dct, scale_factor):

        luminance_quant_tbl = (torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
            q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

        chrominance_quant_tbl = (torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
            q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

        image_yuv_dct = q_image_yuv_dct.clone()
        image_yuv_dct[:, :1, :, :] = q_image_yuv_dct[:, :1, :, :] * luminance_quant_tbl
        image_yuv_dct[:, 1:, :, :] = q_image_yuv_dct[:, 1:, :, :] * chrominance_quant_tbl
        return image_yuv_dct

    def dct(self, image):
        # coff for dct and idct
        coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image.shape[2] // 8
        image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
        image_dct = torch.matmul(coff, image_dct)
        image_dct = torch.matmul(image_dct, coff.permute(1, 0))
        image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image_dct

    def idct(self, image_dct):
        # coff for dct and idct
        coff = torch.zeros((8, 8), dtype=torch.float).to(image_dct.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image_dct.shape[2] // 8
        image = torch.cat(torch.cat(image_dct.split(8, 2), 0).split(8, 3), 0)
        image = torch.matmul(coff.permute(1, 0), image)
        image = torch.matmul(image, coff)
        image = torch.cat(torch.cat(image.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image

    def rgb2yuv(self, image_rgb):
        image_yuv = torch.empty_like(image_rgb)
        image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] \
                                  + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] \
                                  - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] \
                                  - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
        return image_yuv

    def yuv2rgb(self, image_yuv):
        image_rgb = torch.empty_like(image_yuv)
        image_rgb[:, 0:1, :, :] = image_yuv[:, 0:1, :, :] + 1.40198758 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 1:2, :, :] = image_yuv[:, 0:1, :, :] - 0.344113281 * image_yuv[:, 1:2, :, :] \
                                  - 0.714103821 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 2:3, :, :] = image_yuv[:, 0:1, :, :] + 1.77197812 * image_yuv[:, 1:2, :, :]
        return image_rgb

    def yuv_dct(self, image, subsample):
        # clamp and convert from [-1,1] to [0,255]
        image = (image.clamp(-1, 1) + 1) * 255 / 2

        # pad the image so that we can do dct on 8x8 blocks
        pad_height = (8 - image.shape[2] % 8) % 8
        pad_width = (8 - image.shape[3] % 8) % 8
        image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image)

        # convert to yuv
        image_yuv = self.rgb2yuv(image)

        assert image_yuv.shape[2] % 8 == 0
        assert image_yuv.shape[3] % 8 == 0

        # subsample
        image_subsample = self.subsampling(image_yuv, subsample)

        # apply dct
        image_dct = self.dct(image_subsample)

        return image_dct, pad_width, pad_height

    def idct_rgb(self, image_quantization, pad_width, pad_height):
        # apply inverse dct (idct)
        image_idct = self.idct(image_quantization)

        # transform from yuv to to rgb
        image_ret_padded = self.yuv2rgb(image_idct)

        # un-pad
        image_rgb = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
                    :image_ret_padded.shape[3] - pad_width].clone()

        return image_rgb * 2 / 255 - 1

    def subsampling(self, image, subsample):
        if subsample == 2:
            split_num = image.shape[2] // 8
            image_block = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
            for i in range(8):
                if i % 2 == 1: image_block[:, 1:3, i, :] = image_block[:, 1:3, i - 1, :]
            for j in range(8):
                if j % 2 == 1: image_block[:, 1:3, :, j] = image_block[:, 1:3, :, j - 1]
            image = torch.cat(torch.cat(image_block.chunk(split_num, 0), 3).chunk(split_num, 0), 2)
        return image


class Jpeg(JpegBasic):
    def __init__(self, Q, subsample=0):
        super(Jpeg, self).__init__()

        # quantization table
        self.Q = Q
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        # subsample
        self.subsample = subsample

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]

        # [-1,1] to [0,255], rgb2yuv, dct
        image_dct, pad_width, pad_height = self.yuv_dct(img_wm, self.subsample)

        # quantization
        image_quantization = self.std_quantization(image_dct, self.scale_factor)

        # reverse quantization
        image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

        # idct, yuv2rgb, [0,255] to [-1,1]
        noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
        return noised_image.clamp(-1, 1)


class JpegSS(JpegBasic):
    def __init__(self, Q, subsample=0):
        super(JpegSS, self).__init__()

        # quantization table
        self.Q = Q
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        # subsample
        self.subsample = subsample

    def round_ss(self, x):
        cond = torch.tensor((torch.abs(x) < 0.5), dtype=torch.float).to(x.device)
        return cond * (x ** 3) + (1 - cond) * x

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]

        # [-1,1] to [0,255], rgb2yuv, dct
        image_dct, pad_width, pad_height = self.yuv_dct(img_wm, self.subsample)

        # quantization
        image_quantization = self.std_quantization(image_dct, self.scale_factor, self.round_ss)

        # reverse quantization
        image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

        # idct, yuv2rgb, [0,255] to [-1,1]
        noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
        return noised_image.clamp(-1, 1)


class JpegMask(JpegBasic):
    def __init__(self, Q, subsample=0):
        super(JpegMask, self).__init__()

        # quantization table
        self.Q = Q
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        # subsample
        self.subsample = subsample

    def round_mask(self, x):
        mask = torch.zeros(1, 3, 8, 8).to(x.device)
        mask[:, 0:1, :5, :5] = 1
        mask[:, 1:3, :3, :3] = 1
        mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
        return x * mask

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]

        # [-1,1] to [0,255], rgb2yuv, dct
        image_dct, pad_width, pad_height = self.yuv_dct(img_wm, self.subsample)

        # mask
        image_mask = self.round_mask(image_dct)

        # idct, yuv2rgb, [0,255] to [-1,1]
        noised_image = self.idct_rgb(image_mask, pad_width, pad_height)
        return noised_image.clamp(-1, 1)


class Combined(nn.Module):
    """ Fits the case that a list of manipulations are to be used for training. Each manipulation is randomly adopted
        for each mini-batch.
    """

    def __init__(self, lst):
        super(Combined, self).__init__()
        self.lst = lst
        if lst is None:
            self.lst = [Identity()]

    def forward(self, img_wm_device):
        idx = random.randint(0, len(self.lst) - 1)
        # print('idx:', idx, ', manipulation:', self.lst[idx])
        return self.lst[idx](img_wm_device)


class Manipulation(nn.Module):

    def __init__(self, layers):
        super(Manipulation, self).__init__()
        for i in range(len(layers)):
            layers[i] = eval(layers[i])
        self.manipulation = layers

    def forward(self, img_wm_device):
        for layer in self.manipulation:
            img_wm = layer(img_wm_device)
        return img_wm
