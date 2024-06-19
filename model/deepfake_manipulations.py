import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import os
import argparse
from argparse import Namespace
import random

# SimSwap utilities.
from .SimSwap.models.models import create_model
from .SimSwap.options.test_options import TestOptions

# InfoSwap utilities.
from .infoswap.modules.encoder128 import Backbone128
from .infoswap.modules.iib import IIB
from .infoswap.modules.aii_generator import AII512
from .infoswap.modules.decoder512 import UnetDecoder512
from .infoswap.preprocess.mtcnn import MTCNN

# UniFace utilities.
from UniFace.generate_swap import Model as UniFace

# StyleMask utilities.
from .stylemask.libs.models.StyleGAN2.model import Generator as StyleGAN2Generator
from .stylemask.libs.models.mask_predictor import MaskPredictor
from .stylemask.libs.utilities.utils import generate_image, generate_new_stylespace
from .stylemask.libs.utilities.stylespace_utils import decoder
from .stylemask.libs.configs.config_models import stylegan2_ffhq_1024
from .stylemask.libs.utilities.utils_inference import invert_image
from .stylemask.libs.models.inversion.psp import pSp
import face_alignment

# StarGAN utilities.
from .stargan.core.solver import Solver

import sys

sys.path.insert(0, './SimSwap')
sys.path.insert(0, './stargan')
sys.path.insert(0, './stylemask')


class SimSwapModel(nn.Module):

    def __init__(self, img_size=128, mode='test'):
        super(SimSwapModel, self).__init__()
        opt = TestOptions().parse()
        self.img_size = img_size
        if self.img_size == 128:
            opt.crop_size = 224
            opt.image_size = 224
            opt.netG = 'global'
        else:
            opt.crop_size = 224
            opt.image_size = 224
            opt.netG = 'global'  # '550000'
        self.sim_swap = create_model(opt)
        self.sim_swap.eval()
        self.arcface_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.mode = mode

    def one_step_swap(self, source, target, device):
        """ Single directional face-swap from source to target. """
        source = self.arcface_norm(source)
        # (112, 112) required by SimSwap.
        source_downsample = F.interpolate(source, size=(112, 112))
        latent_source = self.sim_swap.netArc(source_downsample)
        latent_source = latent_source.detach().to('cpu')
        latent_source = latent_source / np.linalg.norm(latent_source, axis=1, keepdims=True)
        latent_source = latent_source.to(device)

        # The swapped face has facial id of source but attributes of from target.
        swapped_face = self.sim_swap(source, target, latent_source, latent_source, True)
        return swapped_face

    def forward(self, img_wm_device):
        """ When trained on a batch along with ID-Mark, faces are to be swapped within the batch.
            Source ace at index i + 1 is swapped onto the target face at index i.
        """
        img_wm = img_wm_device[0]
        img = img_wm_device[1]
        device = img_wm_device[2]

        if self.img_size == 128:
            resize = transforms.Resize((224, 224))
        else:
            resize = transforms.Resize((224, 224))
        resize_back = transforms.Resize((self.img_size, self.img_size))
        img_wm = resize(img_wm)
        img = resize(img)

        img_source = torch.roll(img, 1, 0)

        swapped_face_wm = self.one_step_swap(img_source, img_wm, device)
        swapped_face = self.one_step_swap(img_source, img, device)
        swapped_face = resize_back(swapped_face)
        swapped_face_wm = resize_back(swapped_face_wm)

        if self.mode != 'train':
            return swapped_face_wm
        else:
            return swapped_face_wm, swapped_face


class InfoSwapModel(nn.Module):
    def __init__(self, device):
        super(InfoSwapModel, self).__init__()
        self.mtcnn = MTCNN()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=2),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.device = device

        """ Prepare Models: """
        root = './model/infoswap/checkpoints_512/w_kernel_smooth'

        pathG = 'ckpt_ks_G.pth'
        pathE = 'ckpt_ks_E.pth'
        pathI = 'ckpt_ks_I.pth'

        self.encoder = Backbone128(50, 0.6, 'ir_se').eval().to(self.device)
        state_dict = torch.load('./model/infoswap/modules/model_128_ir_se50.pth', map_location=self.device)
        self.encoder.load_state_dict(state_dict, strict=True)
        self.G = AII512().eval().to(self.device)
        self.decoder = UnetDecoder512().eval().to(self.device)

        # Define Information Bottlenecks:
        self.N = 10
        _ = self.encoder(torch.rand(1, 3, 128, 128).to(device), cache_feats=True)
        _readout_feats = self.encoder.features[:(self.N + 1)]  # one layer deeper than the z_attrs needed
        in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
        out_c_list = [_readout_feats[i].shape[-3] for i in range(self.N)]

        self.iib = IIB(in_c, out_c_list, device, smooth=True, kernel_size=1)
        self.iib = self.iib.eval()

        self.G.load_state_dict(torch.load(os.path.join(root, pathG), map_location=self.device), strict=True)
        print("Successfully load G!")
        self.decoder.load_state_dict(torch.load(os.path.join(root, pathE), map_location=self.device), strict=True)
        print("Successfully load Decoder!")
        # 3) load IIB:
        self.iib.load_state_dict(torch.load(os.path.join(root, pathI), map_location=self.device), strict=True)
        print("Successfully load IIB!")

        self.param_dict = []
        for i in range(self.N + 1):
            state = torch.load(f'./model/infoswap/modules/weights128/readout_layer{i}.pth', map_location=self.device)
            n_samples = state['n_samples'].float()
            std = torch.sqrt(state['s'] / (n_samples - 1)).to(self.device)
            neuron_nonzero = state['neuron_nonzero'].float()
            active_neurons = (neuron_nonzero / n_samples) > 0.01
            self.param_dict.append([state['m'].to(self.device), std, active_neurons])

    def one_step_swap(self, source, target):
        B = 16
        source = self.transform(source)
        target = self.transform(target)
        source_id = self.encoder(
            F.interpolate(torch.cat((source, target), dim=0)[:, :, 37:475, 37:475], size=[128, 128], mode='bilinear',
                          align_corners=True), cache_feats=True)
        min_std = torch.tensor(0.01).to(self.device)
        readout_feats = [(self.encoder.features[i] - self.param_dict[i][0]) / torch.max(self.param_dict[i][1], min_std)
                         for i in range(self.N + 1)]
        X_id_restrict = torch.zeros_like(source_id).to(self.device)  # [2*B, 512]
        Xt_feats, X_lambda = [], []
        Xt_lambda = []
        Rs_params, Rt_params = [], []
        for i in range(self.N):
            R = self.encoder.features[i]  # [2*B, Cr, Hr, Wr]
            Z, lambda_, _ = getattr(self.iib, f'iba_{i}')(R, readout_feats, m_r=self.param_dict[i][0],
                                                          std_r=self.param_dict[i][1],
                                                          active_neurons=self.param_dict[i][2], )
            X_id_restrict += self.encoder.restrict_forward(Z, i)

            Rs, Rt = R[:B], R[B:]
            lambda_s, lambda_t = lambda_[:B], lambda_[B:]

            m_s = torch.mean(Rs, dim=0)  # [C, H, W]
            std_s = torch.mean(Rs, dim=0)
            Rs_params.append([m_s, std_s])

            eps_s = torch.randn(size=Rt.shape).to(Rt.device) * std_s + m_s
            feat_t = Rt * (1. - lambda_t) + lambda_t * eps_s

            Xt_feats.append(feat_t)  # only related with lambda
            Xt_lambda.append(lambda_t)

        X_id_restrict /= float(self.N)
        Xs_id = X_id_restrict[:B]
        Xt_feats[0] = target
        Xt_attr, Xt_attr_lamb = self.decoder(Xt_feats, lambs=Xt_lambda, use_lambda=True)

        Y = self.G(Xs_id, Xt_attr, Xt_attr_lamb)
        return Y

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]
        img = img_wm_device[1]

        img_size = img_wm.shape[-1]
        resize_back = transforms.Resize((img_size, img_size))
        img_source = torch.roll(img, 1, 0)

        swapped_face_wm = self.one_step_swap(img_source, img_wm)
        swapped_face_wm = resize_back(swapped_face_wm)

        return swapped_face_wm


class UniFaceModel(nn.Module):
    def __init__(self, device):
        super(UniFaceModel, self).__init__()
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--mixing_type",
            type=str,
            default='examples'
        )
        parser.add_argument("--inter", type=str, default='pair')
        parser.add_argument("--ckpt", type=str, default='session/swap/checkpoints/500000.pt')
        parser.add_argument("--test_path", type=str, default='examples/img/')
        parser.add_argument("--test_txt_path", type=str, default='examples/pair_swap.txt')
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--save_image_dir", type=str, default="expr")

        args = parser.parse_args()

        ckpt = torch.load('./drive/MyDrive/500000.pt')
        train_args = ckpt["train_args"]
        for key in vars(train_args):
            if not (key in vars(args)):
                setattr(args, key, getattr(train_args, key))
        self.swap_model = UniFace(args).half().to(device)
        self.swap_model.g_ema.load_state_dict(ckpt["g_ema"])
        self.swap_model.e_ema.load_state_dict(ckpt["e_ema"])
        self.swap_model.eval()

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]
        img = img_wm_device[1]

        img_size = img_wm.shape[-1]
        resize_back = transforms.Resize((img_size, img_size))
        img_source = torch.roll(img, 1, 0)

        _, _, swapped_img_wm = self.swap_model(
            [img_wm.type(torch.cuda.HalfTensor), img_source.type(torch.cuda.HalfTensor)]
        )
        swapped_face_wm = self.one_step_swap(img_source, img_wm)
        swapped_face_wm = resize_back(swapped_face_wm)
        return swapped_face_wm


class StyleMaskModel(nn.Module):
    def __init__(self, img_size, device, mode='test'):
        super(StyleMaskModel, self).__init__()
        self.img_size = img_size
        self.mode = mode
        self.device = device

        seed = 0
        random.seed(seed)

        self.masknet_path = './model/stylemask/pretrained_models/mask_network_1024.pt'
        self.image_resolution = 1024
        self.resize_image = True
        self.input_is_latent = True

        self.resize_image = True

        self.face_pool = torch.nn.AdaptiveAvgPool2d((self.img_size, self.img_size))
        self.generator_path = stylegan2_ffhq_1024['gan_weights']
        self.channel_multiplier = stylegan2_ffhq_1024['channel_multiplier']
        self.split_sections = stylegan2_ffhq_1024['split_sections']
        self.stylespace_dim = stylegan2_ffhq_1024['stylespace_dim']

        print('----- Load generator from {} -----'.format(self.generator_path))
        self.G = StyleGAN2Generator(self.image_resolution, 512, 8, channel_multiplier=self.channel_multiplier)
        self.G.load_state_dict(torch.load(self.generator_path)['g_ema'], strict=True)
        self.G.to(self.device).eval()
        # use truncation
        self.truncation = 0.7
        self.trunc = self.G.mean_latent(4096).detach().clone()

        print('----- Load mask network from {} -----'.format(self.masknet_path))
        ckpt = torch.load(self.masknet_path, map_location=torch.device('cpu'))
        self.num_layers_control = ckpt['num_layers_control']
        self.mask_net = nn.ModuleDict({})
        for layer_idx in range(self.num_layers_control):
            network_name_str = 'network_{:02d}'.format(layer_idx)

            # Net info
            stylespace_dim_layer = self.split_sections[layer_idx]
            input_dim = stylespace_dim_layer
            output_dim = stylespace_dim_layer
            inner_dim = stylespace_dim_layer

            network_module = MaskPredictor(input_dim, output_dim, inner_dim=inner_dim)
            self.mask_net.update({network_name_str: network_module})
        self.mask_net.load_state_dict(ckpt['mask_net'])
        self.mask_net.to(self.device).eval()

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')
        ### Load inversion model only when the input is image. ###
        self.encoder_path = stylegan2_ffhq_1024['e4e_inversion_model']
        print('----- Load e4e encoder from {} -----'.format(self.encoder_path))
        ckpt = torch.load(self.encoder_path, map_location='cpu')
        opts = ckpt['opts']
        opts['output_size'] = self.image_resolution
        opts['checkpoint_path'] = self.encoder_path
        opts['device'] = self.device
        opts['channel_multiplier'] = self.channel_multiplier
        # opts['dataset'] = self.dataset
        opts = Namespace(**opts)
        self.encoder = pSp(opts, self.device)
        self.encoder.to(self.device).eval()

    def reenact_pair(self, source_code, target_code):
        with torch.no_grad():
            # Get source style space
            source_img, style_source, w_source, noise_source = generate_image(self.G, source_code, self.truncation,
                                                                              self.trunc, self.image_resolution,
                                                                              self.split_sections,
                                                                              input_is_latent=self.input_is_latent,
                                                                              return_latents=True,
                                                                              resize_image=self.resize_image)

            # Get target style space
            target_img, style_target, w_target, noise_target = generate_image(self.G, target_code, self.truncation,
                                                                              self.trunc, self.image_resolution,
                                                                              self.split_sections,
                                                                              input_is_latent=self.input_is_latent,
                                                                              return_latents=True,
                                                                              resize_image=self.resize_image)
            # Get reenacted image
            masks_per_layer = []
            for layer_idx in range(self.num_layers_control):
                network_name_str = 'network_{:02d}'.format(layer_idx)
                style_source_idx = style_source[layer_idx]
                style_target_idx = style_target[layer_idx]
                styles = style_source_idx - style_target_idx
                mask_idx = self.mask_net[network_name_str](styles)
                masks_per_layer.append(mask_idx)

            mask = torch.cat(masks_per_layer, dim=1)
            style_source = torch.cat(style_source, dim=1)
            style_target = torch.cat(style_target, dim=1)

            new_style_space = generate_new_stylespace(style_source, style_target, mask,
                                                      num_layers_control=self.num_layers_control)
            new_style_space = list(
                torch.split(tensor=new_style_space, split_size_or_sections=self.split_sections, dim=1))
            reenacted_img = decoder(self.G, new_style_space, w_source, noise_source, resize_image=self.resize_image)

        return source_img, target_img, reenacted_img

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0].to(self.device)
        img = img_wm_device[1].to(self.device)

        img_source = torch.roll(img, 1, 0)

        self.trunc = self.trunc.to(img_wm.device)

        inv_image, source_code = invert_image(img_source, self.encoder, self.G, self.truncation, self.trunc)
        inv_image, target_code = invert_image(img_wm, self.encoder, self.G, self.truncation, self.trunc)
        source_img, target_img, reenacted_img_wm = self.reenact_pair(target_code, source_code)

        if self.mode != 'train':
            return reenacted_img_wm
        else:
            inv_image, target_code = invert_image(img, self.encoder, self.G, self.truncation, self.trunc)
            source_img, target_img, reenacted_img = self.reenact_pair(target_code, source_code)
            return reenacted_img_wm, reenacted_img


class StarGanModel(nn.Module):
    def __init__(self, img_size, mode='test'):
        super(StarGanModel, self).__init__()
        self.img_size = img_size
        self.mode = mode
        parser = argparse.ArgumentParser()
        self.args = self.get_args(parser)
        torch.manual_seed(self.args.seed)
        self.solver = Solver(self.args)
        self.star_gan = self.solver.get_model()
        self.resize_up = transforms.Resize((256, 256))
        self.resize_down = transforms.Resize((self.img_size, self.img_size))

    def get_args(self, parser):
        # model arguments
        parser.add_argument('--img_size', type=int, default=256,
                            help='Image resolution')
        parser.add_argument('--num_domains', type=int, default=2,
                            help='Number of domains')
        parser.add_argument('--latent_dim', type=int, default=16,
                            help='Latent vector dimension')
        parser.add_argument('--hidden_dim', type=int, default=512,
                            help='Hidden dimension of mapping network')
        parser.add_argument('--style_dim', type=int, default=64,
                            help='Style code dimension')

        # weight for objective functions
        parser.add_argument('--lambda_reg', type=float, default=1,
                            help='Weight for R1 regularization')
        parser.add_argument('--lambda_cyc', type=float, default=1,
                            help='Weight for cyclic consistency loss')
        parser.add_argument('--lambda_sty', type=float, default=1,
                            help='Weight for style reconstruction loss')
        parser.add_argument('--lambda_ds', type=float, default=1,
                            help='Weight for diversity sensitive loss')
        parser.add_argument('--ds_iter', type=int, default=100000,
                            help='Number of iterations to optimize diversity sensitive loss')
        parser.add_argument('--w_hpf', type=float, default=1,
                            help='weight for high-pass filtering')
        parser.add_argument('--resume_iter', type=int, default=100000,
                            help='Iterations to resume training/testing')
        # misc
        parser.add_argument('--mode', type=str, default='sample',
                            choices=['train', 'sample', 'eval', 'align'],
                            help='This argument is used in solver')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of workers used in DataLoader')
        parser.add_argument('--seed', type=int, default=777,
                            help='Seed for random number generator')
        parser.add_argument('--checkpoint_dir', type=str, default='./model/stargan/expr/checkpoints/celeba_hq',
                            help='Directory for saving network checkpoints')
        # face alignment
        parser.add_argument('--wing_path', type=str, default='./model/stargan/expr/checkpoints/wing.ckpt')
        parser.add_argument('--lm_path', type=str, default='./model/stargan/expr/checkpoints/celeba_lm_mean.npz')

        # step size
        parser.add_argument('--print_every', type=int, default=10)
        parser.add_argument('--sample_every', type=int, default=5000)
        parser.add_argument('--save_every', type=int, default=10000)
        parser.add_argument('--eval_every', type=int, default=50000)

        args = parser.parse_args()
        return args

    def reenactment(self, source, target, ref):
        masks = self.star_gan.fan.get_heatmap(source) if self.args.w_hpf > 0 else None
        s_ref = self.star_gan.style_encoder(target, ref)
        x_fake = self.star_gan.generator(source, s_ref, masks=masks)
        return x_fake

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]
        img = img_wm_device[1]

        img_wm = self.resize_up(img_wm)
        img = self.resize_up(img)

        img_source = torch.roll(img, 1, 0)

        N = img_wm.shape[0]
        ref = torch.randint(2, size=(N,), dtype=torch.int)
        reenact_img_wm = self.reenactment(img_source, img_wm, ref)
        if self.mode == 'train':
            reenact_img = self.reenactment(img_source, img, ref)
            return self.resize_down(reenact_img_wm), self.resize_down(reenact_img)
        else:
            return self.resize_down(reenact_img_wm)
