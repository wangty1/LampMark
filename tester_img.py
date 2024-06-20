import torch
import torch.nn as nn
import kornia
from torchvision import transforms

from utils import wm_error_rate
from model.encoder_decoder import Encoder, Decoder
from model.common_manipulations import Manipulation
from model.deepfake_manipulations import SimSwapModel
from model.deepfake_manipulations import InfoSwapModel, UniFaceModel
from model.deepfake_manipulations import StarGanModel, StyleMaskModel


class TesterImg:

    def __init__(self, configs, device):
        super(TesterImg, self).__init__()
        self.configs = configs

        self.img_size = configs.img_size
        self.wm_len = configs.watermark_length
        self.enc_c = configs.encoder_channels
        self.enc_blocks = configs.encoder_blocks
        self.dec_c = configs.decoder_channels
        self.dec_blocks = configs.decoder_blocks

        self.batch_size = configs.batch_size
        self.device = device

        self.encoder = Encoder(self.img_size, self.enc_c, self.enc_blocks, self.wm_len).to(self.device)
        self.decoder = Decoder(self.img_size, self.dec_c, self.dec_blocks, self.wm_len).to(self.device)

        if self.configs.manipulation_mode == 'common':
            self._init_common_manipulation()
        elif self.configs.manipulation_mode == 'deepfake':
            self._init_deepfake_manipulations()
            if configs.do_sim_swap:
                self.sim_swap = SimSwapModel(self.img_size, mode='test').to(self.device)
            if configs.do_info_swap:
                self.info_swap = InfoSwapModel(self.device).to(self.device)
            if configs.do_uni_face:
                self.uni_face = UniFaceModel(self.device).to(self.device)
            if configs.do_style_mask:
                self.style_mask = StyleMaskModel(self.img_size, self.device, mode='test').to(self.device)
            if configs.do_star_gan:
                self.star_gan = StarGanModel(self.img_size, mode='test').to(self.device)
        else:
            raise Exception('Manipulation mode must be one of "common" and "deepfake".')

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1 and self.device != 'cpu':  # For multi-gpu.
            print("Using", torch.cuda.device_count(), "GPUs ...")
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            if self.configs.manipulation_mode == 'common':
                self._common_manipulation_multi_gpu()
            elif self.configs.manipulation_mode == 'deepfake':
                self._deepfake_manipulation_multi_gpu()
            else:
                self.common_manipulation = nn.DataParallel(self.common_manipulation)
                pass

        self.norm_imgnet = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.denorm_imgnet = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        self.norm = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        self.denorm = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )

    def _init_common_manipulation(self):
        self.common_manipulation = Manipulation(self.configs.manipulation_layers).to(self.device)

    def _init_deepfake_manipulations(self):
        self.sim_swap = None
        self.info_swap = None
        self.uni_face = None
        self.e4s = None
        self.star_gan = None
        self.style_mask = None
        self.hyper_reenact = None

    def _common_manipulation_multi_gpu(self):
        self.common_manipulation = nn.DataParallel(self.common_manipulation)

    def _deepfake_manipulation_multi_gpu(self):
        self.sim_swap = nn.DataParallel(self.sim_swap)
        self.info_swap = nn.DataParallel(self.info_swap)
        self.uni_face = nn.DataParallel(self.uni_face)
        self.e4s = nn.DataParallel(self.e4s)
        self.star_gan = nn.DataParallel(self.star_gan)
        self.style_mask = nn.DataParallel(self.style_mask)
        self.hyper_reenact = nn.DataParallel(self.hyper_reenact)

    def test_batch_simswap(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        self.sim_swap.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)
            imgs_denorm, imgs_wm_denorm = self.denorm_imgnet(imgs), self.denorm_imgnet(imgs_wm)
            swapped_img_wm = self.sim_swap([imgs_wm_denorm, imgs_denorm, self.device])
            wms_recover = self.decoder(self.norm_imgnet(swapped_img_wm))
            # Error rate.
            error_rate = wm_error_rate(wms, wms_recover)
            return error_rate

    def test_batch_infoswap(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        self.info_swap.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)
            imgs_denorm, imgs_wm_denorm = self.denorm_imgnet(imgs), self.denorm_imgnet(imgs_wm)
            swapped_img_wm = self.info_swap([imgs_wm_denorm, imgs_denorm, self.device])
            wms_recover = self.decoder(swapped_img_wm)
            # Error rate.
            error_rate = wm_error_rate(wms, wms_recover)
            return error_rate

    def test_batch_uniface(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        self.uni_face.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)
            imgs_denorm, imgs_wm_denorm = imgs, imgs_wm
            swapped_img_wm = self.uni_face([imgs_wm_denorm.half(), imgs_denorm.half()])
            wms_recover = self.decoder(swapped_img_wm.type(torch.cuda.FloatTensor))
            # Error rate.
            error_rate = wm_error_rate(wms, wms_recover)
            return error_rate

    def test_batch_stylemask(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        self.style_mask.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)
            imgs_denorm, imgs_wm_denorm = self.norm(self.denorm_imgnet(imgs)), self.norm(self.denorm_imgnet(imgs_wm))
            reenacted_img_wm = self.style_mask([imgs_wm_denorm, imgs_denorm, self.device])
            wms_recover = self.decoder(reenacted_img_wm)
            # Error rate.
            error_rate = wm_error_rate(wms, wms_recover)
            return error_rate

    def test_batch_stargan(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)
            imgs_denorm, imgs_wm_denorm = self.denorm_imgnet(imgs), self.denorm_imgnet(imgs_wm)
            reenacted_img_wm = self.star_gan([imgs_wm_denorm, imgs_denorm, self.device])
            wms_recover = self.decoder(self.norm_imgnet(reenacted_img_wm))
            # Error rate.
            error_rate = wm_error_rate(wms, wms_recover)
            return error_rate

    def test_one_manipulation(self, imgs, wms, manipulation):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)

            self.common_manipulation = Manipulation([manipulation])

            manipulated_wm_img = self.common_manipulation([imgs_wm, imgs, self.device])
            wms_recover = self.decoder(manipulated_wm_img)

            # Error rate.
            error_rate = wm_error_rate(wms, wms_recover)

            if manipulation == 'Identity()':
                # PSNR.
                psnr = kornia.losses.psnr_loss(self.denorm_imgnet(imgs_wm.detach()), self.denorm_imgnet(imgs.detach()),
                                               2.) * (-1)
                # SSIM.
                ssim = 1 - 2 * kornia.losses.ssim_loss(
                    self.denorm_imgnet(imgs_wm.detach()), self.denorm_imgnet(imgs.detach()),
                    window_size=5, reduction='mean'
                )
                # Error rate.
                error_rate = wm_error_rate(wms, wms_recover)
                return psnr, ssim, error_rate
            else:
                return error_rate

    def load_model(self, encoder_path, decoder_path):
        if self.num_gpus > 1 and self.device != 'cpu':
            self.encoder.module.load_state_dict(torch.load(encoder_path))
            self.decoder.module.load_state_dict(torch.load(decoder_path))
        else:
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.decoder.load_state_dict(torch.load(decoder_path))
        print('Finished loading weights from', encoder_path, 'and', decoder_path)

    def load_model_integral(self, model_path):
        """ This function is used when the model is trained as a whole but needs to be tuned separately for
            encoder and decoder.
        """
        model_dict = torch.load(model_path)
        enc_dict = {k.replace('encoder.', ''): v for k, v in model_dict.items() if k.startswith('encoder.')}
        dec_dict = {k.replace('decoder.', ''): v for k, v in model_dict.items() if k.startswith('decoder.')}
        if self.num_gpus > 1 and self.device != 'cpu':
            self.encoder.module.load_state_dict(enc_dict)
            self.decoder.module.load_state_dict(dec_dict)
        else:
            self.encoder.load_state_dict(enc_dict)
            self.decoder.load_state_dict(dec_dict)
        print('Finished loading weights from', model_path)
