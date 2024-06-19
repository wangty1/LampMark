import torch
import torch.nn as nn
from torch.optim import Adam
import kornia
from torchvision import transforms

from utils import wm_error_rate
from model.encoder_decoder import Encoder, Decoder, LPMark
from model.discriminator import Discriminator
from model.common_manipulations import Manipulation
from model.deepfake_manipulations import SimSwapModel, StarGanModel


class TrainerImg:

    def __init__(self, configs, device):
        super(TrainerImg, self).__init__()
        self.configs = configs

        self.img_size = configs.img_size
        self.wm_len = configs.watermark_length
        self.enc_c = configs.encoder_channels
        self.enc_blocks = configs.encoder_blocks
        self.dec_c = configs.decoder_channels
        self.dec_blocks = configs.decoder_blocks
        self.dis_c = configs.discriminator_channels
        self.dis_blocks = configs.discriminator_blocks

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs
        self.lr = configs.lr
        self.device = device

        if self.configs.sep_model:
            print('Initializing encoder and decoder separately.')
            self.encoder = Encoder(self.img_size, self.enc_c, self.enc_blocks, self.wm_len).to(self.device)
            self.decoder = Decoder(self.img_size, self.dec_c, self.dec_blocks, self.wm_len).to(self.device)
        else:
            print('Initializing encoder and decoder together.')
            self.model = LPMark(
                self.img_size, self.enc_c, self.enc_blocks, self.dec_c, self.dec_blocks, self.wm_len, self.device,
                self.configs.manipulation_layers).to(self.device)
        self.discriminator = Discriminator(self.dis_c, self.dis_blocks).to(self.device)

        if self.configs.manipulation_mode == 'common':
            if self.configs.sep_model:
                self._init_common_manipulation()
        elif self.configs.manipulation_mode == 'deepfake':
            self._init_deepfake_manipulations()
            if configs.do_sim_swap:
                self.sim_swap = SimSwapModel(self.img_size, mode='train').to(self.device)
        else:
            raise Exception('Manipulation mode must be one of "common" and "deepfake".')

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1 and self.device != 'cpu':  # For multi-gpu.
            print("Using", torch.cuda.device_count(), "GPUs ...")
            if self.configs.sep_model:
                self.encoder = nn.DataParallel(self.encoder)
                self.decoder = nn.DataParallel(self.decoder)
            else:
                self.model = nn.DataParallel(self.model)
            self.discriminator = nn.DataParallel(self.discriminator)
            if self.configs.manipulation_mode == 'common':
                if self.configs.sep_model:
                    self._common_manipulation_multi_gpu()
            elif self.configs.manipulation_mode == 'deepfake':
                self._deepfake_manipulation_multi_gpu()
            else:
                raise Exception('Manipulation mode must be one of "common" and "deepfake".')

        # labels: raw image -> 1; watermarked (wmd) image -> 0.
        self.labels_raw = torch.full((self.batch_size, 1), 1, dtype=torch.float, device=self.device)
        self.labels_wmd = torch.full((self.batch_size, 1), 0, dtype=torch.float, device=self.device)

        if self.configs.sep_model:
            self.opt_model = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
            self.opt_encoder = Adam(self.encoder.parameters(), lr=self.lr)
            self.opt_decoder = Adam(self.decoder.parameters(), lr=self.lr)
        else:
            self.opt_model = Adam(self.model.parameters(), lr=self.lr)
        self.opt_discriminator = Adam(self.discriminator.parameters(), lr=self.lr)

        # loss functions
        self.criterion_BCE = nn.BCEWithLogitsLoss().to(self.device)
        self.criterion_MSE = nn.MSELoss().to(self.device)
        self.criterion_SmoothMSE = nn.SmoothL1Loss().to(self.device)

        # weights for loss values
        self.enc_w = configs.encoder_weight
        self.dec_w = configs.decoder_weight
        self.dis_w = configs.discriminator_weight

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
        self.gen_w = self.configs.generative_weight

    def _common_manipulation_multi_gpu(self):
        self.common_manipulation = nn.DataParallel(self.common_manipulation)

    def _deepfake_manipulation_multi_gpu(self):
        self.sim_swap = nn.DataParallel(self.sim_swap)

    def train_batch_common(self, imgs, wms):
        self.model.train()
        self.discriminator.train()

        with torch.enable_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm, manipulated_wm_img, wms_recover = self.model(imgs, wms)

            # Train the discriminator.
            self.opt_discriminator.zero_grad()
            # Detect watermark from original img.
            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_raw_loss.backward()
            # Detect watermark from watermarked img.
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])
            d_wmd_loss.backward()
            # Update weights.
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)
            self.opt_discriminator.step()

            # train ID-Mark Encoder-Decoder
            self.opt_model.zero_grad()
            # Make sure the watermark is un-detectable.
            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            # Visual similarity: L2 norm between watermarked and original images.
            # g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            # Watermark similarity: L2 norm between watermark and the recovered.
            g_loss_dec = self.criterion_MSE(wms_recover, wms)
            g_loss_total = self.enc_w * g_loss_enc + self.dec_w * g_loss_dec + self.dis_w * g_loss_dis
            g_loss_total.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.opt_model.step()

            # PSNR.
            psnr = kornia.losses.psnr_loss(self.denorm_imgnet(imgs_wm.detach()), self.denorm_imgnet(imgs.detach()),
                                           2.) * (-1)
            # SSIM.
            ssim = 1 - 2 * kornia.losses.ssim_loss(self.denorm_imgnet(imgs_wm.detach()),
                                                   self.denorm_imgnet(imgs.detach()), window_size=5, reduction='mean')
            # Error rate.
            error_rate = wm_error_rate(wms, wms_recover)

            result = {
                'error_rate': error_rate,
                'psnr': psnr,
                'ssim': ssim,
                'g_loss': g_loss_total,
                'g_loss_enc': g_loss_enc,
                'g_loss_dec': g_loss_dec,
                'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss,
                'd_loss_wmd': d_wmd_loss
            }
            return result

    def val_batch_common(self, imgs, wms):
        self.model.eval()
        self.discriminator.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)

            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm, manipulated_wm_img, wms_recover = self.model(imgs, wms)

            # Compute loss on discriminator.
            # Detect watermark from original img.
            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            # Detect watermark from watermarked img.
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])

            # Make sure the watermark is un-detectable.
            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            # Visual similarity: L2 norm between watermarked and original images.
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            # Watermark similarity: L2 norm between watermark and the recovered.
            g_loss_dec = self.criterion_MSE(wms_recover, wms)
            g_loss_total = self.enc_w * g_loss_enc + self.dec_w * g_loss_dec + self.dis_w * g_loss_dis

            # PSNR.
            psnr = kornia.losses.psnr_loss(self.denorm_imgnet(imgs_wm.detach()), self.denorm_imgnet(imgs.detach()),
                                           2.) * (-1)
            # SSIM.
            ssim = 1 - 2 * kornia.losses.ssim_loss(self.denorm_imgnet(imgs_wm.detach()),
                                                   self.denorm_imgnet(imgs.detach()), window_size=5, reduction='mean')
            # Error rate.
            error_rate = wm_error_rate(wms, wms_recover)

            result = {
                'error_rate': error_rate,
                'psnr': psnr,
                'ssim': ssim,
                'g_loss': g_loss_total,
                'g_loss_enc': g_loss_enc,
                'g_loss_dec': g_loss_dec,
                'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss,
                'd_loss_wmd': d_wmd_loss
            }
            return result, [imgs, imgs_wm, manipulated_wm_img]

    def tune_batch_simswap(self, imgs, wms):
        self.encoder.train()
        self.decoder.train()
        self.sim_swap.eval()
        self.discriminator.train()

        with torch.enable_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)

            imgs_wm = self.encoder(imgs, wms)  # Encode watermark into img.
            imgs_denorm, imgs_wm_denorm = self.denorm_imgnet(imgs), self.denorm_imgnet(imgs_wm)
            swapped_img_wm, swapped_img = self.sim_swap([imgs_wm_denorm, imgs_denorm, self.device])
            wms_recover = self.decoder(self.norm_imgnet(swapped_img_wm))

            # Train the discriminator.
            self.opt_discriminator.zero_grad()
            # Detect watermark from original img.
            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_raw_loss.backward()
            # Detect watermark from watermarked img.
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])
            d_wmd_loss.backward()
            # Update weights.
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)
            self.opt_discriminator.step()

            # train ID-Mark Encoder-Decoder
            self.opt_model.zero_grad()
            # Make sure the watermark is un-detectable.
            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            # Visual similarity: L2 norm between watermarked and original images.
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            # Watermark similarity: L2 norm between watermark and the recovered.
            g_loss_dec = self.criterion_MSE(wms_recover, wms)
            g_loss_total = self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + self.dec_w * g_loss_dec
            # Face-swap quality on watermarked target image.
            g_loss_gen = self.criterion_MSE(swapped_img_wm, swapped_img)
            g_loss_total += self.gen_w * g_loss_gen
            # This is specially designed for Identity() watermark error_rate to approach 0. Otherwise, we are afraid
            # of dragging the weights towards Deepfake models too much.
            wms_identity = self.decoder(imgs_wm)
            g_loss_id = self.criterion_MSE(wms_identity, wms)
            g_loss_total += g_loss_id * self.dec_w

            g_loss_total.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
            self.opt_model.step()

            # PSNR.
            psnr = kornia.losses.psnr_loss(self.denorm_imgnet(imgs_wm.detach()), self.denorm_imgnet(imgs.detach()),
                                           2.) * (-1)
            # SSIM.
            ssim = 1 - 2 * kornia.losses.ssim_loss(self.denorm_imgnet(imgs_wm.detach()),
                                                   self.denorm_imgnet(imgs.detach()), window_size=5, reduction='mean')
            # Error rate.
            error_rate = wm_error_rate(wms, wms_recover)

            result = {
                'error_rate': error_rate,
                'psnr': psnr,
                'ssim': ssim,
                'g_loss': g_loss_total,
                'g_loss_enc': g_loss_enc,
                'g_loss_dec': g_loss_dec,
                'g_loss_gen': g_loss_gen,
                'g_loss_id': g_loss_id,
                'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss,
                'd_loss_wmd': d_wmd_loss
            }
            return result

    def val_batch_simswap(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        self.sim_swap.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)

            imgs_wm = self.encoder(imgs, wms)  # Encode watermark into img.
            imgs_denorm, imgs_wm_denorm = self.denorm_imgnet(imgs), self.denorm_imgnet(imgs_wm)
            swapped_img_wm, swapped_img = self.sim_swap([imgs_wm_denorm, imgs_denorm, self.device])
            wms_recover = self.decoder(self.norm_imgnet(swapped_img_wm))

            # Compute loss on discriminator.
            # Detect watermark from original img.
            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            # Detect watermark from watermarked img.
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])

            # Make sure the watermark is un-detectable.
            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            # Visual similarity: L2 norm between watermarked and original images.
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            # Watermark similarity: L2 norm between watermark and the recovered.
            g_loss_dec = self.criterion_MSE(wms_recover, wms)
            g_loss_total = self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + self.dec_w * g_loss_dec
            # Face-swap quality on watermarked target image.
            g_loss_gen = self.criterion_MSE(swapped_img_wm, swapped_img)
            g_loss_total += self.gen_w * g_loss_gen
            # This is specially designed for Identity() watermark error_rate to approach 0. Otherwise, we are afraid
            # of dragging the weights towards Deepfake models too much.
            wms_identity = self.decoder(imgs_wm)
            g_loss_id = self.criterion_MSE(wms_identity, wms)
            g_loss_total += g_loss_id * self.dec_w

            # PSNR.
            psnr = kornia.losses.psnr_loss(self.denorm_imgnet(imgs_wm.detach()), self.denorm_imgnet(imgs.detach()),
                                           2.) * (-1)
            # SSIM.
            ssim = 1 - 2 * kornia.losses.ssim_loss(self.denorm_imgnet(imgs_wm.detach()),
                                                   self.denorm_imgnet(imgs.detach()), window_size=5, reduction='mean')
            # Error rate.
            error_rate = wm_error_rate(wms, wms_recover)

            result = {
                'error_rate': error_rate,
                'psnr': psnr,
                'ssim': ssim,
                'g_loss': g_loss_total,
                'g_loss_enc': g_loss_enc,
                'g_loss_dec': g_loss_dec,
                'g_loss_gen': g_loss_gen,
                'g_loss_id': g_loss_id,
                'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss,
                'd_loss_wmd': d_wmd_loss
            }
            return result, [imgs, imgs_wm, swapped_img_wm]

    def save_model(self, encoder_path, decoder_path, discriminator_path):
        if self.num_gpus > 1:
            torch.save(self.encoder.module.state_dict(), encoder_path)
            torch.save(self.decoder.module.state_dict(), decoder_path)
            torch.save(self.discriminator.module.state_dict(), discriminator_path)
        else:
            torch.save(self.encoder.state_dict(), encoder_path)
            torch.save(self.decoder.state_dict(), decoder_path)
            torch.save(self.discriminator.state_dict(), discriminator_path)

    def save_model_integral(self, model_path, discriminator_path):
        if self.num_gpus > 1:
            torch.save(self.model.module.state_dict(), model_path)
            torch.save(self.discriminator.module.state_dict(), discriminator_path)
        else:
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.discriminator.state_dict(), discriminator_path)

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

    def load_discriminator(self, discriminator_path):
        if self.num_gpus > 1 and self.device != 'cpu':
            self.discriminator.module.load_state_dict(torch.load(discriminator_path))
        else:
            self.discriminator.load_state_dict(torch.load(discriminator_path))
