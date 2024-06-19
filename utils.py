import os
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pyecharts.charts import Line
from pyecharts import options as opts
import datetime
import numpy as np
import random
from PIL import Image

from scipy.spatial.distance import squareform
from torch import linalg as LA


class JsonConfig:
    """ Load config files from json. """

    def __init__(self):
        self.__json__ = None

    def load_json_file(self, path):
        with open(path, 'r') as file:
            self.__json__ = json.load(file)
            file.close()

        self.set_items()

    def load_json(self, json_dict):
        self.__json__ = json_dict

        self.set_items()

    def set_items(self):
        for key in self.__json__:
            self.__setattr__(key, self.__json__[key])

    def get_items(self):
        items = []
        for key in self.__json__:
            items.append((key, self.__json__[key]))
        return items


# Utils for datasets.

class LandmarkDataset(Dataset):
    """ Self-implemented Dataset class to load .npy files for landmark dataset only. """

    def __init__(self, path_landmark, img_size, triplet=True):
        super(LandmarkDataset, self).__init__()
        self.path_landmark = path_landmark
        self.img_size = img_size
        self.lst = os.listdir(self.path_landmark)
        self.triplet = triplet  # The flag to indicate triplet training strategy.
        if True:  # self.triplet:
            self.common_lst = ['jpeg', 'gnoise', 'gblur', 'mblur']
            self.deepfake_lst = ['simswap', 'infoswap', 'stargan', 'stylemask']
            self.num_common = len(self.common_lst)
            self.num_deepfake = len(self.deepfake_lst)

    def __getitem__(self, index):
        while True:
            fname = self.lst[index]
            if fname.startswith('.'):
                continue
            raw = np.load(os.path.join(self.path_landmark, fname))
            if raw is None:
                continue
            if not self.triplet:
                return fname, raw  # Returns a single landmark.
            else:
                common_new_path = self.path_landmark.replace('raw',
                                                             self.common_lst[random.randint(0, self.num_common - 1)])
                if not os.path.exists(os.path.join(common_new_path, fname)):
                    count_c = 4
                    random.shuffle(self.common_lst)
                    for item in self.common_lst:
                        common_new_path = self.path_landmark.replace('raw', item)
                        if os.path.exists(os.path.join(common_new_path, fname)):
                            break
                        count_c -= 1
                    if count_c == 0:
                        index += 1
                        continue
                common = np.load(os.path.join(common_new_path, fname))
                deepfake_new_path = self.path_landmark.replace('raw', self.deepfake_lst[
                    random.randint(0, self.num_deepfake - 1)])
                if not os.path.exists(os.path.join(deepfake_new_path, fname)):
                    count_df = 4
                    random.shuffle(self.deepfake_lst)
                    for item in self.deepfake_lst:
                        deepfake_new_path = self.path_landmark.replace('raw', item)
                        if os.path.exists(os.path.join(deepfake_new_path, fname)):
                            break
                        count_df -= 1
                    if count_df == 0:
                        index += 1
                        continue
                deepfake = np.load(os.path.join(deepfake_new_path, fname))
                return fname, [raw, common, deepfake]  # Returns [anchor, positive, negative].
            index += 1

    def __len__(self):
        return len(self.lst)


class ImageDataset(Dataset):
    def __init__(self, path_img, path_wm, img_size, wm_len, mode='train'):
        super(ImageDataset, self).__init__()
        self.img_size = img_size
        self.wm_len = wm_len
        self.path_img = path_img
        self.path_wm = path_wm
        self.lst_wm = os.listdir(self.path_wm)
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((int(self.img_size * 1.1), int(self.img_size * 1.1))),
                transforms.RandomCrop((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])

    def transform_image(self, img):
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        while True:
            wm_name = self.lst_wm[idx]
            img_name = wm_name.replace('npy', 'jpg')
            img = Image.open(os.path.join(self.path_img, img_name)).convert('RGB')
            wm = np.load(os.path.join(self.path_wm, wm_name))
            if img is not None and wm is not None:
                img = self.transform_image(img)
                return img, torch.Tensor(wm)
            print('Somehow skipped the image:', os.path.join(self.path_wm, self.lst_wm[idx]))
            idx += 1

    def __len__(self):
        return len(self.lst_wm)


# Utils for model training and inference.

class ContrastiveLoss(nn.Module):
    """ Contrastive loss for a pair of features <x1> and <x2> given label <y> inferring
        similar (y=1) or dissimilar (y=0). Also pass in dynamic epsilon values when calling
        the loss at each batch regarding the similarity between original inputs.
        Distances are computed using Euclidean.

        <dist> lies in [0, wm_len] and <margin> implies a fixed max gap between negative pairs.
        When passing in a meaningful <margin_dy> value, the loss is computed depending on dynamic margin values.
    """

    def __init__(self, margin=-1, wm_len=128):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.wm_len = wm_len

    def forward(self, x1, x2, y, dynamic=False, margin_dy=None):
        # x1 = torch.round(x1)
        # x2 = torch.round(x2)
        x1 = torch.flatten(F.normalize(x1), start_dim=1)
        x2 = torch.flatten(F.normalize(x2), start_dim=1)
        dist = torch.norm(x1 - x2, p=1, dim=1)
        if self.margin == -1 and not dynamic:
            raise Exception('Need to choose a mode from <margin> and <margin_dy>.')
        elif not dynamic:  # Use fixed margin.
            pos_part = y * dist
            neg_part = (1 - y) * torch.relu(self.margin - dist)
        else:
            # pos_param = epsilon / (epsilon + torch.exp(-1 * dist))
            # neg_param = 2 / (1 + torch.exp(-1 * dist)) - 1
            # neg_margin = 2 / (1 + torch.exp(-1 * margin_dy)) - 1
            # param = torch.exp(margin_dy)
            # margin_dy = torch.clamp(margin_dy * self.wm_len, max=0.4 * self.wm_len)
            pos_part = y * dist
            neg_part = (1 - y) * torch.relu(margin_dy - dist)

        loss = torch.mean(neg_part + pos_part)
        return loss


class TripletLoss(nn.Module):
    """ Triplet loss for a tuple of (anchor, pos, neg).

        When <margin> != -1, it implies a fixed max gap between negative pairs. Otherwise, pass in dynamic
        margin values <margin_dy> from outside regarding the original landmarks and the triplet loss is weighted.
        Features are normalized to [-1, 1] so that easier to control.
    """

    def __init__(self, wm_len, margin=-1):
        super(TripletLoss, self).__init__()
        self.wm_len = wm_len
        self.margin = margin
        self.flat = nn.Flatten(start_dim=1)  # Flatten dimensions other than batch size, for norm computation.

    def forward(self, anchor, pos, neg, dynamic=False, margin_dy=None, epsilon=1e-10):
        if self.margin == -1 and not dynamic:
            raise Exception('Need to choose a mode from <margin> and <margin_dy>.')
        elif not dynamic:
            # To locate values between [0, 1]
            anchor = F.normalize(self.flat(anchor), dim=1) * 0.5 + 0.5
            pos = F.normalize(self.flat(pos), dim=1) * 0.5 + 0.5
            neg = F.normalize(self.flat(neg), dim=1) * 0.5 + 0.5
            loss = torch.norm(pos - anchor, p=2, dim=1) ** 2 + torch.relu(self.margin - torch.norm(
                neg - anchor, p=2, dim=1) ** 2)
        else:
            if margin_dy is None:
                raise Exception('Need to pass in the dynamic margin between anchor and neg landmarks.')
            anchor = F.normalize(self.flat(anchor), dim=1) * 0.5 + 0.5
            pos = F.normalize(self.flat(pos), dim=1) * 0.5 + 0.5
            neg = F.normalize(self.flat(neg), dim=1) * 0.5 + 0.5
            dist_neg = LA.matrix_norm(pos - anchor) ** 2
            dist_pos = LA.matrix_norm(neg - anchor) ** 2
            margin_pos = LA.matrix_norm(margin_dy[0]) ** 2
            margin_neg = LA.matrix_norm(margin_dy[1]) ** 2
            # print(torch.mean(margin_pos), torch.mean(margin_neg), torch.mean(dist_pos), torch.mean(dist_neg))
            loss = torch.relu(margin_pos - dist_pos) + torch.relu(margin_neg - dist_neg)

            # pos_dist = torch.norm(self.flat(pos - anchor), dim=1, p=1)
            # neg_dist = torch.norm(self.flat(neg - anchor), dim=1, p=1)
            # param = epsilon / (epsilon + torch.exp(-1 * margin_dy))
            # margin_dy = torch.clamp(margin_dy / 800.0, max=0.7) * self.wm_len
            # loss = pos_dist + torch.clamp(margin_dy - neg_dist, min=0)
        loss = torch.mean(loss)
        return loss


class DistanceInfoNCE(nn.Module):

    def __init__(self, wm_len, tau, eps=1e-9):
        super(DistanceInfoNCE, self).__init__()
        self.wm_len = wm_len
        self.tau = tau
        self.eps = eps
        self.flat = nn.Flatten(start_dim=1)

    def info_nce_idx(self, neg_lst, anchor, pos_dist, idx_dist, idx):
        anchor_lst = anchor.unsqueeze(0).expand(neg_lst.shape[0], -1)  # [b, wm_len]
        neg_dist = LA.norm(anchor_lst - neg_lst, dim=1) ** 2  # [b]
        neg_dist[idx] = idx_dist
        pos_sim = torch.exp(-pos_dist / self.tau)
        neg_sim = torch.exp(-neg_dist.sum() / self.tau)
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + self.eps))
        return loss

    def forward(self, anchor, pos, neg):
        pos_dist = LA.norm(anchor - pos, dim=1) ** 2
        neg_dist = LA.norm(anchor - neg, dim=1) ** 2
        loss = 0.0
        for idx in range(anchor.shape[0]):
            running_neg = torch.clone(anchor)
            # The positive pair: anchor and pos at the current idx.
            # The negative list: neg at the current idx and anchor at all other idx.
            loss += self.info_nce_idx(running_neg, anchor[idx], pos_dist[idx], neg_dist[idx], idx)
        return loss / anchor.shape[0]


def intra_distance(landmark_tensor):
    dist_lst = []
    for landmark in landmark_tensor:
        t_dist = squareform(F.pdist(landmark, p=2).numpy())  # [106, 106]
        dist_lst.append(t_dist)
    dist_lst = np.array(dist_lst)  # [b, 106, 106]

    return torch.tensor(dist_lst)


def weighted_triplet_loss(anchor, pos, neg, sim_pos, sim_neg, epsilon):
    """ The anchor, positive, and negative output are passed in. Similarity for the anchor in regard of
        positive and negative, respectively, are also passed in to provide extra weight for scaling the distribution.
        Based on sim_pos and sim_neg, we assign weight to the squared l2-norm.
    """
    anchor = torch.round(anchor)
    pos = torch.round(pos)
    neg = torch.round(neg)

    pos_param = 1  # math.exp(1 - torch.mean(sim_pos))
    neg_param = 1  # math.exp(torch.mean(sim_neg))
    loss_val = pos_param * torch.norm(anchor - pos, p=2) ** 2 - neg_param * torch.norm(
        anchor - neg, p=2) ** 2 + epsilon  # * math.exp(1.0)
    # LA.matrix_norm
    loss_val = torch.relu(loss_val)
    return loss_val


# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
def gaussian(window_size, sigma):
    # print(window_size)
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    _ssim_quotient = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    _ssim_divident = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = _ssim_quotient / _ssim_divident

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        # img1=(img1-img1.min())/(img1.max()-img1.min())
        # img2 = (img2-img2.min())/(img2.max()-img2.min())
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def make_loader(configs, data_mode='lm', model_mode='train', shuffle=True):
    """ Construct dataloader.

        data_mode: lm | img | lm-img | wm-img
        model_mode: train | val | test
    """

    if data_mode == 'lm':
        if model_mode == 'all':
            dataset_train = LandmarkDataset(os.path.join(configs.landmark_path, 'raw', 'train'), configs.img_size,
                                            configs.triplet)
            dataset_val = LandmarkDataset(os.path.join(configs.landmark_path, 'raw', 'val'), configs.img_size,
                                          configs.triplet)
            dataset_test = LandmarkDataset(os.path.join(configs.landmark_path, 'raw', 'test'), configs.img_size,
                                           configs.triplet)
        else:
            dataset = LandmarkDataset(os.path.join(configs.landmark_path, 'raw', model_mode), configs.img_size,
                                      configs.triplet)
    elif data_mode == 'img':
        pass
    elif data_mode == 'lm-img':
        pass
    elif data_mode == 'wm-img':
        dataset = ImageDataset(
            os.path.join(configs.img_path, model_mode),
            os.path.join(configs.wm_path, str(configs.img_size), model_mode),
            configs.img_size,
            configs.watermark_length,
            mode=model_mode
        )
    else:
        # raise error
        pass
    if model_mode == 'all':
        loader = DataLoader(
            torch.utils.data.ConcatDataset([dataset_train, dataset_val, dataset_test]),
            batch_size=configs.batch_size,
            shuffle=shuffle
        )
    else:
        loader = DataLoader(dataset, batch_size=configs.batch_size, num_workers=12, shuffle=shuffle, drop_last=True)
    return loader


def plot_curve(epochs, train_loss_rec, val_loss_rec, path, c_type):
    epoch_lst = list(range(0, epochs))
    line = (
        Line().add_xaxis(epoch_lst).add_yaxis('train_loss', train_loss_rec,
                                              markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_=c_type)],
                                                                                symbol_size=70)).add_yaxis(
            'validation_loss', val_loss_rec,
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='min')], symbol_size=70)
        ).set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    line.render(path)


def plot_curves(epochs, train_loss_dict, val_loss_dict, path):
    for key in train_loss_dict:
        c_type = 'min'
        if key == 'psnr' or key == 'ssim':
            c_type = 'max'
        plot_curve(epochs, train_loss_dict[key], val_loss_dict[key], os.path.join(path, key + '.html'), c_type)


def write_log(log_text, batch_count, running_result, loss_record):
    for key in running_result:
        log_text = log_text + key + '=' + str(running_result[key] / batch_count) + '\n'
        loss_record[key].append(float(running_result[key] / batch_count))
    log_text = log_text + '\n'

    if not os.path.exists('./results'):
        os.mkdir('./results')
    with open('./results/train_log.txt', 'a') as file:
        file.write(log_text)
    print(log_text)


def format_time(elapsed):
    """ Format the time from seconds to a string hh:mm:ss.

        Parameters
        ----------
        elapsed: int
            An int represents num of seconds.

        Returns
        -------
        A string hh:mm:ss.
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def decoded_message_error_rate(msgs, wms_recover):
    """ Bit Error Rate (BER) """
    length = msgs.shape[0]

    msgs = msgs.gt(0.5)
    wms_recover = wms_recover.gt(0.5)
    error_rate = float(sum(msgs != wms_recover)) / length

    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return error_rate


def wm_error_rate(wms, wms_recover):
    """ Bit Error Rate (BER) """
    length = wms.shape[0] * wms.shape[1]

    msgs = wms.gt(0.5)
    wms_recover = wms_recover.gt(0.5)
    error_rate = float(sum(sum(msgs != wms_recover))) / length

    return error_rate


def get_random_images(imgs, imgs_wm, manipulated_imgs_wm):
    selected_id = np.random.randint(1, imgs.shape[0]) if imgs.shape[0] > 1 else 1
    img = imgs.cpu()[selected_id - 1:selected_id, :, :, :]
    img_wm = imgs_wm.cpu()[selected_id - 1:selected_id, :, :, :]
    manipulated_img_wm = manipulated_imgs_wm.cpu()[selected_id - 1:selected_id, :, :, :]
    return [img, img_wm, manipulated_img_wm]


def concatenate_images(save_imgs, imgs, imgs_wm, manipulated_imgs_wm):
    saved = get_random_images(imgs, imgs_wm, manipulated_imgs_wm)
    if save_imgs[2].shape[2] != saved[2].shape[2]:
        return save_imgs
    save_imgs[0] = torch.cat((save_imgs[0], saved[0]), 0)
    save_imgs[1] = torch.cat((save_imgs[1], saved[1]), 0)
    save_imgs[2] = torch.cat((save_imgs[2], saved[2]), 0)
    return save_imgs


def save_images(imgs, epoch, folder, resize_to=None):
    original_images, watermarked_images, manipulated_images = imgs

    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2
    manipulated_images = (manipulated_images + 1) / 2

    # resize noised_images
    if manipulated_images.shape != images.shape:
        resize = nn.UpsamplingNearest2d(size=(images.shape[2], images.shape[3]))
        manipulated_images = resize(manipulated_images)

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    diff_images = (watermarked_images - images + 1) / 2

    # transform to rgb
    diff_images_linear = diff_images.clone()
    R = diff_images_linear[:, 0, :, :]
    G = diff_images_linear[:, 1, :, :]
    B = diff_images_linear[:, 2, :, :]
    diff_images_linear[:, 0, :, :] = 0.299 * R + 0.587 * G + 0.114 * B
    diff_images_linear[:, 1, :, :] = diff_images_linear[:, 0, :, :]
    diff_images_linear[:, 2, :, :] = diff_images_linear[:, 0, :, :]
    diff_images_linear = torch.abs(diff_images_linear * 2 - 1)

    # maximize diff in every image
    for id in range(diff_images_linear.shape[0]):
        diff_images_linear[id] = (diff_images_linear[id] - diff_images_linear[id].min()) / (
                diff_images_linear[id].max() - diff_images_linear[id].min())

    stacked_images = torch.cat(
        [images.unsqueeze(0), watermarked_images.unsqueeze(0), manipulated_images.unsqueeze(0),
         diff_images.unsqueeze(0), diff_images_linear.unsqueeze(0)], dim=0)
    shape = stacked_images.shape
    stacked_images = stacked_images.permute(0, 3, 1, 4, 2).reshape(shape[3] * shape[0], shape[4] * shape[1], shape[2])
    stacked_images = stacked_images.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))

    saved_image = Image.fromarray(np.array(stacked_images, dtype=np.uint8)).convert("RGB")
    saved_image.save(filename)
