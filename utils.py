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

def make_loader(configs, model_mode='train', shuffle=True):
    """ Construct dataloader.

        model_mode: train | val | test
    """
    dataset = ImageDataset(
        os.path.join(configs.img_path, model_mode),
        os.path.join(configs.wm_path, str(configs.img_size), model_mode),
        configs.img_size,
        configs.watermark_length,
        mode=model_mode
    )
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
