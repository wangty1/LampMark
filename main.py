import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from utils import JsonConfig
from utils import make_loader, write_log, plot_curves, format_time, get_random_images, save_images, \
    concatenate_images
from trainer_img import TrainerImg
from tester_img import TesterImg

import sys

sys.path.insert(0, './model/SimSwap')
sys.path.insert(0, './model/stylemask')

# Global variables to be used for most functions.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

totensor = transforms.ToTensor()
norm = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)


def define_result_dict(configs, ret_type):
    result_dict = {}
    total_dict = {}

    result_dict['error_rate'] = 0.0
    result_dict['psnr'] = 0.0
    result_dict['ssim'] = 0.0
    result_dict['g_loss'] = 0.0
    result_dict['g_loss_enc'] = 0.0
    result_dict['g_loss_dec'] = 0.0
    result_dict['g_loss_dis'] = 0.0
    result_dict['d_loss_raw'] = 0.0
    result_dict['d_loss_wmd'] = 0.0

    total_dict['error_rate'] = []
    total_dict['psnr'] = []
    total_dict['ssim'] = []
    total_dict['g_loss'] = []
    total_dict['g_loss_enc'] = []
    total_dict['g_loss_dec'] = []
    total_dict['g_loss_dis'] = []
    total_dict['d_loss_raw'] = []
    total_dict['d_loss_wmd'] = []

    if configs.manipulation_mode == 'deepfake':
        result_dict['g_loss_gen'] = 0.0
        result_dict['g_loss_id'] = 0.0
        total_dict['g_loss_gen'] = []
        total_dict['g_loss_id'] = []

    if ret_type == 'value':
        return result_dict
    elif ret_type == 'list':
        return total_dict
    else:
        return result_dict, total_dict


def train_common():
    configs = JsonConfig()
    configs.load_json_file('./configurations/pretrain.json')
    trainer = TrainerImg(configs, device)

    train_loader = make_loader(configs, model_mode='train', shuffle=True)
    val_loader = make_loader(configs, model_mode='val', shuffle=False)

    loss_record_train = define_result_dict(configs, 'list')
    loss_record_val = define_result_dict(configs, 'list')

    print('Training on going ...')
    for epoch in range(configs.epochs):
        start_time = time.time()
        running_result = define_result_dict(configs, 'value')

        # Train
        batch_count = 0
        for imgs, watermarks in tqdm(train_loader):
            result = trainer.train_batch_common(imgs, watermarks)
            for key in running_result:
                running_result[key] += float(result[key])
            batch_count += 1

        log_text = 'Epoch ' + str(epoch + 1) + ': ' + format_time(time.time() - start_time) + ' elapsed.\n'
        for key in running_result:
            log_text = log_text + key + '=' + str(running_result[key] / batch_count) + '\n'
            loss_record_train[key].append(float(running_result[key] / batch_count))
        log_text = log_text + '\n'

        if not os.path.exists('./results/wm-img'):
            os.mkdir('./results/wm-img')
        with open('./results/wm-img/train_log.txt', 'a') as file:
            file.write(log_text)
        print(log_text)

        encoder_path = configs.weight_path + '/wm-img/encoder_epoch_' + str(epoch + 1) + '.pth'
        decoder_path = configs.weight_path + '/wm-img/decoder_epoch_' + str(epoch + 1) + '.pth'
        discriminator_path = configs.weight_path + '/discriminator/epoch_' + str(epoch + 1) + '.pth'
        # trainer.save_model(encoder_path, decoder_path, discriminator_path)
        model_path = configs.weight_path + '/wm-img/model_epoch_' + str(epoch + 1) + '.pth'
        trainer.save_model_integral(model_path, discriminator_path)

        # Validation
        if configs.do_validation:
            start_time = time.time()
            running_result = define_result_dict(configs, 'value')

            save_iters = np.random.choice(np.arange(len(val_loader)), size=configs.save_img_nums, replace=False)
            save_imgs = None
            batch_count = 0
            for imgs, watermarks in tqdm(val_loader):
                result, output_lst = trainer.val_batch_common(imgs, watermarks)
                for key in running_result:
                    running_result[key] += float(result[key])

                if batch_count in save_iters:
                    if save_imgs is None:
                        save_imgs = get_random_images(output_lst[0], output_lst[1], output_lst[2])
                    else:
                        save_imgs = concatenate_images(save_imgs, output_lst[0], output_lst[1], output_lst[2])
                batch_count += 1

            log_text = 'Validation finished in ' + format_time(time.time() - start_time) + '.\n'
            for key in running_result:
                log_text = log_text + key + '=' + str(running_result[key] / batch_count) + '\n'
                loss_record_val[key].append(float(running_result[key] / batch_count))
            log_text = log_text + '\n'
            print(log_text)
            save_images(save_imgs, epoch + 1, './results/images', resize_to=(configs.img_size, configs.img_size))

    plot_curves(configs.epochs, loss_record_train, loss_record_val, './results/wm-img')


def tune_deepfake():
    configs = JsonConfig()
    configs.load_json_file('./configurations/tune_deepfake.json')
    trainer = TrainerImg(configs, device)

    model_path = configs.weight_path + '/wm-img/model_epoch_' + str(configs.epoch) + '.pth'
    trainer.load_model_integral(model_path)
    discriminator_path = configs.weight_path + '/discriminator/epoch_' + str(configs.epoch) + '.pth'
    trainer.load_discriminator(discriminator_path)

    train_loader = make_loader(configs, model_mode='train', shuffle=True)
    val_loader = make_loader(configs, model_mode='val', shuffle=False)

    loss_record_train = define_result_dict(configs, 'list')
    loss_record_val = define_result_dict(configs, 'list')

    print('Training on going ...')
    for epoch in range(configs.epochs):
        start_time = time.time()
        running_result = define_result_dict(configs,  'value')

        # Train
        batch_count = 0
        for imgs, watermarks in tqdm(train_loader):
            result = trainer.tune_batch_simswap(imgs, watermarks)
            for key in running_result:
                running_result[key] += float(result[key])
            batch_count += 1

        log_text = 'Epoch ' + str(epoch + 1) + ': ' + format_time(time.time() - start_time) + ' elapsed.\n'
        for key in running_result:
            log_text = log_text + key + '=' + str(running_result[key] / batch_count) + '\n'
            loss_record_train[key].append(float(running_result[key] / batch_count))
        log_text = log_text + '\n'

        if not os.path.exists('./results/deepfake'):
            os.mkdir('./results/deepfake')
        with open('./results/deepfake/train_log.txt', 'a') as file:
            file.write(log_text)
        print(log_text)

        encoder_path = configs.weight_path + '/deepfake/encoder_epoch_' + str(epoch + 1) + '.pth'
        decoder_path = configs.weight_path + '/deepfake/decoder_epoch_' + str(epoch + 1) + '.pth'
        discriminator_path = configs.weight_path + '/deepfake_discriminator/epoch_' + str(epoch + 1) + '.pth'
        trainer.save_model(encoder_path, decoder_path, discriminator_path)

        # Validation
        if configs.do_validation:
            start_time = time.time()
            running_result = define_result_dict(configs, 'value')

            save_iters = np.random.choice(np.arange(len(val_loader)), size=configs.save_img_nums, replace=False)
            save_imgs = None
            batch_count = 0
            for imgs, watermarks in tqdm(val_loader):
                result, output_lst = trainer.val_batch_simswap(imgs, watermarks)
                for key in running_result:
                    running_result[key] += float(result[key])

                if batch_count in save_iters:
                    if save_imgs is None:
                        if True:  # epoch % 2 == 0:
                            save_imgs = get_random_images(output_lst[0], output_lst[1], output_lst[2])
                        else:
                            save_imgs = get_random_images(output_lst[0], output_lst[1], output_lst[3])
                    else:
                        if True:  # epoch % 2 == 0:
                            save_imgs = concatenate_images(save_imgs, output_lst[0], output_lst[1], output_lst[2])
                        else:
                            save_imgs = concatenate_images(save_imgs, output_lst[0], output_lst[1], output_lst[3])
                batch_count += 1

            log_text = 'Validation finished in ' + format_time(time.time() - start_time) + '.\n'
            for key in running_result:
                log_text = log_text + key + '=' + str(running_result[key] / batch_count) + '\n'
                loss_record_val[key].append(float(running_result[key] / batch_count))
            log_text = log_text + '\n'
            print(log_text)
            save_images(
                save_imgs, epoch + 1, './results/images_deepfake', resize_to=(configs.img_size, configs.img_size)
            )

    plot_curves(configs.epochs, loss_record_train, loss_record_val, './results/deepfake')


def test_simswap():
    configs = JsonConfig()
    configs.load_json_file('./configurations/test_deepfake.json')
    tester = TesterImg(configs, device)

    encoder_path = configs.weight_path + '/deepfake/encoder_epoch_' + str(configs.epoch) + '.pth'
    decoder_path = configs.weight_path + '/deepfake/decoder_epoch_' + str(configs.epoch) + '.pth'
    tester.load_model(encoder_path, decoder_path)

    test_loader = make_loader(configs, model_mode='test', shuffle=False)

    running_error_rate = 0.0
    batch_count = 0
    for imgs, watermarks in tqdm(test_loader):
        result = tester.test_batch_simswap(imgs, watermarks)
        error_rate = result
        running_error_rate += float(error_rate)
        batch_count += 1

    print('Test finished for simswap.')
    print('Error Rate: ' + str(running_error_rate / batch_count))


def test_infoswap():
    configs = JsonConfig()
    configs.load_json_file('./configurations/test_deepfake.json')
    tester = TesterImg(configs, device)

    encoder_path = configs.weight_path + '/deepfake/encoder_epoch_' + str(configs.epoch) + '.pth'
    decoder_path = configs.weight_path + '/deepfake/decoder_epoch_' + str(configs.epoch) + '.pth'
    tester.load_model(encoder_path, decoder_path)

    test_loader = make_loader(configs, model_mode='test', shuffle=False)

    running_error_rate = 0.0
    batch_count = 0
    for imgs, watermarks in tqdm(test_loader):
        result = tester.test_batch_infoswap(imgs, watermarks)
        error_rate = result
        running_error_rate += float(error_rate)
        batch_count += 1

    print('Test finished for infoswap.')
    print('Error Rate: ' + str(running_error_rate / batch_count))


def test_uniface():
    configs = JsonConfig()
    configs.load_json_file('./configurations/test_deepfake.json')
    tester = TesterImg(configs, device)

    encoder_path = configs.weight_path + '/deepfake/encoder_epoch_' + str(configs.epoch) + '.pth'
    decoder_path = configs.weight_path + '/deepfake/decoder_epoch_' + str(configs.epoch) + '.pth'
    tester.load_model(encoder_path, decoder_path)

    test_loader = make_loader(configs, model_mode='test', shuffle=False)

    running_error_rate = 0.0
    batch_count = 0
    for imgs, watermarks in tqdm(test_loader):
        result = tester.test_batch_uniface(imgs, watermarks)
        error_rate = result
        running_error_rate += float(error_rate)
        batch_count += 1

    print('Test finished for uniface.')
    print('Error Rate: ' + str(running_error_rate / batch_count))


def test_stargan():
    configs = JsonConfig()
    configs.load_json_file('./configurations/test_deepfake.json')
    tester = TesterImg(configs, device)

    encoder_path = configs.weight_path + '/deepfake/encoder_epoch_' + str(configs.epoch) + '.pth'
    decoder_path = configs.weight_path + '/deepfake/decoder_epoch_' + str(configs.epoch) + '.pth'
    tester.load_model(encoder_path, decoder_path)

    test_loader = make_loader(configs, model_mode='test', shuffle=False)

    running_error_rate = 0.0
    batch_count = 0
    for imgs, watermarks in tqdm(test_loader):
        result = tester.test_batch_stargan(imgs, watermarks)
        error_rate = result
        running_error_rate += float(error_rate)
        batch_count += 1
    print('Test finished for stargan.')
    print('Error Rate: ' + str(running_error_rate / batch_count))


def test_stylemask():
    configs = JsonConfig()
    configs.load_json_file('./configurations/test_deepfake.json')
    tester = TesterImg(configs, device)

    encoder_path = configs.weight_path + '/deepfake/encoder_epoch_' + str(configs.epoch) + '.pth'
    decoder_path = configs.weight_path + '/deepfake/decoder_epoch_' + str(configs.epoch) + '.pth'
    tester.load_model(encoder_path, decoder_path)

    test_loader = make_loader(configs, model_mode='test', shuffle=False)

    running_error_rate = 0.0
    batch_count = 0
    for imgs, watermarks in tqdm(test_loader):
        result = tester.test_batch_stylemask(imgs, watermarks)
        error_rate = result
        running_error_rate += float(error_rate)
        batch_count += 1
    print('Test finished for stylemask.')
    print('Error Rate: ' + str(running_error_rate / batch_count))


def test_all_common():
    configs = JsonConfig()
    configs.load_json_file('./configurations/test_common.json')
    tester = TesterImg(configs, device)

    encoder_path = configs.weight_path + '/deepfake/encoder_epoch_' + str(configs.epoch) + '.pth'
    decoder_path = configs.weight_path + '/deepfake/decoder_epoch_' + str(configs.epoch) + '.pth'
    tester.load_model(encoder_path, decoder_path)
    # Use the following if plan to check the watermark robustness right after pre-training.
    # model_path = configs.weight_path + '/wm-img/model_epoch_' + str(configs.epoch) + '.pth'
    # tester.load_model_integral(model_path)

    test_loader = make_loader(configs, model_mode='test', shuffle=False)

    manipulation_lst = ['Dropout(0.5)', 'Resize(0.5)', 'GaussianNoise()', 'SaltPepper(0.05)', 'GaussianBlur(2,3)',
                        'MedBlur(3)', 'Brightness(0.5)', 'Contrast(0.5)', 'Saturation(0.5)', 'Hue(0.1)', 'Jpeg(50)',
                        'Identity()']
    error_rate_dict = {}
    for manipulation in manipulation_lst:
        error_rate_dict[manipulation] = 0.0
    psnr = 0.0
    ssim = 0.0

    for imgs, watermarks in tqdm(test_loader):
        for manipulation in manipulation_lst:
            if manipulation == 'Identity()':
                psnr_batch, ssim_batch, error_rate_batch = tester.test_one_manipulation(imgs, watermarks, manipulation)
                psnr += float(psnr_batch)
                ssim += float(ssim_batch)
            else:
                error_rate_batch = tester.test_one_manipulation(imgs, watermarks, manipulation)
            error_rate_dict[manipulation] += error_rate_batch / len(watermarks)

    print('psnr:', float(psnr) / len(test_loader))
    print('ssim:', float(ssim) / len(test_loader))
    for manipulation in error_rate_dict:
        error_rate_dict[manipulation] = error_rate_dict[manipulation] / len(test_loader)
    for manipulation in error_rate_dict:
        print(manipulation + ':', 1.0 - error_rate_dict[manipulation])
    print('Avg:', 1.0 - 1.0 * sum(error_rate_dict.values()) / len(error_rate_dict))


if __name__ == '__main__':
    # Pre-train using benign manipulations.
    train_common()

    # Fine-tune with Deepfake manipulations.
    # tune_deepfake()

    # Test benign.
    # test_all_common()

    # Test Deepfake.
    # test_simswap()
    # test_infoswap()
    # test_uniface()
    # TODO: gather and write code for E4S
    # test_stylemask()
    # test_stargan()
    # TODO: gather and write code for hyper-reenactment
