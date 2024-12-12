from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import random
import numpy as np

import torch
import torchvision
from torch import nn
from torch.backends import cudnn
from model import DehazeNet
from gcl.utils.gan_utils import get_config, prepare_sub_folder, denormalize1

from data.data_loader import CreateDataLoader

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):

    cudnn.benchmark = True

    config = get_config(args.config)

    print("==========\nArgs:{}\n==========".format(args))

    # data load
    data_loader_train = CreateDataLoader(args)
    dataset_train = data_loader_train.load_data()
    dataset_size = len(data_loader_train)
    print('#training images = %d' % dataset_size)

    # training data
    output_directory = osp.join(args.output_path + "/checkpoints", args.name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

    # Trainer
    trainer = DehazeNet(config).cuda()

    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if args.resume else 0

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        trainer = nn.DataParallel(trainer, device_ids=range(torch.cuda.device_count()))

    trainer.train(True)

    for epoch in range(args.epochs):
        for it, data in enumerate(dataset_train):
            clear, pseudo_hazy = data['clear'].cuda(), data['hazy'].cuda()

            dehazed, loss_dis_total, loss_gen_total, loss_gen_adv_recon, loss_gen_recon_x, loss_gen_vgg_f = trainer.forward(clear, pseudo_hazy, config)

            loss_G = loss_gen_total.mean()
            loss_D = loss_dis_total.mean()
            loss_D = loss_D*0.5
            trainer.module.gen_update(loss_G)
            trainer.module.dis_update(loss_D)
            torch.cuda.synchronize()

            if (iterations + 1) % config['image_display_iter'] == 0:
                x_clear = denormalize1(clear)
                x_hazy = denormalize1(pseudo_hazy)
                syn_hazy = denormalize1(dehazed)
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(torch.cat((x_clear, x_hazy, syn_hazy), 0),
                                                nrow=x_clear.shape[0]),
                    os.path.join(image_directory, '{}.jpg'.format(iterations)))

            iterations += 1

            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.module.save(checkpoint_directory, iterations)

        trainer.module.update_learning_rate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    # data
    parser.add_argument('-b', '--batchSize', type=int, default=4)
    parser.add_argument('--serial_batches', action='store_true',
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                        help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--resize_or_crop', type=str, default='resize',
                        help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')

    parser.add_argument('--isTrain', default=True, help='training')

    # gan config
    parser.add_argument('--config', type=str, default='./configs/latest.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--name', type=str, default='latest', help="outputs path")

    parser.add_argument('--epochs', type=int, default=300)
    # training configs

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--resume", default=False)


    main()
