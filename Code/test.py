from __future__ import print_function, absolute_import
import argparse
import os
import random
import numpy as np
import torch
import yaml
import torchvision
from torch import nn
from torch.backends import cudnn
from model import DehazeNet
from gcl.utils.gan_utils import get_config, denormalize1
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
    print('#testing images = %d' % dataset_size)

    # Trainer
    trainer = DehazeNet(config).cuda()

    model_path = './pretrained/SSID.pt'
    state_dict = torch.load(model_path)
    trainer.gen.load_state_dict(state_dict['gen'])


    for m in trainer.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        trainer = nn.DataParallel(trainer, device_ids=range(torch.cuda.device_count()))

    with torch.no_grad():
        for it, data in enumerate(dataset_train):
            pseudo_hazy = data['hazy'].cuda()
            dehazed = trainer.module.sample(pseudo_hazy)
            syn_hazy = denormalize1(dehazed)
            torchvision.utils.save_image(syn_hazy, os.path.join(args.output, '{}.png'.format(it)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    # data
    parser.add_argument('-b', '--batchSize', type=int, default=1)
    parser.add_argument('--serial_batches', action='store_true',
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
    parser.add_argument('--dataroot', type=str, default='./examples/input')
    parser.add_argument('--output', type=str,
                        default='./examples/output')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                        help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--resize_or_crop', type=str, default='resize',
                        help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')

    parser.add_argument('--isTrain', default=False, help='testing')

    # gan config
    parser.add_argument('--config', type=str, default='./configs/latest.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--name', type=str, default='latest', help="outputs path")

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--resume", default=True)

    main()
