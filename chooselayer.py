'''
This program is to choose which layer is the critical layer.
The method is based on the Badnets to choose the critical layer.

To be clear, the method is to choose the biggest decline on ASR out, and the location this decline in is the critical layer.
The method is done on every skip-connection.
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
import core
import numpy as np
import os
import cv2
import argparse



from RegisterHook import register_forwardhook_for_resnet
from core.utils import test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--datasets', type=str, default='cifar10', choices=['cifar10', 'gtsrb'])
    parser.add_argument('--model_path', '--mp', type=str, default='./')
    parser.add_argument('--posiondataset_path', '--pdp', type=str, default='./')
    parser.add_argument('--batch_size', '--bs',type=int, default=128)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()


    deterministic = True
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    poisoned_test_dataset = torch.load(args.posiondataset_path)


    if args.datasets=='cifar10':
        dataset = torchvision.datasets.CIFAR10

        transform_train = Compose([
            ToTensor(),
            RandomHorizontalFlip()
        ])
        trainset = dataset('data', train=True, transform=transform_train, download=True)

        transform_test = Compose([
            ToTensor()
        ])
        testset = dataset('data', train=False, transform=transform_test, download=True)

    elif args.datasets=='gtsrb':
        dataset = torchvision.datasets.DatasetFolder


        # image file -> cv.imread -> numpy.ndarray (H x W x C) -> ToTensor -> torch.Tensor (C x H x W) -> RandomHorizontalFlip -> resize (32) -> torch.Tensor -> network input
        transform_train = Compose([
            ToPILImage(),
            Resize((32, 32)),
            ToTensor()
        ])
        transform_test = Compose([
            ToPILImage(),
            Resize((32, 32)),
            ToTensor()
        ])


        trainset = dataset(
            root='/data/ganguanhao/datasets/GTSRB/train', # please replace this with path to your training set
            loader=cv2.imread,
            extensions=('png',),
            transform=transform_train,
            target_transform=None,
            is_valid_file=None)

        testset = dataset(
            root='/data/ganguanhao/datasets/GTSRB/testset', # please replace this with path to your test set
            loader=cv2.imread,
            extensions=('png',),
            transform=transform_test,
            target_transform=None,
            is_valid_file=None)


    global_different = 0
    location = ''

    for j in range(1, 9):
        loc = 'layer' + str(int((j + 1) / 2)).replace('.0', '') + '.' + str((j + 1) % 2) + '.shortcut'
        print(loc)
        for i in range(0, 11):
            gamma = i * 0.1
            print(gamma)
            mymodel = core.models.ResNet(18)
            mymodel.load_state_dict(torch.load(args.model_path))

            register_forwardhook_for_resnet(mymodel, 'resnet18', gamma, loc)

            test_schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'batch_size': 128,
                'num_workers': 4,
                'metric': 'ASR_NoTarget',
                'y_target': 1,

                'save_dir': 'experiments',
                'experiment_name': 'Layerselect_CIFAR10_BadNets'
            }
            top1, top5 = test(mymodel, poisoned_test_dataset, test_schedule)
            if formervalue == -1:
                formervalue = top1
            else:
                difference = (top1 - formervalue) / np.exp(formervalue)
                if difference > global_different:
                    global_different = difference
                    location = loc
                formervalue = top1

    print(global_different)
    print(location)