import os

import time

import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core
import os
from RegisterHook import train_register_forwardhook_for_resnet
import argparse
from torch.utils.data import DataLoader
import core as core
from torch.utils.data import random_split
import random
from core.utils import test
from RegisterHook import register_forwardhook_for_resnet



def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def adjust_learning_rate(lr, optimizer, epoch):
    if epoch in [20]:
        lr*=0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(model, schedule,train_dataset):
    print("--------fine tuning-------")
    current_schedule=schedule

    if 'pretrain' in current_schedule:
        model.load_state_dict(torch.load(current_schedule['pretrain']), strict=False)

    # Use GPU
    if 'device' in current_schedule and current_schedule['device'] == 'GPU':
        if 'CUDA_VISIBLE_DEVICES' in current_schedule:
            os.environ['CUDA_VISIBLE_DEVICES'] = current_schedule['CUDA_VISIBLE_DEVICES']

        assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
        assert current_schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
        print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {current_schedule['GPU_num']} of them to train.")

        if current_schedule['GPU_num'] == 1:
            device = torch.device("cuda:0")
        else:
            gpus = list(range(current_schedule['GPU_num']))
            model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
            # TODO: DDP training
            pass
    # Use CPU
    else:
        device = torch.device("cpu")

    if current_schedule['benign_training'] is True:
        train_loader = DataLoader(
            train_dataset,
            batch_size=current_schedule['batch_size'],
            shuffle=True,
            num_workers=current_schedule['num_workers'],
            drop_last=True,
            pin_memory=True,
            worker_init_fn=_seed_worker
        )

    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=current_schedule['lr'], momentum=current_schedule['momentum'], weight_decay=current_schedule['weight_decay'])

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

    iteration = 0
    last_time = time.time()

    for i in range(current_schedule['epochs']):
        adjust_learning_rate(current_schedule['lr'],optimizer, i)
        for batch_id, batch in enumerate(train_loader):
            batch_img = batch[0]
            batch_label = batch[1]
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)
            optimizer.zero_grad()
            predict_digits = model(batch_img)
            loss = torch.nn.functional.cross_entropy(predict_digits, batch_label)
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % current_schedule['log_iteration_interval'] == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ",
                                    time.localtime()) + f"Epoch:{i + 1}/{current_schedule['epochs']}, iteration:{batch_id + 1}/{len(train_dataset) // current_schedule['batch_size']}, lr: {current_schedule['lr']}, loss: {float(loss)}, time: {time.time() - last_time}\n"
                last_time = time.time()
                print(msg)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--datasets', type=str, default='cifar10', choices=['cifar10', 'gtsrb'])
    parser.add_argument('--model_path', '--mp', type=str, default='./')
    parser.add_argument('--posiondataset_path', '--pdp', type=str, default='./')
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--layer',type=str, default='layer3')
    parser.add_argument('--location', '--loc', type=str, default='layer3.1.shortcut')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()

    deterministic = True
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_class=0


    if args.datasets=='cifar10':
        # Define Benign Training and Testing Dataset
        dataset = torchvision.datasets.CIFAR10
        # dataset = torchvision.datasets.MNIST

        transform_train = Compose([
            ToTensor(),
            RandomHorizontalFlip()
        ])
        trainset = dataset('data', train=True, transform=transform_train, download=True)

        transform_test = Compose([
            ToTensor()
        ])
        testset = dataset('data', train=False, transform=transform_test, download=True)

        fttrainset, fttestset = random_split(testset, [5000, 5000])
        num_class=10

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
            root='/data/ganguanhao/datasets/GTSRB/train',  # please replace this with path to your training set
            loader=cv2.imread,
            extensions=('png',),
            transform=transform_train,
            target_transform=None,
            is_valid_file=None)

        testset = dataset(
            root='/data/ganguanhao/datasets/GTSRB/testset',  # please replace this with path to your test set
            loader=cv2.imread,
            extensions=('png',),
            transform=transform_test,
            target_transform=None,
            is_valid_file=None)

        fttrainset, fttestset = random_split(testset, [3920, 8710])
        num_class=43


    mymodel=core.models.ResNet(18,num_class)
    mymodel.load_state_dict(torch.load(args.model_path))
    gamma=0
    hook = register_forwardhook_for_resnet(mymodel, args.model, gamma, args.location)
    for name, child in mymodel.named_children():
        # if not "layer2" in name and not"layer3" in name:
        if not args.layer in name:
            for param in child.parameters():
                param.requires_grad = False

    poisoned_test_dataset=torch.load(args.posiondataset_path)

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': args.gpu,
        'GPU_num': 1,

        'benign_training': True, # Train Benign Model
        'batch_size': 128,
        'num_workers': 4,

        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [],

        'epochs': 10,
        'log_iteration_interval': 100,
    }

    train(mymodel,schedule,fttrainset)
    test_schedule2 = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': args.gpu,
        'GPU_num': 1,

        'batch_size': 128,
        'num_workers': 4,
        'metric': 'BA',

        'save_dir': 'experiments',
        'experiment_name': 'finetuning_CIFAR10_BadNets'
    }
    test_schedule3 = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': args.gpu,
        'GPU_num': 1,

        'batch_size': 128,
        'num_workers': 4,
        'metric': 'ASR_NoTarget',
        'y_target': args.target,

        'save_dir': 'experiments',
        'experiment_name': 'finetuning_CIFAR10_BadNets'
    }
    test(mymodel,fttestset,test_schedule2)
    test(mymodel, poisoned_test_dataset, test_schedule3)