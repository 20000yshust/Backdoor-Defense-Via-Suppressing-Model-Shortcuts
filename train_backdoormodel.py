import os

import cv2
import torch
import torch.nn as nn
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



def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--datasets', type=str, default='cifar10', choices=['cifar10', 'gtsrb'])
    parser.add_argument('--attack', type=str, default='benign', choices=['benign', 'badnets','blended','label_consistent','wanet'])
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--poisoned_rate',type=float,default=0.05)
    parser.add_argument('--batch_size', '--bs',type=int, default=128)
    parser.add_argument('--model_save_path', '--mp', type=str, default='experiments')
    parser.add_argument('--posiondataset_save_path', '--pdp', type=str, default='./')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()

    deterministic = True
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    num_classes=0

    if args.datasets == 'cifar10':
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
        num_classes=10

        if args.attack == 'benign':
            pattern = torch.zeros((32, 32), dtype=torch.uint8)
            pattern[-3:, -3:] = 255
            weight = torch.zeros((32, 32), dtype=torch.float32)
            weight[-3:, -3:] = 1.0

            badnets = core.BadNets(
                train_dataset=trainset,
                test_dataset=testset,
                model=core.models.ResNet(18,num_classes),
                loss=nn.CrossEntropyLoss(),
                y_target=args.target,
                poisoned_rate=args.poisoned_rate,
                pattern=pattern,
                weight=weight,
                seed=args.seed,
                deterministic=deterministic
            )
            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'benign_training': True,
                'batch_size': 128,
                'num_workers': 2,

                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [150, 180],

                'epochs': 200,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 10,

                'save_dir': args.model_save_path,
                'experiment_name': 'ResNet-18_CIFAR-10_Benign'
            }
            badnets.train(schedule)

        elif args.attack == 'badnets':
            pattern = torch.zeros((32, 32), dtype=torch.uint8)
            pattern[-3:, -3:] = 255
            weight = torch.zeros((32, 32), dtype=torch.float32)
            weight[-3:, -3:] = 1.0

            badnets = core.BadNets(
                train_dataset=trainset,
                test_dataset=testset,
                model=core.models.ResNet(18,num_classes),
                loss=nn.CrossEntropyLoss(),
                y_target=args.target,
                poisoned_rate=args.poisoned_rate,
                pattern=pattern,
                weight=weight,
                seed=args.seed,
                deterministic=deterministic
            )
            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'benign_training': False,
                'batch_size': 128,
                'num_workers': 2,

                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [150, 180],

                'epochs': 200,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 10,

                'save_dir': args.model_save_path,
                'experiment_name': 'ResNet-18_CIFAR-10_BadNets'
            }
            badnets.train(schedule)
            poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
            path=os.path.join(args.posiondataset_save_path,'poisoned_test_dataset_badnets_cifar10.pth')
            torch.save(poisoned_test_dataset,path)

        elif args.attack=='blended':
            pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            pattern[0, -3:, -3:] = 255
            weight = torch.zeros((1, 32, 32), dtype=torch.float32)
            weight[0, -3:, -3:] = 0.2

            blended = core.Blended(
                train_dataset=trainset,
                test_dataset=testset,
                model=core.models.ResNet(18, num_classes),
                # model=core.models.BaselineMNISTNetwork(),
                loss=nn.CrossEntropyLoss(),
                pattern=pattern,
                weight=weight,
                y_target=args.target,
                poisoned_rate=args.poisoned_rate,
                seed=args.seed,
                deterministic=deterministic
            )

            poisoned_train_dataset, poisoned_test_dataset = blended.get_poisoned_dataset()

            # Train Infected Model
            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'benign_training': False,
                'batch_size': 128,
                'num_workers': 4,

                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [150, 180],

                'epochs': 200,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 10,

                'save_dir': args.model_save_path,
                'experiment_name': 'train_poisoned_CIFAR10_Blended'
            }
            blended.train(schedule)
            path=os.path.join(args.posiondataset_save_path,'poisoned_test_dataset_blended_cifar10.pth')
            torch.save(poisoned_test_dataset,path)

        elif args.attack=='label_consistent':
            adv_model = core.models.ResNet(18,num_classes)
            #replace with your benign model
            adv_model.load_state_dict(torch.load("/data/yangsheng/graduationproject/Mybenigndmodel_666_resnet18.pth.tar"))

            pattern = torch.zeros((32, 32), dtype=torch.uint8)
            pattern[-1, -1] = 255
            pattern[-1, -3] = 255
            pattern[-3, -1] = 255
            pattern[-2, -2] = 255

            pattern[0, -1] = 255
            pattern[1, -2] = 255
            pattern[2, -3] = 255
            pattern[2, -1] = 255

            pattern[0, 0] = 255
            pattern[1, 1] = 255
            pattern[2, 2] = 255
            pattern[2, 0] = 255

            pattern[-1, 0] = 255
            pattern[-1, 2] = 255
            pattern[-2, 1] = 255
            pattern[-3, 0] = 255

            weight = torch.zeros((32, 32), dtype=torch.float32)
            weight[:3, :3] = 1.0
            weight[:3, -3:] = 1.0
            weight[-3:, :3] = 1.0
            weight[-3:, -3:] = 1.0

            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'benign_training': False,  # Train Attacked Model
                'batch_size': 128,
                'num_workers': 8,

                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [150, 180],

                'epochs': 200,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 10,

                'save_dir': 'experiments',
                'experiment_name': 'train_poisioned_CIFAR10_LabelConsistent'
            }

            eps = 8
            alpha = 1.5
            steps = 100
            max_pixel = 255
            poisoned_rate = 0.05

            label_consistent = core.LabelConsistent(
                train_dataset=trainset,
                test_dataset=testset,
                model=core.models.ResNet(18,num_classes),
                adv_model=adv_model,
                adv_dataset_dir=f'./adv_dataset/CIFAR-10_resnet18_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{args.poisoned_rate}_seed{args.seed}',
                loss=nn.CrossEntropyLoss(),
                y_target=args.target,
                poisoned_rate=args.poisoned_rate,
                pattern=pattern,
                weight=weight,
                eps=eps,
                alpha=alpha,
                steps=steps,
                max_pixel=max_pixel,
                poisoned_transform_train_index=0,
                poisoned_transform_test_index=0,
                poisoned_target_transform_index=0,
                schedule=schedule,
                seed=args.seed,
                deterministic=True
            )

            label_consistent.train()
            poisoned_train_dataset, poisoned_test_dataset = label_consistent.get_poisoned_dataset()
            path=os.path.join(args.posiondataset_save_path,'poisoned_test_dataset_label_consistent_cifar10.pth')
            torch.save(poisoned_test_dataset,path)

        elif args.attack=='wanet':
            identity_grid, noise_grid = gen_grid(32, 4)
            torch.save(identity_grid, 'ResNet-18_CIFAR-10_WaNet_identity_grid.pth')
            torch.save(noise_grid, 'ResNet-18_CIFAR-10_WaNet_noise_grid.pth')
            wanet = core.WaNet(
                train_dataset=trainset,
                test_dataset=testset,
                model=core.models.ResNet(18,num_classes),
                loss=nn.CrossEntropyLoss(),
                y_target=args.target,
                poisoned_rate=args.poisoned_rate,
                identity_grid=identity_grid,
                noise_grid=noise_grid,
                noise=False,
                seed=args.seed,
                deterministic=deterministic
            )

            poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()
            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'benign_training': False,  # Train Attacked Model
                'batch_size': 128,
                'num_workers': 8,

                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [150, 180],

                'epochs': 200,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 10,

                'save_dir': 'experiments',
                'experiment_name': 'train_poisioned_CIFAR10_WaNet'
            }
            wanet.train(schedule)
            path=os.path.join(args.posiondataset_save_path,'poisoned_test_dataset_wanet_cifar10.pth')
            torch.save(poisoned_test_dataset,path)








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
        num_classes=43
        if args.attack == 'benign':
            pattern = torch.zeros((32, 32), dtype=torch.uint8)
            pattern[-3:, -3:] = 255
            weight = torch.zeros((32, 32), dtype=torch.float32)
            weight[-3:, -3:] = 1.0

            badnets = core.BadNets(
                train_dataset=trainset,
                test_dataset=testset,
                model=core.models.ResNet(18,num_classes),
                loss=nn.CrossEntropyLoss(),
                y_target=args.target,
                poisoned_rate=args.poisoned_rate,
                pattern=pattern,
                weight=weight,
                poisoned_transform_train_index=2,
                poisoned_transform_test_index=2,
                seed=args.seed,
                deterministic=deterministic
            )
            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'benign_training': True,
                'batch_size': 128,
                'num_workers': 2,

                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [20],

                'epochs': 30,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 10,

                'save_dir': args.model_save_path,
                'experiment_name': 'ResNet-18_GTSRB_Benign'
            }
            badnets.train(schedule)

        elif args.attack == 'badnets':
            pattern = torch.zeros((32, 32), dtype=torch.uint8)
            pattern[-3:, -3:] = 255
            weight = torch.zeros((32, 32), dtype=torch.float32)
            weight[-3:, -3:] = 1.0

            badnets = core.BadNets(
                train_dataset=trainset,
                test_dataset=testset,
                model=core.models.ResNet(18,num_classes),
                loss=nn.CrossEntropyLoss(),
                y_target=args.target,
                poisoned_rate=args.poisoned_rate,
                pattern=pattern,
                weight=weight,
                poisoned_transform_train_index=2,
                poisoned_transform_test_index=2,
                seed=args.seed,
                deterministic=deterministic
            )
            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'benign_training': False,
                'batch_size': 128,
                'num_workers': 2,

                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [20],

                'epochs': 30,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 10,

                'save_dir': args.model_save_path,
                'experiment_name': 'ResNet-18_GTSRB_BadNets'
            }
            badnets.train(schedule)
            poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
            path=os.path.join(args.posiondataset_save_path,'poisoned_test_dataset_badnets_gtsrb.pth')
            torch.save(poisoned_test_dataset,path)

        elif args.attack=='blended':
            pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            pattern[0, -3:, -3:] = 255
            weight = torch.zeros((1, 32, 32), dtype=torch.float32)
            weight[0, -3:, -3:] = 0.2

            blended = core.Blended(
                train_dataset=trainset,
                test_dataset=testset,
                model=core.models.ResNet(18, num_classes),
                # model=core.models.BaselineMNISTNetwork(),
                loss=nn.CrossEntropyLoss(),
                pattern=pattern,
                weight=weight,
                poisoned_transform_train_index=2,
                poisoned_transform_test_index=2,
                y_target=args.target,
                poisoned_rate=args.poisoned_rate,
                seed=args.seed,
                deterministic=deterministic
            )

            poisoned_train_dataset, poisoned_test_dataset = blended.get_poisoned_dataset()

            # Train Infected Model
            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'benign_training': False,
                'batch_size': 128,
                'num_workers': 4,

                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [20],

                'epochs': 30,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 10,

                'save_dir': args.model_save_path,
                'experiment_name': 'train_poisoned_GTSRB_Blended'
            }
            blended.train(schedule)
            path=os.path.join(args.posiondataset_save_path,'poisoned_test_dataset_blended_gtsrb.pth')
            torch.save(poisoned_test_dataset,path)

        elif args.attack=='label_consistent':
            adv_model = core.models.ResNet(18,num_classes)
            #replace with your benign model
            adv_model.load_state_dict(torch.load("/data/yangsheng/graduationproject/Mybenigndmodel_666_gtsrb.pth.tar"))

            pattern = torch.zeros((32, 32), dtype=torch.uint8)
            pattern[-1, -1] = 255
            pattern[-1, -3] = 255
            pattern[-3, -1] = 255
            pattern[-2, -2] = 255

            pattern[0, -1] = 255
            pattern[1, -2] = 255
            pattern[2, -3] = 255
            pattern[2, -1] = 255

            pattern[0, 0] = 255
            pattern[1, 1] = 255
            pattern[2, 2] = 255
            pattern[2, 0] = 255

            pattern[-1, 0] = 255
            pattern[-1, 2] = 255
            pattern[-2, 1] = 255
            pattern[-3, 0] = 255

            weight = torch.zeros((32, 32), dtype=torch.float32)
            weight[:3, :3] = 1.0
            weight[:3, -3:] = 1.0
            weight[-3:, :3] = 1.0
            weight[-3:, -3:] = 1.0

            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'benign_training': False,  # Train Attacked Model
                'batch_size': 128,
                'num_workers': 8,

                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [20],

                'epochs': 30,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 10,

                'save_dir': 'experiments',
                'experiment_name': 'train_poisioned_gtsrb_LabelConsistent'
            }

            eps = 16
            alpha = 1.5
            steps = 100
            max_pixel = 255
            poisoned_rate = 0.5

            label_consistent = core.LabelConsistent(
                train_dataset=trainset,
                test_dataset=testset,
                model=core.models.ResNet(18, num_classes),
                adv_model=adv_model,
                adv_dataset_dir=f'./adv_dataset/GTSRB3_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{args.poisoned_rate}_seed{args.seed}',
                loss=nn.CrossEntropyLoss(),
                y_target=args.target,
                poisoned_rate=args.poisoned_rate,
                adv_transform=Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), ToTensor()]),
                pattern=pattern,
                weight=weight,
                eps=eps,
                alpha=alpha,
                steps=steps,
                max_pixel=max_pixel,
                poisoned_transform_train_index=2,
                poisoned_transform_test_index=2,
                poisoned_target_transform_index=0,
                schedule=schedule,
                seed=args.seed,
                deterministic=True
            )

            label_consistent.train()
            poisoned_train_dataset, poisoned_test_dataset = label_consistent.get_poisoned_dataset()
            path=os.path.join(args.posiondataset_save_path,'poisoned_test_dataset_label_consistent_gtsrb.pth')
            torch.save(poisoned_test_dataset,path)

        elif args.attack=='wanet':
            identity_grid, noise_grid = gen_grid(32, 4)
            torch.save(identity_grid, 'ResNet-18_GTSRB_WaNet_identity_grid.pth')
            torch.save(noise_grid, 'ResNet-18_GTSRB_WaNet_noise_grid.pth')
            wanet = core.WaNet(
                train_dataset=trainset,
                test_dataset=testset,
                model=core.models.ResNet(18,num_classes),
                loss=nn.CrossEntropyLoss(),
                y_target=args.target,
                poisoned_rate=args.poisoned_rate,
                identity_grid=identity_grid,
                noise_grid=noise_grid,
                noise=False,
                seed=args.seed,
                deterministic=deterministic
            )

            poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()
            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': args.gpu,
                'GPU_num': 1,

                'benign_training': False,  # Train Attacked Model
                'batch_size': 128,
                'num_workers': 8,

                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [20],

                'epochs': 30,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 10,

                'save_dir': 'experiments',
                'experiment_name': 'train_poisioned_gtsrb_WaNet'
            }
            wanet.train(schedule)
            path=os.path.join(args.posiondataset_save_path,'poisoned_test_dataset_wanet_gtsrb.pth')
            torch.save(poisoned_test_dataset,path)






