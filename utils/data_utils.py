import logging
from PIL import Image
import os

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.transforms import InterpolationMode

from .dataset import CUB, CarsDataset, NABirds, dogs, INat2017, food101, food172, food200  ##
from .autoaugment import AutoAugImageNetPolicy
from timm.data import create_transform, mixup

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'CUB_200_2011':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform = test_transform)
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
                            os.path.join(args.data_root,'cars_train'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((256, 256), Image.BILINEAR),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
        testset = CarsDataset(os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
                            os.path.join(args.data_root,'cars_test'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((256, 256), Image.BILINEAR),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
    elif args.dataset == 'dog':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                                train=True,
                                cropped=False,
                                transform=train_transform,
                                download=False
                                )
        testset = dogs(root=args.data_root,
                                train=False,
                                cropped=False,
                                transform=test_transform,
                                download=False
                                )
    elif args.dataset == 'nabirds':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                        transforms.RandomCrop((448, 448)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                        transforms.CenterCrop((448, 448)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.RandomCrop((304, 304)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.CenterCrop((304, 304)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)

    elif args.dataset == 'food101':
        train_transform = transforms.Compose([transforms.Resize((int(args.img_size / 0.875), int(args.img_size / 0.875)), InterpolationMode.BICUBIC),
                                              transforms.RandomCrop((int(args.img_size), int(args.img_size))),
                                              transforms.ColorJitter(brightness=0.126, saturation=0.5),
                                              transforms.RandomHorizontalFlip(),
                                              AutoAugImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5457954,0.44430383,0.34424934], [0.23273608,0.24383051,0.24237761])])

        test_transform = transforms.Compose([transforms.Resize((int(args.img_size / 0.875), int(args.img_size / 0.875)), InterpolationMode.BICUBIC),
                                             transforms.CenterCrop((int(args.img_size), int(args.img_size))),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5457954,0.44430383,0.34424934], [0.23273608,0.24383051,0.24237761])])

        trainset = food101(args.data_root, train_transform, True, False, args.multi_label)
        testset = food101(args.data_root, test_transform, False, False, args.multi_label)


    elif args.dataset == 'food172':
        
        train_transform = transforms.Compose([transforms.Resize((int(args.img_size / 0.875), int(args.img_size / 0.875)), InterpolationMode.BICUBIC),  ##
                                              transforms.RandomCrop((int(args.img_size), int(args.img_size))),
                                              transforms.ColorJitter(brightness=0.126, saturation=0.5),
                                              transforms.RandomHorizontalFlip(),
                                              AutoAugImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.600703, 0.506933, 0.386077], [0.074954, 0.082138, 0.094773])])

        test_transform = transforms.Compose([transforms.Resize((int(args.img_size / 0.875), int(args.img_size / 0.875)), InterpolationMode.BICUBIC),
                                             transforms.CenterCrop((int(args.img_size), int(args.img_size))),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.600703, 0.506933, 0.386077], [0.074954, 0.082138, 0.094773])])
        trainset = food172(args.data_root, train_transform, True, args.multi_label)  ##
        testset = food172(args.data_root, test_transform, False,  args.multi_label)

    elif args.dataset == 'food200':
        train_transform = transforms.Compose([transforms.Resize((int(args.img_size / 0.875), int(args.img_size / 0.875)), InterpolationMode.BICUBIC),  ##
                                              transforms.RandomCrop((int(args.img_size), int(args.img_size))),
                                              transforms.ColorJitter(brightness=0.126, saturation=0.5),
                                              transforms.RandomHorizontalFlip(),
                                              AutoAugImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.6257545, 0.5311906, 0.42223153], [0.26860937, 0.28052565, 0.30595714])])
        test_transform = transforms.Compose([transforms.Resize((int(args.img_size / 0.875), int(args.img_size / 0.875)), InterpolationMode.BICUBIC),
                                             transforms.CenterCrop((int(args.img_size), int(args.img_size))),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.6257545, 0.5311906, 0.42223153], [0.26860937, 0.28052565, 0.30595714])])
        trainset = food200(args.data_root, train_transform, True, args.multi_label)  ##
        testset = food200(args.data_root, test_transform, False,  args.multi_label)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)

    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.worker_num,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=args.worker_num,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
