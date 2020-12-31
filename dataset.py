import torch
import torchvision
import torchvision.transforms as transforms

DATAPATH = 'dataset/'

def dataset_cifar(mode, root=DATAPATH, aug=False):
    """
    Args:
        mode : cifar10, cifar100
    """
    
    transform_test  = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                             std=(0.2023, 0.1994, 0.2010))])
    if aug:
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                std=(0.2023, 0.1994, 0.2010))])
    else:
        transform_train = transform_test

    if mode == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                                download=True, transform=transform_test)
    elif mode == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False,
                                                download=True, transform=transform_test)
    else:
        print(f'{mode} mode is not available')
        raise NotImplementedError

    return trainset, testset