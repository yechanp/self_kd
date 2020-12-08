import torch
import torchvision
import torchvision.transforms as transforms

DATAPATH = '/home/hyoje/pyfiles/datasets/'

def dataset_cifar(mode, transform=None, root=DATAPATH):
    """
    Args:
        mode : cifar10, cifar100
    """

    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

    if mode == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                                download=True, transform=transform)
    elif mode == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=root, train=False,
                                                download=True, transform=transform)
    else:
        print(f'{mode} mode is not available')

    return trainset, testset