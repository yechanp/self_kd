from typing import Dict, List, Callable, Union, Any, TypeVar, Tuple
import os
import random
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, Sampler, DataLoader

DATAPATH = 'dataset/'

def make_loader(dataset: str = 'CIFAR100', root: str = DATAPATH, 
                batch_size: int = 128, aug: bool = False, 
                sampler: str = 'none', num_workers: int = 1) -> Tuple[DataLoader, DataLoader]:
    """
    Args:
        dataset: 'CIFAR10', 'CIFAR100', 'CUB200'
        sampler: 'none', 'CS_KD', 'DDGSD'
    """
    if 'CIFAR' in dataset:
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
        
        if dataset == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                    download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                                    download=True, transform=transform_test)
        elif dataset == 'CIFAR100':
            trainset = torchvision.datasets.CIFAR100(root=root, train=True,
                                                    download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(root=root, train=False,
                                                    download=True, transform=transform_test)

    elif 'CUB200' in dataset:
        transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))
        ])
    
        if aug:
            transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                        std=(0.229, 0.224, 0.225))
                ])
        else:
            transform_train = transform_test
        trainset = Cub2011(root=root, train=True,
                           download=True, transform=transform_train)
        testset = Cub2011(root=root, train=False,
                          download=True, transform=transform_test)
    else:
        raise NotImplementedError(f"{dataset} is not available")

    batch_size_test = batch_size
    if sampler == 'DDGSD':
        sampler = DDGSD_Sampler(trainset, batch_size)
        batch_size = 1  # if not 1, error raise
        print("Use DDGSD Sampler")
    elif sampler == 'CS_KD':
        trainset = DatasetWrapper(trainset)
        sampler = PairBatchSampler(trainset, batch_size)
        batch_size = 1  # if not 1, error raise
        print("Use CS_KD Sampler")
    elif sampler == 'none':
        sampler = None
    else:
        raise NotImplementedError(f"{sampler} is not available")
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=(sampler is None),
                             batch_sampler=sampler, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size_test,
                            shuffle=False, num_workers=num_workers)

    return trainloader, testloader

class DDGSD_Sampler(Sampler):
    def __init__(self, dataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            offset = k*self.batch_size
            batch_indices = indices[offset:offset+self.batch_size]
            yield batch_indices + batch_indices

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class Cub2011(Dataset):
    """
    From https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
    """
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root=DATAPATH, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
        self.targets = self.data.target.to_numpy()

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

from collections import defaultdict        
from torchvision import datasets

class DatasetWrapper(Dataset):
    """
    From https://github.com/alinlab/cs-kd/blob/master/datasets.py
    """
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]

class PairBatchSampler(Sampler):
    """
    From https://github.com/alinlab/cs-kd/blob/master/datasets.py
    """
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations

if __name__ == '__main__':
    trainloader, _ = make_loader('CUB200', batch_size=8, aug=True, sampler='DDGSD')
    for _ in trainloader:
        break
    print('finish')