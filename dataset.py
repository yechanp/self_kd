from typing import Dict, List, Callable, Union, Any, TypeVar, Tuple
import os
from os.path import join
import random
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, Sampler, DataLoader
from collections import defaultdict
from PIL import Image
import scipy

DATAPATH = 'dataset/'

def make_loader(dataset: str = 'CIFAR100', root: str = DATAPATH, 
                batch_size: int = 128, aug: bool = False, 
                sampler: str = 'none', num_workers: int = 1) -> Tuple[DataLoader, DataLoader]:
    """
    Args:
        dataset: 'CIFAR10', 'CIFAR100', 'CUB200', 'DOG'
        sampler: 'none', 'CS_KD', 'DDGSD'
    """
    if any(dname == dataset for dname in ['CIFAR10', 'CIFAR100']):
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

    elif any(dname == dataset for dname in ['CUB200', 'DOG']):
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
        if dataset == 'CUB200':
            trainset = Cub2011(root=root, train=True,
                            download=True, transform=transform_train)
            testset = Cub2011(root=root, train=False,
                            download=True, transform=transform_test)
        elif dataset == 'DOG':
            trainset =StanfordDogs(root=root, train=True,
                            download=False, transform=transform_train)
            testset = StanfordDogs(root=root, train=False,
                            download=False, transform=transform_test)
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
            if isinstance(self.base_dataset, torchvision.datasets.ImageFolder):
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

class StanfordDogs(Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again. Shih-Tzu
    """

    folder = 'stanfordDogs/cropped/cropped'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self,
                 root: str = DATAPATH, 
                 train: bool = True, download: bool = False,
                 transform: transforms.Compose = None):
        
        self.root = join(root, self.folder)

        # self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        # self.cropped = cropped
        self.transform = transform
        # self.target_transform = target_transform

        if download:
            self.download()
        if self.train:
            self.files_target = join(self.root,'train')
        else:
            self.files_target = join(self.root,'test')
        self.classes = ['-'.join(item.split('-')[1:]).lower() for item in os.listdir(self.files_target) ] 
        self.cls_dir = {item.lower(): idx for (idx,item) in enumerate(self.classes) }    
            
            
        self._flat_breed_images     = []
        self._flat_breed_annotations= []
        for cls in os.listdir(self.files_target):
            cls_real     = '-'.join(cls.split('-')[1:]).lower()
            
            # print('cls_real',cls_real)
            # print('self.cls_dir' , self.cls_dir)
            
            cls_real_idx = self.cls_dir[cls_real]
            self._flat_breed_images      += [ join(self.files_target,cls,item) for item in  os.listdir(join(self.files_target,cls)) ] 
            self._flat_breed_annotations += [ cls_real_idx                 for item in  os.listdir(join(self.files_target,cls)) ]
            
        assert len(self._flat_breed_images ) == len(self._flat_breed_annotations)
        self.targets = self._flat_breed_annotations

        # split = self.load_split()

        # self.images_folder = join(self.root, 'Images')
        # self.annotations_folder = join(self.root, 'Annotation')
        # self._breeds = os.list_dir(self.images_folder)

        # if self.cropped:
            # self._breed_annotations = [[(annotation, box, idx)
                                        # for box in self.get_boxes(join(self.annotations_folder, annotation))]
                                        # for annotation, idx in split]
            # self._flat_breed_annotations = sum(self._breed_annotations, [])

            # self._flat_breed_images = [(annotation+'.jpg', idx) for annotation, box, idx in self._flat_breed_annotations]
        # else:
            # self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in split]

            # self._flat_breed_images = self._breed_images

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        # image_name, target_class = self._flat_breed_images[index]
        # image_path = join(self.images_folder, image_name)
        # image = Image.open(image_path).convert('RGB')

        # if self.cropped:
            # image = image.crop(self._flat_breed_annotations[index][1])
        image_path   = self._flat_breed_images[index]
        target_class = self._flat_breed_annotations[index]
        
        image        = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            # print('do transform')

        # if self.target_transform:
            # target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self._flat_breed_images), len(counts.keys()), float(len(self._flat_breed_images))/float(len(counts.keys()))))

        return counts
        
if __name__ == '__main__':
    trainloader, _ = make_loader('DOG', batch_size=8, aug=True, sampler='none')
    for _ in trainloader:
        break
    print('finish')