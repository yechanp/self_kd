from typing import Callable, Optional
import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets
from torchvision import models
from torchvision.transforms import transforms
from tqdm.autonotebook import tqdm

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal.modelwrapper import ModelWrapper

from torch.utils.data import Subset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from baal.utils.cuda_utils import to_cuda
from models.methods import kl_div_loss

class ModelWrapper_SDD(ModelWrapper):
    def __init__(self, model, criterion, replicate_in_memory=True):
        super().__init__(model, criterion, replicate_in_memory=replicate_in_memory)

    def train_on_batch(
        self, data, target, optimizer, cuda=False, regularizer: Optional[Callable] = None
    ):
        """
        Train the current model on a batch using `optimizer`.

        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.
            optimizer (optim.Optimizer): An optimizer.
            cuda (bool): Use CUDA or not.
            regularizer (Optional[Callable]): The loss regularization for training.


        Returns:
            Tensor, the loss computed from the criterion.
        """

        if cuda:
            data, target = to_cuda(data), to_cuda(target)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        output2 = self.model(data)
        t = 3.0
        loss_kl = (t**2)*kl_div_loss(output, output2, t=t)
        loss += loss_kl

        if regularizer:
            regularized_loss = loss + regularizer()
            regularized_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        self._update_metrics(output, target, loss, filter="train")
        return loss

class TransformAdapter(Subset):
    @property
    def transform(self):
        if hasattr(self.dataset, 'transform'):
            return self.dataset.transform
        else:
            raise AttributeError()

    @transform.setter
    def transform(self, transform):
        if hasattr(self.dataset, 'transform'):
            self.dataset.transform = transform
            
def get_datasets(initial_pool, num_data):
    """
    Let's create a subset of CIFAR10 named CIFAR3, so that we can visualize thing better.

    We will only select the classes airplane, cat and dog.

    Args:
        initial_pool: Amount of labels to start with.

    Returns:
        ActiveLearningDataset, Dataset, the training and test set.
    """
    # airplane, cat, dog
#     classes_to_keep = [0, 3, 5]
    transform = transforms.Compose(
        [
#             transforms.Resize((32, 32)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(30),
            transforms.ToTensor(),
#             transforms.Normalize(3 * [0.5], 3 * [0.5]), 
            transforms.Normalize([0.5], [0.5]), 
        ])
    test_transform = transforms.Compose(
        [
#             transforms.Resize((32, 32)),
            transforms.ToTensor(),
#             transforms.Normalize(3 * [0.5], 3 * [0.5]),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
#     train_ds = datasets.CIFAR10('dataset', train=True,
#                                 transform=transform, target_transform=None, download=True)
    train_ds = datasets.MNIST('dataset', train=True,
                                transform=transform, target_transform=None, download=True)

#     train_mask = np.where([y in classes_to_keep for y in train_ds.targets])[0]
    train_mask = np.arange(len(train_ds))
    train_mask = np.random.choice(train_mask, size=num_data)
    train_ds = TransformAdapter(train_ds, train_mask)

    # In a real application, you will want a validation set here.
#     test_set = datasets.CIFAR10('dataset', train=False,
#                                 transform=test_transform, target_transform=None, download=True)
    test_set = datasets.MNIST('dataset', train=False,
                                transform=test_transform, target_transform=None, download=True)
#     test_mask = np.where([y in classes_to_keep for y in test_set.targets])[0]
    test_mask = np.arange(len(test_set))
    test_mask = np.random.choice(test_mask, size=1000)
    test_set = TransformAdapter(test_set, test_mask)

    # Here we set `pool_specifics`, where we set the transform attribute for the pool.
    active_set = ActiveLearningDataset(train_ds, pool_specifics={'transform': test_transform})

    # We start labeling randomly.
    active_set.label_randomly(initial_pool)
    return active_set, test_set

def get_labels(dataset):
    labels = []
    for data, label in DataLoader(dataset, 32, False, num_workers=4):
        labels.extend(label.numpy().tolist())
    return labels

@dataclass
class ExperimentConfig:
    num_data: int = 1000
    batch_size: int = 16
    initial_pool: int = 100
    query_size: int = 100
    epoch: int = num_data//query_size
    lr: float = 0.1
    heuristic: str = 'bald'
    iterations: int = 40
    training_duration: int = 10
    seed: int = 71
    SDD: bool = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(5*5*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def do_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num) # if use multi-GPU
    # It could be slow
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def active_learning(hyperparams: ExperimentConfig):
    use_cuda = torch.cuda.is_available()
    seed_num = hyperparams.seed
    do_seed(seed_num)
    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    # Get datasets
    active_set, test_set = get_datasets(hyperparams.initial_pool, hyperparams.num_data)

    # Get our model.
    heuristic = get_heuristic(hyperparams.heuristic)
    criterion = CrossEntropyLoss()
    model = patch_module(Net())
    model_active = patch_module(Net())


    if use_cuda:
        model.cuda()
        model_active.cuda()

    # Wraps the model into a usable API.
    if hyperparams.SDD:
        model_active = ModelWrapper_SDD(model_active, criterion)
    else:
        model_active = ModelWrapper(model_active, criterion)
    model = ModelWrapper(model, criterion)
    

    # for ActiveLearningLoop we use a smaller batchsize
    # since we will stack predictions to perform MCDropout.
    active_loop = ActiveLearningLoop(active_set,
                                     model_active.predict_on_dataset,
                                     heuristic,
                                     hyperparams.query_size,
                                     batch_size=1,
                                     iterations=hyperparams.iterations,
                                     use_cuda=use_cuda,
                                     verbose=False)

    # We will reset the weights at each active learning step so we make a copy.
    init_weights = deepcopy(model.state_dict())
    labels_test = get_labels(test_set)
    labelling_progress = active_set._labelled.copy().astype(np.uint16)
    test_losses = []
    test_scores = []
    should_continue = True
    for epoch in tqdm(range(hyperparams.epoch)):
        if not should_continue:
            break
        # Load the initial weights.
        model.load_state_dict(init_weights)
        model_active.load_state_dict(init_weights)
        optimizer = optim.SGD(model.model.parameters(), lr=hyperparams.lr)
        optimizer_active = optim.SGD(model_active.model.parameters(), lr=hyperparams.lr)

        # Train the model on the currently labelled dataset.
        _ = model.train_on_dataset(active_set, optimizer=optimizer, batch_size=hyperparams.batch_size,
                                   use_cuda=use_cuda, epoch=hyperparams.training_duration)

        _ = model_active.train_on_dataset(active_set, optimizer=optimizer_active, batch_size=hyperparams.batch_size,
                                   use_cuda=use_cuda, epoch=hyperparams.training_duration)

        # Get test NLL!
        do_seed(seed_num)
        with torch.no_grad():
    #         model.test_on_dataset(test_set, hyperparams.batch_size, use_cuda,
    #                               average_predictions=hyperparams.iterations)
            output = model.predict_on_dataset(test_set, hyperparams.batch_size, 
                                              hyperparams.iterations, use_cuda, verbose=False)
            
            score = (output.mean(-1).argmax(-1) == np.array(labels_test)).mean()
            test_loss = criterion(input=torch.from_numpy(output.mean(-1)), 
                                  target=torch.from_numpy(np.array(labels_test)))
        metrics = model.metrics

        # We can now label the most uncertain samples according to our heuristic.
        should_continue = active_loop.step()
        # Keep track of progress
        labelling_progress += active_set._labelled.astype(np.uint16)

        logs = {
            "train_nll": metrics['train_loss'].value,
            "test_nll": test_loss.item(),
            "accuracy": score,
            "epoch": epoch+1,
            "Next Training set size": len(active_set)
        }
        print("#"*100)
        print(logs)
        print("#"*100)
    
        test_losses.append(test_loss.item())
        test_scores.append(score)
    return test_losses, test_scores

hyperparams = ExperimentConfig()
hyperparams.heuristic = 'bald'
hyperparams.SDD = True
test_losses, test_scores = active_learning(hyperparams)

hyperparams.heuristic = 'random'
test_losses2, test_scores2 = active_learning(hyperparams)