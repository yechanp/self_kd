import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import backbones

class dataset_cifar_corrupted(Dataset):
    def __init__(self, data_foler, corruption="brightness") -> None:
        super().__init__()
        self.data_path = os.path.join(data_foler, corruption + ".npy")
        self.label_path = os.path.join(data_foler, "labels.npy")
        self.data = np.load(self.data_path)
        self.labels = np.load(self.label_path)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                  std=(0.2023, 0.1994, 0.2010))])

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.data)

def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data',  type=str, default='',  help="path of dataset")
    parser.add_argument('--path_model', type=str, default='',  help="path of model")
    parser.add_argument('--gpu',        type=str, default='0', help="gpu")

    args, _ = parser.parse_known_args()
    return args

distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]

@ torch.no_grad()
def main():
    args = parser_arg()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    model: nn.Module = backbones.__dict__["resnet18_cifar"](num_classes=100)
    if torch.cuda.is_available():
        model.cuda()

    state = torch.load(os.path.join(args.path_model, "checkpoint_last.pth.tar"))
    state_dict_load = state['state_dict']
    state_dict_new = model.state_dict()
    for key in state_dict_load.keys():
        state_dict_new[key.replace("backbone.", "")] = state_dict_load[key]

    model.load_state_dict(state_dict_new)
    print("Load Model Successfully!")

    acc_list = []

    for distortion in tqdm(distortions, leave=True, position=0):
        dataset = dataset_cifar_corrupted(os.path.join(args.path_data, "CIFAR-100-C"), distortion)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

        model.eval()
        correct = 0
        total = 0

        for x, y in data_loader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            outputs = model.forward(x)
            _, preds = torch.max(outputs, dim=1)
            
            # log
            total += y.size(0)
            correct += (preds == y).sum().item()
        acc = 100*correct/total
        print(f"Corruption : {distortion} - {acc:.2f} %")
        acc_list.append(acc)

    print(acc_list)
    print(f"Mean Corruption Accuracy : {np.mean(acc_list):.2f} %")
    with open(os.path.join(args.path_model, "results_cifar100_C.txt"), "w") as f:
        f.write(str(acc_list))
        f.write(f"\nMean Corruption Accuracy : {np.mean(acc_list):.2f} %")

if __name__ == "__main__":
    main()
        
        