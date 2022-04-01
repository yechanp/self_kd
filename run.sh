python main.py --method BaseMethod --backbone resnet18_cifar --dataset CIFAR100 --seed 41 --gpu 0 &
python main.py --method SD_Dropout --backbone resnet18_cifar --dataset CIFAR100 --seed 41 --gpu 0 &
python main.py --method SD_Dropout_v2 --backbone resnet18_cifar --dataset CIFAR100 --seed 41 --gpu 0 &
python main.py --method SD_Dropout_v2 --backbone resnet18_cifar --dataset CIFAR100 --seed 41 --gpu 0 --detach &