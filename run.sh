## byot


# byot with dropoutKD
python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.0 --beta 1.0 --seed 41 --gpu 0 & # byot baseline
python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.0 --beta 1.0 --seed 42 --gpu 0 & 
python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.0 --beta 0.1 --seed 41 --gpu 0 &
python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.0 --beta 0.1 --seed 42 --gpu 0 & 

python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 1.0 --seed 41  --gpu 1 &
python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 1.0 --seed 42  --gpu 1 & 
python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 0.1 --seed 41  --gpu 1 &
python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 0.1 --seed 42  --gpu 1 & 

python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 1.0 --seed 41 --detach --gpu 2 &
python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 1.0 --seed 42 --detach --gpu 2 &
python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 0.1 --seed 41 --detach --gpu 2 &
python main.py --method BYOT_Dropout --backbone byot_resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 0.1 --seed 42 --detach --gpu 2 &

## dml
# dml baseline
# dml with dropoutKD
python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.0 --beta 1.0 --seed 41 --gpu 3 &
python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.0 --beta 1.0 --seed 42 --gpu 3 &
python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.0 --beta 0.1 --seed 41 --gpu 3 &
python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.0 --beta 0.1 --seed 42 --gpu 3 &

python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 1.0 --seed 41 --gpu 4 &
python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 1.0 --seed 42 --gpu 4 &
python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 0.1 --seed 41 --gpu 4 &
python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 0.1 --seed 42 --gpu 4 &

python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 1.0 --seed 41 --detach --gpu 5 &
python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 1.0 --seed 42 --detach --gpu 5 &
python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 0.1 --seed 41 --detach --gpu 5 &
python main.py --method DML_Dropout --backbone resnet18_cifar --dataset CIFAR100 --alpha 0.1 --beta 0.1 --seed 42 --detach --gpu 5 &

# intra kl
# sshfs -o allow_other -o uid=`id -u hseo0618` -o gid=`id -g hseo0618` -p 11022 icodelab.asuscomm.com:/home/hseo0618/src/git /data1/home/hseo0618/paper
# sshfs -o allow_other -o uid=`id -u hseo0618` -o gid=`id -g hseo0618` -p 11022 icodelab.asuscomm.com:/home/hseo0618/src/git /home/hseo0618/paper



###################
