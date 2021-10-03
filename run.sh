# screen -d -m -S exp1 bash -c 'source activate torch17 ; python main.py --exp_name Adam_check_init_4 --method SD_Dropout_uncertainty --seed 41 &&
#                                                         python main.py --exp_name Adam_check_init_4 --method SD_Dropout_uncertainty --detach --seed 41' &
# screen -d -m -S exp3 bash -c 'source activate torch17 ; python main.py --exp_name Adam_check --seed 41 &&
#                                                         python main.py --exp_name Adam_check_wd --method CS_KD --seed 41' &

# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BaseMethod              --backbone resnet18  --optim adam --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout_uncertainty  --backbone resnet18  --optim adam --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout              --backbone resnet18  --optim adam --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method CS_KD                   --backbone resnet18  --optim adam --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DDGSD                   --backbone resnet18  --optim adam --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BYOT                    --backbone resnet18  --optim adam --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DML                     --backbone resnet18  --optim adam --wd 1e-4 --detach &&

# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BaseMethod              --backbone resnet18  --optim sgd  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout_uncertainty  --backbone resnet18  --optim sgd  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout              --backbone resnet18  --optim sgd  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method CS_KD                   --backbone resnet18  --optim sgd  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DDGSD                   --backbone resnet18  --optim sgd  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BYOT                    --backbone resnet18  --optim sgd  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DML                     --backbone resnet18  --optim sgd  --wd 1e-4 --detach &&

# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BaseMethod              --backbone resnet18  --optim sgd  --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout_uncertainty  --backbone resnet18  --optim sgd  --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout              --backbone resnet18  --optim sgd  --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method CS_KD                   --backbone resnet18  --optim sgd  --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DDGSD                   --backbone resnet18  --optim sgd  --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BYOT                    --backbone resnet18  --optim sgd  --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DML                     --backbone resnet18  --optim sgd  --wd 5e-4 --detach

# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BaseMethod              --backbone resnet18  --optim adam  --lr 0.001   --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout_uncertainty  --backbone resnet18  --optim adam  --lr 0.001   --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout              --backbone resnet18  --optim adam  --lr 0.001   --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method CS_KD                   --backbone resnet18  --optim adam  --lr 0.001   --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DDGSD                   --backbone resnet18  --optim adam  --lr 0.001   --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BYOT                    --backbone resnet18  --optim adam  --lr 0.001   --wd 5e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DML                     --backbone resnet18  --optim adam  --lr 0.001   --wd 5e-4 --detach &&

# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BaseMethod              --backbone resnet18  --optim adam  --lr 0.0001  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout_uncertainty  --backbone resnet18  --optim adam  --lr 0.0001  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout              --backbone resnet18  --optim adam  --lr 0.0001  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method CS_KD                   --backbone resnet18  --optim adam  --lr 0.0001  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DDGSD                   --backbone resnet18  --optim adam  --lr 0.0001  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BYOT                    --backbone resnet18  --optim adam  --lr 0.0001  --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DML                     --backbone resnet18  --optim adam  --lr 0.0001  --wd 1e-4 --detach &&

# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BaseMethod              --backbone resnet18  --optim adam  --lr 0.01    --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout_uncertainty  --backbone resnet18  --optim adam  --lr 0.01    --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method SD_Dropout              --backbone resnet18  --optim adam  --lr 0.01    --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method CS_KD                   --backbone resnet18  --optim adam  --lr 0.01    --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DDGSD                   --backbone resnet18  --optim adam  --lr 0.01    --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method BYOT                    --backbone resnet18  --optim adam  --lr 0.01    --wd 1e-4 --detach &&
# python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100   --method DML                     --backbone resnet18  --optim adam  --lr 0.01    --wd 1e-4 --detach

python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100 --method SD_Dropout_uncertainty --backbone resnet18 --optim sgd --lr 0.1 --wd 0.0001 --detach  --init_var 4.0 --init_var_2 -1.0 --wd_only_log 1.0 &&
python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100 --method SD_Dropout_uncertainty --backbone resnet18 --optim sgd --lr 0.1 --wd 0.0001 --detach  --init_var 6.0 --init_var_2 -1.0 --wd_only_log 1.0 &&
python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100 --method SD_Dropout_uncertainty --backbone resnet18 --optim sgd --lr 0.1 --wd 0.0001 --detach  --init_var 4.0 --init_var_2 -1.0 --wd_only_log 0.5 &&
python main.py --exp_name Exp --seed 41 --num_workers 4 --dataset CIFAR100 --method SD_Dropout_uncertainty --backbone resnet18 --optim sgd --lr 0.1 --wd 0.0001 --detach  --init_var 6.0 --init_var_2 -1.0 --wd_only_log 0.5