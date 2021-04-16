## byot
# byot with dropoutKD
# byot baseline
# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.0 --beta 1.0 --seed 41 --gpu 6 &&
# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.0 --beta 1.0 --seed 42 --gpu 6 &&
# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.0 --beta 0.1 --seed 41 --gpu 6 &&
# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.0 --beta 0.1 --seed 42 --gpu 6

# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 1.0 --seed 41 --gpu 7 &&
# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 1.0 --seed 42 --gpu 7 && 
# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 0.1 --seed 41 --gpu 7 &&
# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 0.1 --seed 42 --gpu 7 

# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 1.0 --seed 41 --detach --gpu 0 &&
# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 1.0 --seed 42 --detach --gpu 0 &&
# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 0.1 --seed 41 --detach --gpu 0 &&
# python main.py --method BYOT_Dropout --backbone byot_resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 0.1 --seed 42 --detach --gpu 0 
## dml
# dml with dropoutKD
# python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.0 --beta 1.0 --seed 41 --gpu 1 &
# python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.0 --beta 1.0 --seed 42 --gpu 1 &
# python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.0 --beta 0.1 --seed 41 --gpu 1 &
# python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.0 --beta 0.1 --seed 42 --gpu 1 &

# python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 1.0 --seed 41 --gpu 2 &
# python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 1.0 --seed 42 --gpu 2 &
# python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 0.1 --seed 41 --gpu 2 &
# python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 0.1 --seed 42 --gpu 2 &

python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 1.0 --seed 41 --detach --gpu 3 &
python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 1.0 --seed 42 --detach --gpu 3 &
python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 0.1 --seed 41 --detach --gpu 3 &
python main.py --method DML_Dropout --backbone resnet18 --dataset CUB200 --batch_size 32 --alpha 0.1 --beta 0.1 --seed 42 --detach --gpu 3 &

