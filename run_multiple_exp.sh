exp_name=$1
seed=$2
dataset=$3
backbone=$4
optim=$5
lr=$6
wd=$7

python main.py --num_workers 4 --method BaseMethod             --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd --detach &&
python main.py --num_workers 4 --method SD_Dropout_uncertainty --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd --detach &&
python main.py --num_workers 4 --method SD_Dropout             --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd --detach &&
python main.py --num_workers 4 --method CS_KD                  --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd --detach &&
python main.py --num_workers 4 --method DDGSD                  --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd --detach &&
python main.py --num_workers 4 --method BYOT                   --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd --detach &&
python main.py --num_workers 4 --method DML                    --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd --detach