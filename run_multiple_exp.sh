exp_name=$1
seed=$2
dataset=$3
backbone=$4
optim=$5
lr=$6
wd=$7
init_var_sd=$8
wd_only_log=$9

python main.py --num_workers 4 --method BaseMethod             --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd &&
python main.py --num_workers 4 --method SD_Dropout_uncertainty --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd --init_var_sd $init_var_sd --wd_only_log $wd_only_log --detach &&
python main.py --num_workers 4 --method SD_Dropout             --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd &&
python main.py --num_workers 4 --method CS_KD                  --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd &&
python main.py --num_workers 4 --method DDGSD                  --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd &&
python main.py --num_workers 4 --method BYOT                   --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd &&
python main.py --num_workers 4 --method DML                    --exp_name $exp_name --seed $seed --dataset $dataset --backbone $backbone --optim $optim --lr $lr --wd $wd