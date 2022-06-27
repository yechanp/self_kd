exp_name=$1
method=$2
gpu=$3
lam_sdd=$4
lam_kd=$5

python main.py --exp_name $exp_name --method $method --seed 201 --gpu $gpu --lam_sdd $lam_sdd --lam_kd $lam_kd &&
python main.py --exp_name $exp_name --method $method --seed 202 --gpu $gpu --lam_sdd $lam_sdd --lam_kd $lam_kd &&
python main.py --exp_name $exp_name --method $method --seed 203 --gpu $gpu --lam_sdd $lam_sdd --lam_kd $lam_kd &&
python main.py --exp_name $exp_name --method $method --seed 204 --gpu $gpu --lam_sdd $lam_sdd --lam_kd $lam_kd &&
python main.py --exp_name $exp_name --method $method --seed 205 --gpu $gpu --lam_sdd $lam_sdd --lam_kd $lam_kd &&
python main.py --exp_name $exp_name --method $method --seed 206 --gpu $gpu --lam_sdd $lam_sdd --lam_kd $lam_kd &&
python main.py --exp_name $exp_name --method $method --seed 207 --gpu $gpu --lam_sdd $lam_sdd --lam_kd $lam_kd &&
python main.py --exp_name $exp_name --method $method --seed 208 --gpu $gpu --lam_sdd $lam_sdd --lam_kd $lam_kd &&
python main.py --exp_name $exp_name --method $method --seed 209 --gpu $gpu --lam_sdd $lam_sdd --lam_kd $lam_kd &&
python main.py --exp_name $exp_name --method $method --seed 210 --gpu $gpu --lam_sdd $lam_sdd --lam_kd $lam_kd 