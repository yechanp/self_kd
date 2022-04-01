# Self Knowledge Distillation
2022-04-01

## /models/method.py

KL Divergence Loss는 **kl_div_loss**라는 funtion으로 구현.

기본적으로 BaseMethod class를 상속받아서 method를 구현하면 됩니다.
상속 후, 수정할 부분은 init과 calculate_loss 입니다.
calculate_loss 에서 return 값은 기본적으로 2개 입니다.
1. loss 값들의 Dict. 최종 loss로 'loss_total'이 반드시 필요합니다. 
예) {'loss_ce': loss_ce, 'loss_kl': loss_kl, 'loss_total': loss_ce+loss_kl}
2. 추후에 SD_Dropout을 하기 위해, 미리 마지막 layer의 feature를 return.
output, feats = self.backbone(x, return_feat=True)로 feats를 추출한 후
make_feature_vector(feats[-1])을 이용하여 마지막 layer의 feature를 구해서 return 하면 됩니다.

CS_KD_Dropout, DDGSD_Dropout를 참고하세요


## /main.py

```
python main.py --method BaseMethod      --dataset CUB200   --backbone resnet18  --seed 41
python main.py --method CS_KD           --dataset CIFAR100 --backbone resnet18_cifar --batch_size 128 --beta 1.0 
python main.py --method CS_KD_Dropout   --dataset CIFAR100 --backbone resnet18_cifar --batch_size 128 --beta 1.0 --alpha 0.1
```

## Dataset
```
--dataset [CIFAR100, CUB200]
```

## Method
```
--method [BaseMethod, CS_Kd, CS_Kd_Dropout, DDGSD, DDGSD_Dropout]
```

## Backbone
CIFAR dataset의 경우, **반드시 resnet18_cifar 사용**.
```
--backbone [resnet18, resnet18_cifar]
```

## Loss with lam_sdd, lam_kd
기본적으로   
```
loss = ce_loss + lam_sdd*SD_Dropout_loss + lam_kd*Method_loss    
```
입니다.

CS_KD_Dropout의 경우, lam_sdd=0.1, lam_kd=1.0으로   
loss = ce_loss + 0.1\*SD_Dropout_loss + 1.0\*Method_loss    
입니다.