import os
# os.chdir('../')
from itertools import product
import torch
import matplotlib.pyplot as pyplot
from dataset import make_loader
from models import methods, resnet
from models.methods import BaseMethod
import numpy as np
from tqdm import tqdm
import pandas as pd


exp_df = pd.DataFrame.from_dict(
    [{'dataset': 'CIFAR100', 'method': 'DML', 'type':    'base', 'path': '/home/hseo0618/src/all_results/hyun/CIFAR100_DML_Dropout_resnet18_cifar__p0.5_t3.0_alpha0.0_beta1.0_B128_seed41_detachFalse'},
     {'dataset': 'CUB200', 'method': 'DML', 'type':   'base',
         'path': '/home/hseo0618/src/all_results/hyun/CUB200_DML_Dropout_resnet18__p0.5_t3.0_alpha0.0_beta1.0_B32_seed41_detachFalse'},
        {'dataset': 'DOG', 'method': 'DML', 'type':   'base',
            'path': '/home/hseo0618/src/all_results/hyun/DOG_DML_Dropout_resnet18__p0.5_t3.0_alpha0.0_beta1.0_B32_seed41_detachFalse'},
     {'dataset': 'CIFAR100', 'method': 'DML', 'type': 'drop',
         'path': '/home/hseo0618/src/all_results/hyun/CIFAR100_DML_Dropout_resnet18_cifar__p0.5_t3.0_alpha0.1_beta1.0_B128_seed41_detachFalse'},
     {'dataset': 'CUB200', 'method': 'DML', 'type':   'drop',
         'path': '/home/hseo0618/src/all_results/hyun/CUB200_DML_Dropout_resnet18__p0.5_t3.0_alpha0.1_beta1.0_B32_seed41_detachFalse'},
     {'dataset': 'DOG', 'method': 'DML', 'type':   'drop',
         'path': '/home/hseo0618/src/all_results/hyun/DOG_DML_Dropout_resnet18__p0.5_t3.0_alpha0.1_beta1.0_B32_seed41_detachFalse'},
     {'dataset': 'CIFAR100', 'method': 'BYOT', 'type':    'base',
         'path': '/home/hseo0618/src/all_results/hyun/CIFAR100_BYOT_Dropout_byot_resnet18_cifar__p0.5_t3.0_alpha0.0_beta0.1_B128_seed41_detachFalse'},
     {'dataset': 'CUB200', 'method': 'BYOT', 'type':  'base',
         'path': '/home/hseo0618/src/all_results/hyun/CUB200_BYOT_Dropout_byot_resnet18__p0.5_t3.0_alpha0.0_beta0.1_B32_seed41_detachFalse'},
     {'dataset': 'DOG', 'method': 'BYOT', 'type':  'base',
         'path': '/home/hseo0618/src/all_results/hyun/DOG_BYOT_Dropout_byot_resnet18__p0.5_t3.0_alpha0.0_beta0.1_B32_seed41_detachFalse'},
     {'dataset': 'CIFAR100', 'method': 'BYOT', 'type':    'drop',
         'path': '/home/hseo0618/src/all_results/hyun/CIFAR100_BYOT_Dropout_byot_resnet18_cifar__p0.5_t3.0_alpha0.1_beta0.1_B128_seed41_detachFalse'},
     {'dataset': 'CUB200', 'method': 'BYOT', 'type':  'drop',
         'path': '/home/hseo0618/src/all_results/hyun/CUB200_BYOT_Dropout_byot_resnet18__p0.5_t3.0_alpha0.1_beta0.1_B32_seed41_detachFalse'},
        {'dataset': 'DOG', 'method': 'BYOT', 'type':  'drop', 'path': '/home/hseo0618/src/all_results/hyun/DOG_BYOT_Dropout_byot_resnet18__p0.5_t3.0_alpha0.1_beta0.1_B32_seed41_detachFalse'}, ]
)
exp_list = list(product(['CIFAR100', 'CUB200', 'DOG'], ['BYOT', 'DML']))
print(exp_list)

print(exp_df)
num_classes = {'CIFAR10': 10, 'CIFAR100': 100, 'CUB200': 200, 'DOG': 120}
METHOD_NAMES = [name for name in methods.__all__
                if not name.startswith('__') and callable(methods.__dict__[name])]
BACKBONE_NAMES = sorted(name for name in resnet.__all__
                        if name.islower() and not name.startswith("__")
                        and callable(resnet.__dict__[name]))
# print(f"{METHOD_NAMES}")
# print(f"{BACKBONE_NAMES}")


# dml byot
class Args():
    num_workers = 1
    beta = 0
    aug = True
    batch_size = 128
    dataset = ''
    method = ''
    backbone = ''
    # byot baseline
    path_base = ""
    # byot with self dropout
    path_drop = ""


args = Args()

result_list = []


for exp in exp_list:
    args.dataset = exp[0]
    args.method = exp[1]
    args.backbone = 'resnet18'
    if args.dataset == 'CIFAR100':
        args.backbone = 'resnet18_cifar'
        if args.method == 'BYOT':
            args.backbone = 'byot_resnet18_cifar'
    else:
        args.backbone = 'resnet18'
        if args.method == 'BYOT':
            args.backbone = 'byot_resnet18'

    args.path_base = exp_df[(exp_df.dataset == exp[0])].loc[(
        exp_df.method == exp[1])].loc[(exp_df.type == 'base')].path.to_list()[0]
    args.path_drop = exp_df[(exp_df.dataset == exp[0])].loc[(
        exp_df.method == exp[1])].loc[(exp_df.type == 'drop')].path.to_list()[0]
    args.path_base = os.path.join(args.path_base, 'checkpoint_last.pth.tar')
    args.path_drop = os.path.join(args.path_drop, 'checkpoint_last.pth.tar')

    print(args.__dict__)

    # 데이터 로더
    trainloader, testloader = make_loader(
        args.dataset, batch_size=args.batch_size, aug=args.aug, num_workers=args.num_workers)

    # 모델 로드
    backbone = resnet.__dict__[args.backbone](
        num_classes=num_classes[args.dataset])
    state_dict = torch.load(args.path_base)
    if args.method == 'DML':
        backbone2 = resnet.__dict__[args.backbone](
            num_classes=num_classes[args.dataset])
        model = methods.__dict__[args.method](args, backbone, backbone2)
    else:
        model = methods.__dict__[args.method](args, backbone)
    model.load_state_dict(state_dict['state_dict'])
    model.cuda()

    # baseline + self dropout
    backbone = resnet.__dict__[args.backbone](
        num_classes=num_classes[args.dataset])
    state_dict_drop = torch.load(args.path_drop)
    if args.method == 'DML':
        backbone2 = resnet.__dict__[args.backbone](
            num_classes=num_classes[args.dataset])
        model_drop = methods.__dict__[args.method](args, backbone, backbone2)
    else:
        model_drop = methods.__dict__[args.method](args, backbone)
    model_drop.load_state_dict(state_dict_drop['state_dict'])
    model_drop.cuda()

    print('model load complete')

    # 그냥 인퍼 acc 측정
    model.eval()
    model_drop.eval()

    pred = []
    pred_drop = []
    Y = []

    with torch.no_grad():
        for x, y in tqdm(testloader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

                outputs = model(x)
                pred.append(torch.softmax(
                    outputs, dim=1).detach().cpu().numpy())
                Y.append(y.detach().cpu().numpy())

                outputs_drop = model_drop(x)
                pred_drop.append(torch.softmax(
                    outputs_drop, dim=1).detach().cpu().numpy())

    Y = np.concatenate(Y)
    pred = np.concatenate(pred)
    pred_drop = np.concatenate(pred_drop)

    acc = (np.argmax(pred, axis=1) == Y)
    confidence = np.max(pred, axis=1)

    acc_drop = (np.argmax(pred_drop, axis=1) == Y)
    confidence_drop = np.max(pred_drop, axis=1)

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np

    from art.attacks.evasion import FastGradientMethod
    from art.estimators.classification import PyTorchClassifier
    import art.metrics as metrics

    # robustness 측정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=tuple(x.shape[1:]),
        nb_classes=num_classes[args.dataset],
    )

    classifier_drop = PyTorchClassifier(
        model=model_drop,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=tuple(x.shape[1:]),
        nb_classes=num_classes[args.dataset],
    )

    Acc_ori = []
    Acc_drop = []

    Acc_ori_attacked = []
    Acc_drop_attacked = []

    for x_test, y_test in tqdm(testloader):
        predictions = classifier.predict(x_test)
        Acc_ori.append(np.argmax(predictions, axis=1) == np.array(y_test))

        attack = FastGradientMethod(estimator=classifier, eps=0.2)
        x_test_adv = attack.generate(x=x_test)
        predictions = classifier.predict(x_test_adv)
        Acc_ori_attacked.append(
            np.argmax(predictions, axis=1) == np.array(y_test))

        predictions = classifier_drop.predict(x_test)
        Acc_drop.append(np.argmax(predictions, axis=1) == np.array(y_test))

        attack = FastGradientMethod(estimator=classifier_drop, eps=0.2)
        x_test_adv = attack.generate(x=x_test)
        predictions = classifier.predict(x_test_adv)
        Acc_drop_attacked.append(
            np.argmax(predictions, axis=1) == np.array(y_test))

    Acc_ori = np.concatenate(Acc_ori)
    Acc_ori_attacked = np.concatenate(Acc_ori_attacked)
    Acc_drop = np.concatenate(Acc_drop)
    Acc_drop_attacked = np.concatenate(Acc_drop_attacked)

    print(f"Acc_ori : {np.sum(Acc_ori)/len(Acc_ori)}")
    print(
        f"Acc_ori_attacked : {np.sum(Acc_ori_attacked)/len(Acc_ori_attacked)}")
    print(f"Acc_drop : {np.sum(Acc_drop)/len(Acc_drop)}")
    print(
        f"Acc_drop_attacked : {np.sum(Acc_drop_attacked)/len(Acc_drop_attacked)}")

    result = {
        'dataset': args.dataset,
        'method': args.method,
        "Acc_ori": np.sum(Acc_ori)/len(Acc_ori),
        "Acc_ori_attacked": np.sum(Acc_ori_attacked)/len(Acc_ori_attacked),
        "Acc_drop": np.sum(Acc_drop)/len(Acc_drop),
        "Acc_drop_attacked": np.sum(Acc_drop_attacked)/len(Acc_drop_attacked)
    }

    result_list.append(result)

results_df = pd.DataFrame().from_dict(result_list)
results_df.to_csv('results.csv')
