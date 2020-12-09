"""
2020-12-09
Hyoje Lee

"""
# imports base packages
import os
import time
import random
import argparse
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

import tensorboard_logger as tb_logger

# custom packages
from dataset import dataset_cifar
from utils import cal_num_parameters, log, current_time, AverageMeter, ProgressMeter
from models import models, resnet

method_names = [name for name in models.__all__
                if not name.startswith('__') and callable(models.__dict__[name])]
backbone_names = sorted(name for name in resnet.__all__
                        if name.islower() and not name.startswith("__")
                        and callable(resnet.__dict__[name]))

# deal with params
def add_args(args):
    if not os.path.isdir('saved_models'):
        os.mkdir('saved_models')
    if not os.path.isdir('tb_results'):
        os.mkdir('tb_results')
    args.save_folder = os.path.join('saved_models', args.exp_name)
    args.tb_folder = os.path.join('tb_results', args.exp_name)
    if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)
    if not os.path.isdir(args.tb_folder):
        os.mkdir(args.tb_folder)
    args.logfile = os.path.join(args.save_folder, 'log.txt')
    with open(args.logfile, 'w') as f:
        print(f'Strat time : {current_time(easy=True)}', file=f)
    for key in args.__dict__.keys():
        log(f'{key} : {args.__dict__[key]}', logfile=args.logfile)
    
    return args

def parser_arg():
    parser = argparse.ArgumentParser()
    ## 
    parser.add_argument('--exp_name', type=str, default='debug', help="model_name")
    parser.add_argument('-g', '--gpu', type=int, dest='gpu', default=0, help="gpu")
    parser.add_argument('-l', '--load', type=str, dest='load_model', default='', help='name of model')
    parser.add_argument('-f', '--force', dest='force', action='store_true', help='remove by force (default: False)')
    parser.add_argument('--seed', type=int, default=0, help='seed number. if 0, do not fix seed (default: 0)')

    ## hyper-parameters
    parser.add_argument('--method', type=str, default='BaseMethod', metavar='METHOD', choices=method_names, help='model_names: '+
                                                                                                           ' | '.join(method_names)+
                                                                                                           ' (defualt: BaseMethod)')
    parser.add_argument('--backbone', type=str, default='resnet18', metavar='BACKBONE', choices=backbone_names, help='Backbone models: '+
                                                                                                                     ' | '.join(backbone_names)+
                                                                                                                     ' (default: resnet18)')
    parser.add_argument('--epochs', type=int, default=300, help="epoch (default: 300)")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size (default: 128)")

    ## debug
    # args, _ = parser.parse_known_args('-g 0 --exp_name debug \
    #                                    --backbone resnet18 --method AFD \
    #                                    --batch_size 128'.split())

    ## real
    args, _ = parser.parse_known_args()

    return add_args(args)

# set environment variables: gpu, num_thread
args = parser_arg()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# torch.set_num_threads(1)

# RANDOM SEED
def seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num) # if use multi-GPU
    # It could be slow
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if args.seed:    
    seed(args.seed)
### Step 1: init dataloader
train_dataset, test_dataset = dataset_cifar('cifar100')
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*2,
                                            shuffle=False, num_workers=1)
### Step 2: init neural networks
print("init neural networks")
## construct the model
backbone = resnet.__dict__[args.backbone](num_classes=100)
if args.method in ['BaseMethod', 'SelfKD']:
    model = models.__dict__[args.method](backbone)
elif args.method == 'AFD':
    backbone2 = resnet.__dict__[args.backbone](num_classes=100)
    model = models.AFD(backbone, backbone2)
else:
    print(f'{args.method} is not available')
if torch.cuda.is_available():
    model.cuda()

# calculate the number of parameters
cal_num_parameters(model.parameters(), file=args.logfile)

############### Training ###############
## log
log_list = [meter for meter in model.set_log(0, 0)[0] if not meter.name == 'Data']
log_list.append(AverageMeter('Test_Acc', ':.4f'))
progress = ProgressMeter(args.epochs, meters=log_list, prefix=f'EPOCH')
logger = tb_logger.Logger(logdir=args.tb_folder)
end = time.time()
max_acc = 0.0
for epoch in range(1, args.epochs+1):
    ## train
    loss_list = model.train_loop(trainloader, epoch=epoch)
    log_list[0].update(time.time() - end)
    
    ## eval
    eval_acc = model.evaluation(testloader)
    
    ## log
    for i, loss in enumerate(loss_list):
        log_list[i+1].update(loss.avg)
        logger.log_value(loss.name, loss.avg, epoch)
    lr_name = 'lr'
    for i, opt in enumerate(model.optimizer.optimizers):
        logger.log_value(lr_name, opt.param_groups[0]['lr'], epoch)
        lr_name += f'_{i+2}'
    log_list[-1].update(eval_acc)
    logger.log_value(log_list[-1].name, eval_acc, epoch)

    log(progress.display(epoch), args.logfile, consol=False)

    ## save
    state = {'args' : args,
             'epoch' : epoch,
             'state_dict' : model.state_dict(),}
            #  'optimizer' : model.optimizer.state_dict()}

    if max_acc < eval_acc:
        max_acc = eval_acc
        filename = os.path.join(args.save_folder, 'checkpoint_best.pth.tar')
        log('#'*20+'Save Best Model'+'#'*20, args.logfile)
        torch.save(state, filename)