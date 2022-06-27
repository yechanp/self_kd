"""
2022-04-01
Hyoje Lee

python main.py --method BaseMethod      --backbone resnet18_cifar --seed 41
python main.py --method CS_KD           --backbone resnet18_cifar --seed 41
python main.py --method CS_KD_Dropout   --backbone resnet18_cifar --seed 41

"""
# imports base packages
import os, sys
import time
import argparse
from typing import Dict, Tuple
import torch
from torch.utils.tensorboard import SummaryWriter

# custom packages
from dataset import make_loader
from utils.utils import cal_num_parameters, do_seed, log_optim, AverageMeter, ProgressMeter, Logger
from utils.config import Config
from models import methods
import backbones

METHOD_NAMES = [name for name in methods.__all__
                if not name.startswith('__') and callable(methods.__dict__[name])]
BACKBONE_NAMES = sorted(name for name in backbones.__all__
                        if name.islower() and not name.startswith("__")
                        and callable(backbones.__dict__[name]))


def set_log(epochs: int, log_names=None) -> Tuple[Dict[str, AverageMeter], ProgressMeter]:
    meters = {}
    meters['epoch_time'] = AverageMeter('Time', ':.3f')
    meters['test_acc'] = AverageMeter('Test_Acc', ':.4f')
    if log_names is not None:
        for key in log_names:
            if key in ['data_time', 'batch_time']: continue
            meters[key] = AverageMeter(key, ':.4f')

    progress = ProgressMeter(epochs, 
                             meters.values(),
                             prefix=f'EPOCH')
    return meters, progress

def update_log(loss_meters: Dict[str, AverageMeter], 
               meters: Dict[str, AverageMeter], 
               progress: ProgressMeter, 
               writer: SummaryWriter,
               end) -> Tuple[Dict[str, AverageMeter], ProgressMeter]:
    if len(meters.keys()) != (len(loss_meters.keys())+2):
        meters, progress = set_log(progress.num_batchs, log_names=loss_meters.keys())
    meters['epoch_time'].update(time.time() - end)
    meters['test_acc'].update(eval_acc)
    for key in loss_meters.keys():
        loss = loss_meters[key]
        meters[key].update(loss.avg)
        writer.add_scalar(loss.name, loss.avg, epoch)
    lr_name = 'lr'
    for i, opt in enumerate(model.optimizer.optimizers):
        writer.add_scalar(lr_name, opt.param_groups[0]['lr'], epoch)
        lr_name = f'lr_{i+2}'

    writer.add_scalar(meters['test_acc'].name, eval_acc, epoch)

    return meters, progress

# deal with params
def parser_arg() -> Config:
    parser = argparse.ArgumentParser()
    ## 
    parser.add_argument('--exp_name', type=str, default='', help="the name of experiment")
    parser.add_argument('-g', '--gpu', type=int, dest='gpu', metavar='N', default=0, help="gpu")
    parser.add_argument('--num_workers', type=int, default=1, metavar='N', help="the number of workers in dataloader (default: 1)")
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='seed number. if 0, do not fix seed (default: 0)')
    parser.add_argument('--resume', type=str, default='', help='resume path')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset', 
                        choices=['CIFAR10', 'CIFAR100', 'CUB200', 'DOG'])

    ## hyper-parameters
    parser.add_argument('--method', type=str, default='BaseMethod', metavar='METHOD', choices=METHOD_NAMES, help='model_names: '+
                                                                                                           ' | '.join(METHOD_NAMES)+
                                                                                                           ' (defualt: BaseMethod)')
    parser.add_argument('--backbone', type=str, default='resnet18_cifar', metavar='BACKBONE', choices=BACKBONE_NAMES, help='Backbone models: '+
                                                                                                                     ' | '.join(BACKBONE_NAMES)+
                                                                                                                     ' (default: resnet18)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help="epoch (default: 200)")
    parser.add_argument('-t', type=float, default=3.0, help="temperature (default: 3.0)")
    parser.add_argument('-p', type=float, default=0.5, help="the probability of dropout (default: 0.5)")
    parser.add_argument('--lam_sdd', type=float, default=0.1, help="the weight for SD_Dropout loss (default: 0.1)")
    parser.add_argument('--lam_kd', type=float, default=1.0, help="the weight for the method loss (default: 1.0)")
    parser.add_argument('--detach', dest='detach', action='store_true', help="detach or not when calculate KL loss using Dropout (default: False)")
    parser.add_argument('--woAug', dest='aug', action='store_false', help="data augmentation or not (default: True)")
    parser.add_argument('-f', '--force', dest='force', action='store_true', help="Run by force (default: False)")

    args, _ = parser.parse_known_args()

    return Config(args)

if __name__ == "__main__":
        
    # set environment variables: gpu, num_thread
    args = parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # torch.set_num_threads(1)

    # logger
    logger = Logger(args.logfile)
    logger(' '.join(sys.argv))
    logger.print_args(args)

    # random seed
    if args.seed:    
        do_seed(args.seed)
        logger(f'The fixed seed number is {args.seed}')

    ############### Load Data ###############
    if 'CS_KD' in args.method:
        trainloader, testloader = make_loader(args.dataset, batch_size=args.batch_size, aug=args.aug, 
                                              sampler='CS_KD', num_workers=args.num_workers)
        logger('Dataset for Class-wise Self KD')
    elif 'DDGSD' in args.method:
        trainloader, testloader = make_loader(args.dataset, batch_size=args.batch_size, aug=args.aug, 
                                              sampler='DDGSD', num_workers=args.num_workers)
        logger('Dataset for Data Distortion Guided Self Distillation')
    else:
        trainloader, testloader = make_loader(args.dataset, batch_size=args.batch_size, aug=args.aug,
                                              num_workers=args.num_workers)
        logger('Dataset for the method without sampler')

    ############### Define Model ###############
    print("init neural networks")
    ## construct the model
    num_classes = {'CIFAR10': 10, 'CIFAR100':100, 'CUB200':200, 'DOG':120}
    backbone = backbones.__dict__[args.backbone](num_classes=num_classes[args.dataset])
    model: methods.BaseMethod   # type hint
    if any(c in args.method for c in ['Base', 'KD', 'SD', 'BYOT']):
        model = methods.__dict__[args.method](args, backbone)
    elif any(c in args.method for c in ['DML']):
        backbone2 = backbones.__dict__[args.backbone](num_classes=num_classes[args.dataset])
        backbone2.cuda()
        model = methods.__dict__[args.method](args, backbone, backbone2)
    else:
        logger(f'{args.method} is not available')
        raise NotImplementedError()

    if torch.cuda.is_available():
        model.cuda()

    ## load model
    if args.resume:
        state = torch.load(args.resume)
        epoch_init = state['epoch']
        logger(f'Re-Training, Load Model {args.resume}')
        logger(f'Load at epoch {epoch_init}')
        model.load_state_dict(state['state_dict'])
        model.optimizer.load_state_dict(state['optimizer'])
        epoch_init += 1
    else:
        epoch_init = 1

    # calculate the number of parameters
    cal_num_parameters(model.parameters(), file=args.logfile)

    ############### Training ###############
    # log optimizer informations
    log_optim(model.optimizer, model.lr_scheduler, logger)
    meters, progress = set_log(args.epochs)
    writer = SummaryWriter(log_dir=args.tb_folder)  # tensorboard writer
    end = time.time()
    max_acc = 0.0
    for epoch in range(epoch_init, args.epochs+1):
        ## train
        loss_meters = model.train_loop(trainloader, epoch=epoch)
        
        ## eval
        eval_acc = model.evaluation(testloader)
        
        ## log
        meters, progress = update_log(loss_meters, meters, progress, writer, end)
        logger(progress.display(epoch), consol=False)

        ## save
        state = {'args' : args,
                'epoch' : epoch,
                'state_dict' : model.state_dict(),
                'optimizer' : model.optimizer.state_dict()}

        if max_acc < eval_acc:
            max_acc = eval_acc
            filename = os.path.join(args.save_folder, 'checkpoint_best.pth.tar')
            logger('#'*20+'Save Best Model'+'#'*20)
            # torch.save(state, filename)
        
        end = time.time()
    filename = os.path.join(args.save_folder, 'checkpoint_last.pth.tar')
    torch.save(state, filename)
    writer.close()