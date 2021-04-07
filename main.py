"""
2020-12-30
Hyoje Lee

python main.py --method BaseMethod --backbone resnet34 --seed 1
python main.py --method SelfKD_KL  --backbone resnet34 --seed 1 -p 0.5
python main.py --method CS_KD      --backbone resnet34 --seed 1

"""
# imports base packages
import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# custom packages
from dataset import make_loader
from utils import cal_num_parameters, set_args, do_seed, log_optim, AverageMeter, ProgressMeter, Logger
from models import methods, resnet
from models.methods import method_use_alpha, method_beta_scheduling, method_eta_CosAnealing

METHOD_NAMES = [name for name in methods.__all__
                if not name.startswith('__') and callable(methods.__dict__[name])]
BACKBONE_NAMES = sorted(name for name in resnet.__all__
                        if name.islower() and not name.startswith("__")
                        and callable(resnet.__dict__[name]))

# deal with params
def parser_arg():
    parser = argparse.ArgumentParser()
    ## 
    parser.add_argument('--exp_name', type=str, default='', help="the name of experiment")
    parser.add_argument('-g', '--gpu', type=int, dest='gpu', metavar='N', default=0, help="gpu")
    parser.add_argument('--num_threads', type=int, default=4, metavar='N', help="the number of threads (default: 1)")
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='seed number. if 0, do not fix seed (default: 0)')
    parser.add_argument('--resume', type=str, default='', help='resume path')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset', 
                        choices=['CIFAR10', 'CIFAR100', 'CUB200'])

    ## hyper-parameters
    parser.add_argument('--method', type=str, default='BaseMethod', metavar='METHOD', choices=METHOD_NAMES, help='model_names: '+
                                                                                                           ' | '.join(METHOD_NAMES)+
                                                                                                           ' (defualt: BaseMethod)')
    parser.add_argument('--backbone', type=str, default='resnet18', metavar='BACKBONE', choices=BACKBONE_NAMES, help='Backbone models: '+
                                                                                                                     ' | '.join(BACKBONE_NAMES)+
                                                                                                                     ' (default: resnet18)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help="epoch (default: 200)")
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help="batch size (default: 128)")
    parser.add_argument('-t', type=float, default=3.0, help="temperature (default: 3.0)")
    parser.add_argument('-p', type=float, default=0.5, help="the probability of dropout (default: 0.5)")
    parser.add_argument('-a', '--alpha', type=float, default=0.0, help="the balanced weight between losses (default: 0.0)")
    parser.add_argument('--beta', type=float, default=0.0, help="the weight for KD loss (default: 0.0)")
    parser.add_argument('--eta', type=int, default=0, help="T_max value of Cosine Anealing weight for KD loss (default: 0)")
    parser.add_argument('--woAug', dest='aug', action='store_false', help="data augmentation or not (default: True)")
    parser.add_argument('--lambda_byot', type=float, default=0., help="hyperparams of byot loss")


    ## debug
    # args, _ = parser.parse_known_args('-g 0 --exp_name debug --seed 777 \
    #                                    --backbone resnet18_cifar --method SelfKD_KL --dataset CIFAR100 \
    #                                    --batch_size 128 --num_threads 4'.split())
                                       
    ## real
    args, _ = parser.parse_known_args()

    return set_args(args)

if __name__ == "__main__":
        
    # set environment variables: gpu, num_thread
    args = parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.num_threads)
    # torch.set_num_threads(1)

    # logger
    logger = Logger(args.logfile)
    logger.print_args(args)

    # random seed
    if args.seed:    
        do_seed(args.seed)
        logger.log(f'The fixed seed number is {args.seed}')

    ############### Load Data ###############
    if 'CS_KD' in args.method:
        trainloader, testloader = make_loader(args.dataset, batch_size=args.batch_size, aug=args.aug, 
                                              sampler='CS_KD', num_workers=args.num_threads)
        logger.log('Dataset for Class-wise Self KD')
    elif 'DDGSD' in args.method:
        trainloader, testloader = make_loader(args.dataset, batch_size=args.batch_size, aug=args.aug, 
                                              sampler='DDGSD', num_workers=args.num_threads)
        logger.log('Dataset for Data Distortion Guided Self Distillation')
    else:
        trainloader, testloader = make_loader(args.dataset, batch_size=args.batch_size, aug=args.aug,
                                              num_workers=args.num_threads)
        logger.log('Dataset for the method without sampler')

    ############### Define Model ###############
    print("init neural networks")
    ## construct the model
    num_classes = {'CIFAR10': 10, 'CIFAR100':100, 'CUB200':200}
    backbone = resnet.__dict__[args.backbone](num_classes=num_classes[args.dataset])
    if any(c in args.method for c in ['Base', 'KD', 'SD']):
        if args.alpha:
            model = method_use_alpha(methods.__dict__[args.method])(args, backbone)
        elif args.beta:
            model = method_beta_scheduling(methods.__dict__[args.method])(args, backbone)
        elif args.eta:
            model = method_eta_CosAnealing(methods.__dict__[args.method])(args, backbone)
        else:
            model = methods.__dict__[args.method](args, backbone)
    elif any(c in args.method for c in ['AFD', 'DML']):
        backbone2 = resnet.__dict__[args.backbone](num_classes=num_classes[args.dataset])
        model = methods.__dict__[args.method](args, backbone, backbone2)

    elif any(c in args.method for c in ['BYOT']):
        model = methods.__dict__[args.method](args, backbone)
    else:
        logger.log(f'{args.method} is not available')
        raise NotImplementedError()

    if torch.cuda.is_available():
        model.cuda()

    ## load model
    if args.resume:
        state = torch.load(args.resume)
        epoch_init = state['epoch']
        logger.log(f'Re-Training, Load Model {args.resume}')
        logger.log(f'Load at epoch {epoch_init}')
        model.load_state_dict(state['state_dict'])
        model.optimizer.load_state_dict(state['optimizer'])
        epoch_init += 1
    else:
        epoch_init = 1

    # calculate the number of parameters
    cal_num_parameters(model.parameters(), file=args.logfile)

    ############### Training ###############
    ## log
    log_optim(model.optimizer, model.lr_scheduler, logger)
    log_list = [meter for meter in model.set_log(0, 0)[0].values() if not meter.name == 'Data']
    log_list.append(AverageMeter('Test_Acc', ':.4f'))
    progress = ProgressMeter(args.epochs, meters=log_list, prefix=f'EPOCH')
    writer = SummaryWriter(log_dir=args.tb_folder)
    end = time.time()
    max_acc = 0.0
    for epoch in range(epoch_init, args.epochs+1):
        ## train
        loss_list = model.train_loop(trainloader, epoch=epoch)
        log_list[0].update(time.time() - end)
        
        ## eval
        eval_acc = model.evaluation(testloader)
        
        ## log
        for i, loss in enumerate(loss_list):
            log_list[i+1].update(loss.avg)
            writer.add_scalar(loss.name, loss.avg, epoch)
        lr_name = 'lr'
        for i, opt in enumerate(model.optimizer.optimizers):
            writer.add_scalar(lr_name, opt.param_groups[0]['lr'], epoch)
            lr_name = f'lr_{i+2}'
        log_list[-1].update(eval_acc)
        writer.add_scalar(log_list[-1].name, eval_acc, epoch)

        logger.log(progress.display(epoch), consol=False)

        ## save
        state = {'args' : args,
                'epoch' : epoch,
                'state_dict' : model.state_dict(),
                'optimizer' : model.optimizer.state_dict()}

        if max_acc < eval_acc:
            max_acc = eval_acc
            filename = os.path.join(args.save_folder, 'checkpoint_best.pth.tar')
            logger.log('#'*20+'Save Best Model'+'#'*20)
            torch.save(state, filename)
        
        # if epoch in [49, 50, 99, 100, 149, 150]:
        #     filename = os.path.join(args.save_folder, f'checkpoint_epoch{epoch:03d}.pth.tar')
        #     torch.save(state, filename)
        
        end = time.time()
    filename = os.path.join(args.save_folder, 'checkpoint_last.pth.tar')
    torch.save(state, filename)
    writer.close()