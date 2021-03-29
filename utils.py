from typing import List
import os
import random
import shutil
import numpy as np
from datetime import datetime
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

def current_time(easy=False):
    """
    return : 
        if easy==False, '20190212_070531'
        if easy==True, '2019-02-12 07:05:31'
    """
    now = datetime.now()
    if not easy:
        current_time = '{0.year:04}{0.month:02}{0.day:02}_{0.hour:02}{0.minute:02}{0.second:02}'.format(now)
    else:
        current_time = '{0.year:04}-{0.month:02}-{0.day:02} {0.hour:02}:{0.minute:02}:{0.second:02}'.format(now)

    return current_time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def cal_num_parameters(parameters, file=None):
    """
    Args:
        parameters : model.parameters()
    """
    model_parameters = filter(lambda p: p.requires_grad, parameters)
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'The number of parameters : {num_params/1000000:.2f}M')
    if file is not None:
        with open(file, 'a') as f:
            print(f'The number of parameters : {num_params/1000000:.2f}M', file=f)
    return num_params

class Logger():
    def __init__(self, logfile) -> None:
        self.logfile = logfile
        f = open(self.logfile, 'w')
        f.close()

    def print_args(self, args):
        self.log(f'Strat time : {current_time(easy=True)}')
        for key in args.__dict__.keys():
            self.log(f'{key} : {args.__dict__[key]}')
    
    def log(self, text : str, consol : bool = True) -> None:
        with open(self.logfile, 'a') as f:
            print(text, file=f)
        if consol:
            print(text)

class MultipleOptimizer():
    def __init__(self, op: List[Optimizer]) -> None:
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self) -> List:
        state = []
        for op in self.optimizers:
            state.append(op.state_dict())
        return state
    
    def load_state_dict(self, state: List) -> None:
        assert len(self.optimizers) == len(state), "Lengths must be equal"
        for op, st in zip(self.optimizers, state):
            op.load_state_dict(st)

class MultipleSchedulers():
    def __init__(self, schs: List[MultiStepLR]):
        self.schedulers = schs

    def step(self):
        for sch in self.schedulers:
            sch.step()

def log_optim(optimizers: MultipleOptimizer, schedulers: MultipleSchedulers, logger: Logger) -> None:
    logger.log(optimizers.optimizers)
    for sche in schedulers.schedulers:    
        logger.log("milestones : " + str(dict(sche.milestones)))
        logger.log("gamma : " + str(sche.gamma))

def set_args(args):
    ## experiment name
    args.exp_name = f'{args.dataset}_{args.method}_{args.backbone}_{args.exp_name}'
    if 'Self' in args.method: args.exp_name += f'_p{args.p}'
    if any(c in args.method for c in ['KD', 'SD']): args.exp_name += f'_t{args.t}'
    if args.alpha: args.exp_name += f'_alpha{args.alpha}'
    if args.beta: args.exp_name += f'_beta{args.beta}'
    if args.eta: args.exp_name += f'_CosAnealT{args.eta}'
    args.exp_name += f'_B{args.batch_size}'
    if args.seed: args.exp_name += f'_seed{args.seed}'
    ## directory
    args.save_folder = os.path.join('saved_models', args.exp_name)
    args.tb_folder = os.path.join('tb_results', args.exp_name)
    if os.path.exists(args.save_folder) or os.path.exists(args.tb_folder):
        print(f"Current Experiment is : {args.exp_name}")
        isdelete = input("delete exist exp dir (y/n): ")
        if isdelete == "y":
            if os.path.exists(args.save_folder): shutil.rmtree(args.save_folder) 
            if os.path.exists(args.tb_folder):   shutil.rmtree(args.tb_folder) 
        elif isdelete == "n":
            raise FileExistsError
        else:
            raise FileExistsError

    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(args.tb_folder, exist_ok=True)
    ## logfile
    args.logfile = os.path.join(args.save_folder, 'log.txt')
    
    return args

def do_seed(seed_num, cudnn_ok=True):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num) # if use multi-GPU
    # It could be slow
    if cudnn_ok:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False