#-------------------------------------
# 
#-------------------------------------
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
import time
from utils import AverageMeter, ProgressMeter

class BaseTempelet(nn.Module):

    def __init__(self, backbone):
        super(BaseTempelet, self).__init__()
        self.backbone = backbone
        self.set_optimizer()
        
    def set_log(self, epoch, num_batch):
        batch_time = AverageMeter('Time', ':.3f')
        data_time = AverageMeter('Data', ':.3f')
        losses = AverageMeter('Loss', ':.4f')
        
        progress = ProgressMeter(num_batch,
                                [batch_time, data_time, losses],
                                prefix=f'Epoch[{epoch}] Batch')
        
        return batch_time, data_time, losses, progress

    def set_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 225], gamma=0.1)
        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)

    def forward(self, x):
        return self.backbone(x)
    
    def train_loop(self, dataloader, epoch, freq=10):
        self.train()
        batch_time, data_time, losses, progress = self.set_log(epoch, len(dataloader))
        end = time.time()
        for i, (x, y) in enumerate(dataloader):
            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            outputs = self.backbone(x)
            loss = self.criterion(outputs, y)
            
            ############### Backpropagation ###############
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ###############################################

            # log
            losses.update(loss.item(), x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (i%freq) == 0:
                progress.display(i)
        return losses.avg

    @torch.no_grad()
    def evaluation(self, dataloader):
        self.eval()
        correct = 0
        total = 0
        for i, (x, y) in enumerate(dataloader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            outputs = self.backbone(x)
            _, preds = torch.max(outputs, dim=1)
            
            # log
            total += y.size(0)
            correct += (preds == y).sum().item()
        # print(f'Acc {100*correct/total:.4f}')
        return 100*correct/total
        
class AFD(BaseTempelet):
    def __init__(self, backbone1, backbone2):
        super(BaseTempelet, self).__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.set_optimizer()

    def forward(self, x):
        return self.backbone1(x)
    
    def train_loop(self, dataloader, epoch, freq=10):
        self.train()
        batch_time, data_time, losses, progress = self.set_log(epoch, len(dataloader))
        end = time.time()
        for i, (x, y) in enumerate(dataloader):
            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            outputs = self.backbone(x)
            loss = self.criterion(outputs, y)
            
            ############### Backpropagation ###############
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ###############################################

            # log
            losses.update(loss.item(), x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (i%freq) == 0:
                progress.display(i)
        return losses.avg
    
