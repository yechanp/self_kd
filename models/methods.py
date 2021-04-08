#-------------------------------------
# 기본적으로 BaseMethod class를 상속받아서 method를 구현하면 됩니다.
# 상속 후, 수정할 부분은 init과 calculate_loss 입니다.
# calculate_loss 에서 return 값은 기본적으로 2개 입니다.
# 1. loss 값들의 Dict. 최종 loss로 'loss_total'이 반드시 필요합니다.
# 2. 추후에 SD_Dropout을 하기 위해, 미리 마지막 layer의 feature를 return.
# output, feats = self.backbone(x, return_feat=True)로 feats를 추출한 후
# make_feature_vector(feats[-])을 이용하여 마지막 layer의 feature를 구해서 return 하면 됩니다.
# 
# CS_KD_Dropout, DDGSD_Dropout를 참고하세요
# 
# BYOT는 서현 님이 작성하여 위의 method들과 구조가 다릅니다. 곧 통일해보겠습니다.
# 
# 2021-04-08 by Hyoje
#-------------------------------------

from typing import Dict, List, Callable, Union, Any, Tuple
from torch import Tensor
from torch.nn import Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import re
import math
from utils import AverageMeter, ProgressMeter, MultipleOptimizer, MultipleSchedulers

__all__ = ['BaseMethod', 'SD_Dropout', 
           'CS_KD', 'DDGSD', 'BYOT',
           'CS_KD_Dropout', 'DDGSD_Dropout']

def kl_div_loss(pred: Tensor, target: Tensor, t: float = 3.0) -> Tensor:
    x_log = F.log_softmax(pred/t, dim=1)
    y = F.softmax(target/t, dim=1)
    
    return F.kl_div(x_log, y, reduction='batchmean')

def make_output(net, feat: Tensor) -> Tensor:
    """
    make output of feature
    feat: (B x C x H x W)
    """
    out = net.avgpool(feat)
    out = torch.flatten(out, 1)
    out = net.fc(out)

    return out

def make_feature_vector(x: Tensor) -> Tensor:
    """
    Use Global Average Pooling.
    Args:
        x: Tensor. (B x C x H x W)
    return:
        out: Tensor. (B x C)
    """
    out = F.adaptive_avg_pool2d(x, output_size=(1, 1))
    out = torch.flatten(out, 1)

    return out

################ BASE MODEL ################
class BaseMethod(nn.Module):
    """
    """
    def __init__(self, args, backbone: nn.Module) -> None:
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.set_optimizer()
        self.meters: Dict[str, AverageMeter]
        self.progress: ProgressMeter
        
    def set_log(self, epoch: int, num_batchs: int, log_names=None):
        self.meters = {}
        self.meters['batch_time'] = AverageMeter('Time', ':.3f')
        self.meters['data_time'] = AverageMeter('Data', ':.3f')
        self.meters['loss_total'] = AverageMeter('loss_total', ':.4f')
        if log_names is not None:
            for key in log_names:
                self.meters[key] = AverageMeter(key, ':.4f')

        self.progress = ProgressMeter(num_batchs, 
                                      self.meters.values(),
                                      prefix=f'Epoch[{epoch}] Batch')

    def update_log(self, 
                   losses: Dict[str, Tensor], 
                   size: int, end):
        for key in losses.keys():
            self.meters[key].update(losses[key].item(), size)
        self.meters['batch_time'].update(time.time() - end)

    def set_optimizer(self) -> None:
        self.criterion_ce = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        self.optimizer = MultipleOptimizer([optimizer])
        self.lr_scheduler = MultipleSchedulers([lr_scheduler])

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
    
    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Any]:
        """
        Args:
            x, y: Tensor.
        Return:
            losses: Dict
            feature_vectors: Tensor
        """
        outputs = self.forward(x)
        loss_ce = self.criterion_ce(outputs, y)
        loss_total = loss_ce

        return {'loss_ce':loss_ce, 'loss_total':loss_total}, None

    def update_optimizer(self, losses: Dict[str, Tensor]) -> None:
        assert 'loss_total' in losses.keys(), "loss_total needs"
        loss = losses['loss_total']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_loop(self, dataloader, epoch: int, freq: int = 10) -> Dict[str, AverageMeter]:
        self.train()
        self.set_log(epoch, len(dataloader))
        end = time.time()
        for i, (x, y) in enumerate(dataloader):
            self.meters['data_time'].update(time.time() - end)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            
            # calculate loss
            losses, _ = self.calculate_loss(x, y)
            
            # back-propagation
            self.update_optimizer(losses)

            # log
            if len(self.meters.keys()) != (len(losses.keys())+2):
                self.set_log(epoch, len(dataloader), log_names=losses.keys())
            self.update_log(losses, x.size(0), end)
            end = time.time()

            if (i%freq) == 0:
                self.progress.display(i)

        ## lr schedulder
        self.lr_scheduler.step()
        del self.meters['data_time'], self.meters['batch_time']
        return self.meters

    @torch.no_grad()
    def evaluation(self, dataloader) -> float:
        self.eval()
        correct = 0
        total = 0
        for x, y in dataloader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            outputs = self.forward(x)
            _, preds = torch.max(outputs, dim=1)
            
            # log
            total += y.size(0)
            correct += (preds == y).sum().item()
        # print(f'Acc {100*correct/total:.4f}')
        return 100*correct/total

class SD_Dropout(BaseMethod):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.T = args.t
        self.P = args.p

    def calculate_loss(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        # forward
        output_wo_dropout, feats = self.backbone(x, return_feat=True)
        
        # cross-entropy loss
        loss_ce = self.criterion_ce(output_wo_dropout, y)
        
        # dropout only last feature
        feature_vector = make_feature_vector(feats[-1])
        feats_dp = [F.dropout(feature_vector, p=self.P) for _ in range(2)]
        output_dp1, output_dp2 = [self.backbone.fc(feats_dp[j]) for j in range(2)]
        
        # KL Divergence
        loss_kl1 = kl_div_loss(pred=output_dp1, target=output_dp2.detach(), t=self.T)
        loss_kl2 = kl_div_loss(pred=output_dp2, target=output_dp1.detach(), t=self.T)
        loss_kl = (self.T**2)*(loss_kl1 + loss_kl2)

        # total loss
        loss_total = loss_ce + loss_kl

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_total':loss_total}, None

class CS_KD(BaseMethod):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.T = args.t

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:
        batch_size = x.size(0)

        y_ = y[:batch_size//2]
        output, feats = self.backbone(x[:batch_size//2], return_feat=True)
        loss_ce = self.criterion_ce(output, y_)

        with torch.no_grad():
            outputs_cls, feats_cls = self.backbone(x[batch_size//2:], return_feat=True)
        loss_kl = (self.T**2)*kl_div_loss(output, outputs_cls.detach())

        loss_total = loss_ce + loss_kl

        return ({'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_total':loss_total},
                make_feature_vector(torch.cat([feats[-1], feats_cls[-1]], dim=0)) )

class CS_KD_Dropout(CS_KD):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.P = args.p

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], List[Tensor]]:
        losses, feats = super().calculate_loss(x, y)
        batch_size = x.shape[0] // 2

        # dropout only last feature
        feats_dp = F.dropout(feats, p=self.P)
        output_dp = self.backbone.fc(feats_dp)
        output_dp1, output_dp2 = output_dp[:batch_size], output_dp[batch_size:]

        # KL Divergence using dropout
        loss_kl_dp = (self.T**2)*kl_div_loss(output_dp1, output_dp2.detach(), t=self.T)

        losses['loss_kl_dropout'] = loss_kl_dp
        losses['loss_total'] += loss_kl_dp

        return losses, None

class DDGSD(BaseMethod):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.T = args.t

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:
        batch_size = x.shape[0] // 2
        assert (y[:batch_size] == y[batch_size:]).all(), "Wrong DataLoader"
        y = y[:batch_size]
        ## forward
        output, feats = self.backbone(x, return_feat=True)
        output_1, output_2 = output[:batch_size], output[batch_size:]
        
        # global feature
        feats = make_feature_vector(feats[-1])
        # mean vector of global feature
        feat_1, feat_2 = feats[:batch_size].mean(dim=0), feats[batch_size:].mean(dim=0)
        
        # cross-entropy loss
        loss_ce1 = self.criterion_ce(output_1, y)
        loss_ce2 = self.criterion_ce(output_2, y)
        loss_ce = loss_ce1 + loss_ce2
        
        # KL Divergence
        loss_kl1 = kl_div_loss(output_1, output_2.detach(), t=self.T)
        loss_kl2 = kl_div_loss(output_2, output_1.detach(), t=self.T)
        loss_kl = (loss_kl1 + loss_kl2)

        # MMD loss
        loss_mmd = F.mse_loss(feat_1, feat_2, reduction='sum')

        # total loss
        loss_total = loss_ce + loss_kl + 0.005*loss_mmd

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_mmd':loss_mmd, 'loss_total':loss_total}, feats

class DDGSD_Dropout(DDGSD):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.P = args.p

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Any]:
        losses, feats = super().calculate_loss(x, y)
        batch_size = x.shape[0] // 2

        # dropout only last feature
        feats_dp = F.dropout(feats, p=self.P)
        output_dp = self.backbone.fc(feats_dp)
        output_dp1, output_dp2 = output_dp[:batch_size], output_dp[batch_size:]

        # KL Divergence using dropout
        loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2.detach(), t=self.T)
        loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1.detach(), t=self.T)
        loss_kl_dp = (self.T**2)*(loss_kl_dp1 + loss_kl_dp2)

        losses['loss_kl_dropout'] = loss_kl_dp
        losses['loss_total'] += loss_kl_dp

        return losses, None

#############################################################

def kd_loss_function(output, target_output, args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / args.t
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

class BYOT(nn.Module):

    def __init__(self, args, backbone: Module) -> None:
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.T = args.t
        self.P = args.p
        self.dropout = nn.Dropout2d(p=self.P)
        self.set_optimizer()

    def set_log(self, epoch: int, num_batchs: int) -> Tuple[Dict[str, AverageMeter], ProgressMeter]:
        meters = {}
        meters['batch_time'] = AverageMeter('Time', ':.3f')
        meters['data_time'] = AverageMeter('Data', ':.3f')
        meters['losses'] = AverageMeter('Loss', ':.4f')

        progress = ProgressMeter(num_batchs,
                                 meters.values(),
                                 prefix=f'Epoch[{epoch}] Batch')

        return meters, progress

    def set_optimizer(self) -> None:
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')


        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        self.optimizer = MultipleOptimizer([optimizer])
        self.lr_scheduler = MultipleSchedulers([lr_scheduler])

    def compute_kl_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        x_log = F.log_softmax(pred/self.T, dim=1)
        y = F.softmax(target/self.T, dim=1)
        
        return self.criterion_kl(x_log, y)

    def kd_loss_function(output, target_output, args):
        """Compute kd loss"""
        """
        para: output: middle ouptput logits.
        para: target_output: final output has divided by temperature and softmax.
        """

        output = output / args.t
        output_log_softmax = torch.log_softmax(output, dim=1)
        loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
        return loss_kd

    def make_feats_dropout(self, feat: Tensor, num: int = 2) -> List[Tensor]:
        """
        make features with dropout
        feat: (B x C x H x W)
        num: # of features with dropout
        """
        return [self.dropout(feat) for _ in range(num)]

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
        
    def make_output(self, feat: Tensor) -> Tensor:
        """
        make output of feature
        feat: (B x C x H x W)
        """
        net = self.backbone
        out = net.avgpool(feat)
        out = torch.flatten(out, 1)
        out = net.fc(out)

        return out
    def calculate_loss(self, x: Tensor, target: Tensor) -> Dict[str, Tensor]:

        output, middle_output1, middle_output2, middle_output3, final_fea, middle1_fea, middle2_fea, middle3_fea = self.forward(x)
        loss_ce = self.criterion(output, target)

        # dropout only last feature
        feats_dp = self.make_feats_dropout(final_fea)
        output_dp1, output_dp2 = [self.make_output(feats_dp[j]) for j in range(2)]
        
        # KL Divergence
        loss_kl1 = self.compute_kl_loss(output_dp1, output_dp2.detach())
        loss_kl2 = self.compute_kl_loss(output_dp2, output_dp1.detach())
        loss_self_kl = (self.T**2)*(loss_kl1 + loss_kl2)



        middle1_loss = self.criterion(middle_output1, target)
        middle2_loss = self.criterion(middle_output2, target)
        middle3_loss = self.criterion(middle_output3, target)
        temp4 = output / self.args.t
        temp4 = torch.softmax(temp4, dim=1)

        loss1by4 = kd_loss_function(
            middle_output1, temp4.detach(), self.args) * (self.args.t**2)
        loss2by4 = kd_loss_function(
            middle_output2, temp4.detach(), self.args) * (self.args.t**2)
        loss3by4 = kd_loss_function(
            middle_output3, temp4.detach(), self.args) * (self.args.t**2)

        feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach())
        feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach())
        feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach())

        byot_loss =  (middle1_loss + middle2_loss + middle3_loss) \
            + 0.1 * (loss1by4 + loss2by4 + loss3by4) \
                + 0.000001 * (feature_loss_1 + feature_loss_2 + feature_loss_3)

        loss = loss_ce + self.args.alpha*loss_self_kl + self.args.lambda_byot*byot_loss
        return {'loss': loss}

    def update_optimizer(self, results: Dict[str, Tensor]) -> None:
        loss = results['loss']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_log(self,
                   results: Dict[str, Tensor],
                   meters: Dict[str, AverageMeter],
                   size: int, end) -> Dict[str, AverageMeter]:
        meters['losses'].update(results['loss'].item(), size)
        meters['batch_time'].update(time.time() - end)

        return meters

    def train_loop(self, dataloader, epoch: int, freq: int = 10) -> Dict[str, AverageMeter]:
        self.train()
        meters, progress = self.set_log(epoch, len(dataloader))
        end = time.time()
        for i, (x, y) in enumerate(dataloader):
            meters['data_time'].update(time.time() - end)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            # calculate loss
            results = self.calculate_loss(x, y)

            # back-propagation
            self.update_optimizer(results)

            # log
            meters = self.update_log(results, meters, x.size(0), end)
            end = time.time()

            if (i % freq) == 0:
                progress.display(i)

        # lr schedulder
        self.lr_scheduler.step()
        return [meter for meter in meters.values() if 'Loss' in meter.name]

    @torch.no_grad()
    def evaluation(self, dataloader) -> float:
        self.eval()
        correct = 0
        total = 0
        for x, y in dataloader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            outputs = self.forward(x)
            _, preds = torch.max(outputs[0], dim=1)

            # log
            total += y.size(0)
            correct += (preds == y).sum().item()
        # print(f'Acc {100*correct/total:.4f}')
        return 100*correct/total

################################
