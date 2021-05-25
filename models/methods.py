#-------------------------------------
# loss_total = ce_loss + beta*method_loss + alpha*dropout_kl_loss
# KL Divergence Loss는 kl_div_loss라는 funtion으로 구현.
# 
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
           'CS_KD', 'DDGSD',
           'CS_KD_Dropout', 'DDGSD_Dropout',
            'BYOT', 'BYOT_Dropout',
            'DML', 'DML_Dropout','DML_Dropout_V1',
            'Base_Dropout', 'Base_Dropout_v2',
            'BaseMethod_LS', 'SD_Dropout_LS']

def kl_div_loss(pred: Tensor, target: Tensor, t: float = 3.0) -> Tensor:
    """
    Calculate KL Divergence Loss.
    Args:
        pred: output logit before softmax
        target: output logit before softmax
        t: temperature
    Return:
        kl divergence loss
    """
    x_log = F.log_softmax(pred/t, dim=1)
    y = F.softmax(target/t, dim=1)
    
    return F.kl_div(x_log, y, reduction='batchmean')

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

    def forward(self, x: Tensor, return_feat=False) -> Tensor:
        return self.backbone(x, return_feat)
    
    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Args:
            x, y: Tensor.
        Return:
            losses: Dict. {'loss_ce': loss_ce, 'loss_kl': loss_kl, 'loss_total': loss_ce+loss_kl}. Need 'loss_total'.
            feature_vectors: (Tensor)
        """
        outputs, feats = self.backbone(x, return_feat=True)
        loss_ce = self.criterion_ce(outputs, y)
        loss_total = loss_ce.clone()

        return {'loss_ce':loss_ce, 'loss_total':loss_total}, make_feature_vector(feats[-1])

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
        self.alpha = args.alpha
        self.detach = args.detach

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Any]:
        losses, feature_vector = super().calculate_loss(x, y)
        
        # dropout only last feature
        feats_dp = [F.dropout(feature_vector, p=self.P) for _ in range(2)]
        output_dp1, output_dp2 = [self.backbone.fc(feats_dp[j]) for j in range(2)]
        
        # KL Divergence
        if not self.detach:     # no detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2, t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1, t=self.T)
        else:                   # detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2.detach(), t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1.detach(), t=self.T)
        loss_kl = (self.T**2)*(loss_kl_dp1 + loss_kl_dp2)

        # total loss
        losses['loss_kl'] = loss_kl
        losses['loss_total'] += self.alpha*loss_kl

        return losses, None

class CS_KD(BaseMethod):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.beta = args.beta

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:
        batch_size = x.size(0)

        y_ = y[:batch_size//2]
        output, feats = self.backbone(x[:batch_size//2], return_feat=True)
        loss_ce = self.criterion_ce(output, y_)

        with torch.no_grad():
            outputs_cls, _ = self.backbone(x[batch_size//2:], return_feat=True)
        loss_kl = (4.0**2)*kl_div_loss(output, outputs_cls.detach(), t=4.0)

        loss_total = loss_ce + self.beta*loss_kl

        return ({'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_total':loss_total},
                make_feature_vector(feats[-1]) )

class CS_KD_Dropout(CS_KD):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.T = args.t
        self.P = args.p
        self.alpha = args.alpha
        self.detach = args.detach

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], List[Tensor]]:
        losses, feature_vector = super().calculate_loss(x, y)

        # dropout only last feature
        feats_dp = [F.dropout(feature_vector, p=self.P) for _ in range(2)]
        output_dp1, output_dp2 = [self.backbone.fc(feats_dp[j]) for j in range(2)]

        # KL Divergence using dropout
        if not self.detach:     # no detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2, t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1, t=self.T)
        else:                   # detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2.detach(), t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1.detach(), t=self.T)

        loss_kl_dp = (self.T**2)*(loss_kl_dp1 + loss_kl_dp2)

        losses['loss_kl_dropout'] = loss_kl_dp
        losses['loss_total'] += self.alpha*loss_kl_dp

        return losses, None

class DDGSD(BaseMethod):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.beta = args.beta

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
        loss_kl1 = kl_div_loss(output_1, output_2.detach(), t=3.0)
        loss_kl2 = kl_div_loss(output_2, output_1.detach(), t=3.0)
        loss_kl = (3.0**2)*(loss_kl1 + loss_kl2)

        # MMD loss
        loss_mmd = F.mse_loss(feat_1, feat_2, reduction='sum')

        # total loss
        loss_total = loss_ce + self.beta*loss_kl + 0.005*loss_mmd

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_mmd':loss_mmd, 'loss_total':loss_total}, feats

class DDGSD_Dropout(DDGSD):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.T = args.t
        self.P = args.p
        self.alpha = args.alpha
        self.detach = args.detach

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Any]:
        losses, feature_vector = super().calculate_loss(x, y)
        batch_size = x.shape[0] // 2

        # dropout only last feature
        feats_dp = F.dropout(feature_vector, p=self.P)
        output_dp = self.backbone.fc(feats_dp)
        output_dp1, output_dp2 = output_dp[:batch_size], output_dp[batch_size:]

        # KL Divergence using dropout
        if not self.detach:     # no detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2, t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1, t=self.T)
        else:                   # detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2.detach(), t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1.detach(), t=self.T)
        loss_kl_dp = (self.T**2)*(loss_kl_dp1 + loss_kl_dp2)

        losses['loss_kl_dropout'] = loss_kl_dp
        losses['loss_total'] += self.alpha*loss_kl_dp

        return losses, None


class BYOT(BaseMethod):
    def __init__(self, args, backbone: nn.Module) -> None:
        super().__init__(args, backbone)
        self.beta = args.beta

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Any]:

        output, [middle_output1, middle_output2, middle_output3, final_fea,\
             middle1_fea, middle2_fea, middle3_fea] = self.backbone(x, return_feat=True)
        
        # final ce loss
        loss_ce = self.criterion_ce(output, y)
        
        # middle ce loss
        middle_loss_1 = self.criterion_ce(middle_output1, y)
        middle_loss_2 = self.criterion_ce(middle_output2, y)
        middle_loss_3 = self.criterion_ce(middle_output3, y)
        middle_loss = middle_loss_1 + middle_loss_2 + middle_loss_3

        # middle kl loss
        loss1by4 = kl_div_loss(middle_output1, output.detach(), t=3.0)
        loss2by4 = kl_div_loss(middle_output2, output.detach(), t=3.0)
        loss3by4 = kl_div_loss(middle_output3, output.detach(), t=3.0)

        loss_kl_btw_blk = (loss1by4+loss2by4+loss3by4) * 3.0**2  # (?)

        # middle feature loss
        feature_loss_1 = F.mse_loss(middle1_fea, final_fea.detach())
        feature_loss_2 = F.mse_loss(middle2_fea, final_fea.detach())
        feature_loss_3 = F.mse_loss(middle3_fea, final_fea.detach())
        feature_loss = (feature_loss_1 + feature_loss_2 + feature_loss_3)

        loss_total = loss_ce + self.beta*(0.9*middle_loss + 0.1*loss_kl_btw_blk + 0.000001*feature_loss) 

        return {'loss_ce':loss_ce, 'loss_kl':middle_loss, 'loss_mmd':loss_kl_btw_blk, 'loss_total':loss_total}, final_fea


class BYOT_Dropout(BYOT):
    def __init__(self, args, backbone: nn.Module) -> None:
        super().__init__(args, backbone)
        self.T = args.t
        self.P = args.p
        self.alpha = args.alpha
        self.detach = args.detach

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], List[Tensor]]:
        losses, feature_vector = super().calculate_loss(x, y)

        # dropout only last feature
        feats_dp = [F.dropout(feature_vector, p=self.P).squeeze() for _ in range(2)]
        output_dp1, output_dp2 = [self.backbone.fc(feats_dp[j]) for j in range(2)]

        # KL Divergence using dropout
        if not self.detach:     # no detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2, t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1, t=self.T)
        else:                   # detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2.detach(), t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1.detach(), t=self.T)

        loss_kl_dp = (self.T**2)*(loss_kl_dp1 + loss_kl_dp2)

        losses['loss_kl_dropout'] = loss_kl_dp
        losses['loss_total'] += self.alpha*loss_kl_dp

        return losses, None


class DML(BaseMethod):
    def __init__(self, args, backbone: Module, backbone2: Module =None) -> None:
        super().__init__(args, backbone)
        self.beta = args.beta
        self.backbone2 = backbone2
        self.set_optimizer()

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Any]:
        output_1, feats_1 = self.backbone(x, return_feat=True)
        output_2, feats_2 = self.backbone2(x, return_feat=True)

        # ce loss
        loss_ce = 0.5*(self.criterion_ce(output_1, y) + self.criterion_ce(output_2, y))

        # kl btw two model
        loss_kl_12 = kl_div_loss(output_1, output_2.detach(), t=3.0)
        loss_kl_21 = kl_div_loss(output_2, output_1.detach(), t=3.0)

        loss_kl = 3.0**2 * (loss_kl_12 + loss_kl_21)

        loss_total = loss_ce + self.beta*loss_kl

        return ({'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_total':loss_total},
                [make_feature_vector(feats_1[-1]),
                make_feature_vector(feats_2[-1]) ] )


class DML_Dropout(DML):
    def __init__(self, args, backbone: Module, backbone2: Module=None) -> None:
        super().__init__(args, backbone, backbone2)
        self.T = args.t
        self.P = args.p
        self.alpha = args.alpha
        self.detach = args.detach

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Any]:
        losses, feature_vectors = super().calculate_loss(x, y)

        # dropout only last feature
        feats_dp = [F.dropout(feature_vectors[i], p=self.P) for i in range(2)]
        output_dp1, output_dp2 = [self.backbone.fc(feats_dp[j]) for j in range(2)]

        # KL Divergence using dropout
        if not self.detach:     # no detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2, t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1, t=self.T)
        else:                   # detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2.detach(), t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1.detach(), t=self.T)

        loss_kl_dp = (self.T**2)*(loss_kl_dp1 + loss_kl_dp2)

        losses['loss_kl_dropout'] = loss_kl_dp
        losses['loss_total'] += self.alpha*loss_kl_dp

        return losses, None


class DML_Dropout_V1(DML):
    def __init__(self, args, backbone: Module, backbone2: Module) -> None:
        super().__init__(args, backbone, backbone2)
        self.T = args.t
        self.P = args.p
        self.alpha = args.alpha
        self.detach = args.detach

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Any]:
        losses, feature_vectors = super().calculate_loss(x, y)

        # dropout only last feature
        feats_dp = [F.dropout(feature_vectors[0], p=self.P) for _ in range(2)]
        output_dp1, output_dp2 = [self.backbone.fc(feats_dp[j]) for j in range(2)]

        # KL Divergence using dropout
        if not self.detach:     # no detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2, t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1, t=self.T)
        else:                   # detach
            loss_kl_dp1 = kl_div_loss(output_dp1, output_dp2.detach(), t=self.T)
            loss_kl_dp2 = kl_div_loss(output_dp2, output_dp1.detach(), t=self.T)

        loss_kl_dp = (self.T**2)*(loss_kl_dp1 + loss_kl_dp2)

        losses['loss_kl_dropout'] = loss_kl_dp
        losses['loss_total'] += self.alpha*loss_kl_dp

        return losses, None

class Base_Dropout(BaseMethod):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.T = args.t
        self.P = args.p
        self.alpha = args.alpha
        self.detach = args.detach

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Any]:
        _, feature_vector = super().calculate_loss(x, y)
        
        # dropout only last feature
        feats_dp = F.dropout(feature_vector, p=self.P)
        output_dp = self.backbone.fc(feats_dp)
        
        loss_dropout = self.criterion_ce(output_dp, y)

        # total loss
        losses = {}
        losses['loss_dropout'] = loss_dropout
        losses['loss_total'] = self.alpha*loss_dropout

        return losses, None

class Base_Dropout_v2(BaseMethod):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)
        self.T = args.t
        self.P = args.p
        self.alpha = args.alpha
        self.detach = args.detach

    def calculate_loss(self, x: Tensor, y: Tensor) -> Tuple[Dict[str, Tensor], Any]:
        losses, feature_vector = super().calculate_loss(x, y)
        
        # dropout only last feature
        feats_dp = F.dropout(feature_vector, p=self.P)
        output_dp = self.backbone.fc(feats_dp)
        
        loss_dropout = self.criterion_ce(output_dp, y)

        # total loss
        losses['loss_dropout'] = loss_dropout
        losses['loss_total'] += self.alpha*loss_dropout

        return losses, None

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

class BaseMethod_LS(BaseMethod):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)

    def set_optimizer(self) -> None:
        super().set_optimizer()
        self.criterion_ce = LabelSmoothingCrossEntropy()
        
class SD_Dropout_LS(SD_Dropout):
    def __init__(self, args, backbone: Module) -> None:
        super().__init__(args, backbone)

    def set_optimizer(self) -> None:
        super().set_optimizer()
        self.criterion_ce = LabelSmoothingCrossEntropy()