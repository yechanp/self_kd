#-------------------------------------
# 
#-------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import AverageMeter, ProgressMeter, MultipleOptimizer, MultipleSchedulers

__all__ = ['BaseMethod', 'AFD', 'SelfKD']
class BaseMethod(nn.Module):

    def __init__(self, backbone):
        super(BaseMethod, self).__init__()
        self.backbone = backbone
        self.optimizer = MultipleOptimizer(self.set_optimizer())
        
    def set_log(self, epoch, num_batchs):
        batch_time = AverageMeter('Time', ':.3f')
        data_time = AverageMeter('Data', ':.3f')
        losses = AverageMeter('Loss', ':.4f')

        progress = ProgressMeter(num_batchs, 
                                [batch_time, data_time, losses],
                                prefix=f'Epoch[{epoch}] Batch')
        
        return [batch_time, data_time, losses], progress

    def set_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer]

    def forward(self, x):
        return self.backbone(x)
    
    def train_loop(self, dataloader, epoch, freq=10):
        self.train()
        [batch_time, data_time, losses], progress = self.set_log(epoch, len(dataloader))
        end = time.time()
        for i, (x, y) in enumerate(dataloader):
            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            outputs = self.forward(x)
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

        ## lr schedulder
        self.lr_scheduler.step()
        return [losses]

    @torch.no_grad()
    def evaluation(self, dataloader):
        self.eval()
        correct = 0
        total = 0
        for i, (x, y) in enumerate(dataloader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            outputs = self.forward(x)
            _, preds = torch.max(outputs, dim=1)
            
            # log
            total += y.size(0)
            correct += (preds == y).sum().item()
        # print(f'Acc {100*correct/total:.4f}')
        return 100*correct/total
        
class AFD(BaseMethod):
    def __init__(self, backbone, backbone2):
        super(BaseMethod, self).__init__()
        self.T = 3
        self.backbone = backbone
        self.backbone2 = backbone2
        self.D_1 = self._make_discriminator()
        self.D_2 = self._make_discriminator()
        ## parameters
        self.backbones_parameters = list(backbone.parameters()) + list(backbone2.parameters())
        self.set_optimizer()

    def set_log(self, epoch, num_batchs):
        log_list, _ = super().set_log(epoch, num_batchs)
        log_list.append(AverageMeter('KL_Loss', ':.4f'))
        log_list.append(AverageMeter('Feat_Loss', ':.4f'))
        progress = ProgressMeter(num_batchs, meters=log_list,
                                prefix=f'Epoch[{epoch}] Batch')
        return log_list, progress

    def _make_discriminator(self):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=2),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        return layer

    def set_optimizer(self):
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        optimizer_logit = torch.optim.SGD(self.backbones_parameters, lr=0.1, momentum=0.9, weight_decay=1e-4)
        optimizer_feat = torch.optim.Adam(self.parameters(), lr=2e-5, weight_decay=0.1)
        lr_scheduler_logit = torch.optim.lr_scheduler.MultiStepLR(optimizer_logit, milestones=[100, 150], gamma=0.1)
        lr_scheduler_feat = torch.optim.lr_scheduler.MultiStepLR(optimizer_feat, milestones=[50, 100], gamma=0.1)
        self.optimizer = MultipleOptimizer([optimizer_logit, optimizer_feat])
        self.lr_schedulers = MultipleSchedulers([lr_scheduler_logit, lr_scheduler_feat])

    def forward(self, x):
        return self.backbone(x)
    
    def compute_kl_loss(self, out1, out2):
        pred1_log = F.log_softmax(out1/self.T, dim=1)
        pred2 = F.softmax(out2/self.T, dim=1)
        return self.criterion_kl(pred1_log, pred2)

    def compute_D_loss(self, D, feat1, feat2):
        d_out1 = D(feat1).view(-1, 1)
        d_out2 = D(feat2).view(-1, 1)
        return torch.mean((1 - d_out1)**2 + (d_out2)**2, dim=0)
    
    def train_loop(self, dataloader, epoch, freq=10):
        self.train()
        [batch_time, data_time, losses, kl_losses, feat_losses], progress = self.set_log(epoch, len(dataloader))
        end = time.time()
        for i, (x, y) in enumerate(dataloader):
            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            outputs_1, feat_1 = self.backbone(x, return_feat=True)
            outputs_2, feat_2 = self.backbone2(x, return_feat=True)
            loss_ce1 = self.criterion_ce(outputs_1, y)
            loss_ce2 = self.criterion_ce(outputs_2, y)

            loss_kl1 = self.compute_kl_loss(outputs_2, outputs_1)
            loss_kl2 = self.compute_kl_loss(outputs_1, outputs_2)

            loss_ce = loss_ce1 + loss_ce2
            loss_kl = (self.T**2)*loss_kl1 + (self.T**2)*loss_kl2
            loss_logit = loss_ce + loss_kl
            loss_feat = self.compute_D_loss(self.D_1, feat_1, feat_2) + self.compute_D_loss(self.D_2, feat_2, feat_1)
            
            ############### Backpropagation ###############
            self.optimizer.zero_grad()
            # TODO: The results are different
            (loss_logit + loss_feat).backward()
            # loss_logit.backward(retain_graph=True)
            # loss_feat.backward()
            self.optimizer.step()
            ###############################################

            # log
            losses.update(loss_ce.item(), x.size(0))
            kl_losses.update(loss_kl.item(), x.size(0))
            feat_losses.update(loss_feat.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i%freq) == 0:
                progress.display(i)

        ## lr schedulder
        self.lr_schedulers.step()
        return [losses, kl_losses, feat_losses]

class SelfKD(AFD):
    def __init__(self, backbone):
        super(BaseMethod, self).__init__()
        self.T = 3
        self.P = 0.2
        self.backbone = backbone
        self.backbones_parameters = backbone.parameters()
        self.D_1 = self._make_discriminator()
        self.D_2 = self._make_discriminator()
        self.set_optimizer()

    def make_output(self, x):
        net = self.backbone
        x = net.avgpool(x)
        x = torch.flatten(x, 1)
        out = net.fc(x)

        return out

    def make_feats(self, x):
        net = self.backbone
        _, feat = net(x, return_feat=True)
        feats = [F.dropout2d(feat, p=self.P) for _ in range(2)]

        return feats

    def make_feats_with_multiple_dropout(self, x):
        net = self.backbone
        x = net.relu(net.bn1(net.conv1(x)))
        x = net.layer1(x)
        feats = [F.dropout2d(x, p=self.P) for _ in range(2)]
        feats = [net.layer2(feats[i]) for i in range(2)]
        feats = [F.dropout2d(feats[i], p=self.P) for i in range(2)]
        feats = [net.layer3(feats[i]) for i in range(2)]
        feats = [F.dropout2d(feats[i], p=self.P) for i in range(2)]
        feats = [net.layer4(feats[i]) for i in range(2)]

        return feats

    def train_loop(self, dataloader, epoch, freq=10):
        self.train()
        [batch_time, data_time, losses, kl_losses, feat_losses], progress = self.set_log(epoch, len(dataloader))
        end = time.time()
        for i, (x, y) in enumerate(dataloader):
            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            # feats = self.make_feats(x)
            feats = self.make_feats_with_multiple_dropout(x)
            outputs_1, outputs_2 = [self.make_output(feats[j]) for j in range(2)]
            loss_ce1 = self.criterion_ce(outputs_1, y)
            loss_ce2 = self.criterion_ce(outputs_2, y)

            loss_kl1 = self.compute_kl_loss(outputs_2, outputs_1)
            loss_kl2 = self.compute_kl_loss(outputs_1, outputs_2)

            loss_ce = loss_ce1 + loss_ce2
            loss_kl = (self.T**2)*loss_kl1 + (self.T**2)*loss_kl2
            loss_logit = loss_ce + loss_kl
            loss_feat = self.compute_D_loss(self.D_1, feats[0], feats[1]) + self.compute_D_loss(self.D_2, feats[1], feats[0])
            
            ############### Backpropagation ###############
            self.optimizer.zero_grad()
            # TODO: The results are different
            # loss_logit.backward(retain_graph=True)
            # loss_feat.backward()
            (loss_logit + loss_feat).backward()
            self.optimizer.step()
            ###############################################

            # log
            losses.update(loss_ce.item(), x.size(0))
            kl_losses.update(loss_kl.item(), x.size(0))
            feat_losses.update(loss_feat.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i%freq) == 0:
                progress.display(i)
            break

        ## lr schedulder
        self.lr_schedulers.step()
        return [losses, kl_losses, feat_losses]

"""
def kl_loss_compute(logits1, logits2):
    
    pred1 = tf.nn.softmax(logits1)
    pred2 = tf.nn.softmax(logits2)
    loss = tf.reduce_mean(tf.reduce_sum(pred2 * tf.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

    return loss
"""