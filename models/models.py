#-------------------------------------
# 
#-------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import AverageMeter, ProgressMeter, MultipleOptimizer, MultipleSchedulers

__all__ = ['BaseMethod', 'AFD', 'DML', 
           'SelfKD_KL', 'SelfKD_KL_Delay', 'SelfKD_AFD',
           'SelfKD_KL_ExclusiveDropout', 'SelfKD_KL_Multi',
           'SelfKD_KL_once', 'CS_KD']
class BaseMethod(nn.Module):

    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.set_optimizer()
        
    def set_log(self, epoch, num_batchs):
        meters = {}
        meters['batch_time'] = AverageMeter('Time', ':.3f')
        meters['data_time'] = AverageMeter('Data', ':.3f')
        meters['losses'] = AverageMeter('Loss', ':.4f')

        progress = ProgressMeter(num_batchs, 
                                meters.values(),
                                prefix=f'Epoch[{epoch}] Batch')
        
        return meters, progress

    def set_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        self.optimizer = MultipleOptimizer([optimizer])
        self.lr_scheduler = MultipleSchedulers([lr_scheduler])

    def forward(self, x):
        return self.backbone(x)
    
    def calculate_loss(self, x, y):
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)

        return {'loss':loss}

    def update_optimizer(self, results):
        loss = results['loss']
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_log(self, results, meters, size, end):
        meters['losses'].update(results['loss'].item(), size)
        meters['batch_time'].update(time.time() - end)

        return meters

    def train_loop(self, dataloader, epoch, freq=10):
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

            if (i%freq) == 0:
                progress.display(i)

        ## lr schedulder
        self.lr_scheduler.step()
        return [meter for meter in meters.values() if 'Loss' in meter.name]

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

class DML(BaseMethod):
    def __init__(self, args, backbone, backbone2):
        super(BaseMethod, self).__init__()
        self.args = args
        self.T = args.t
        self.backbone = backbone
        self.backbone2 = backbone2
        ## parameters
        self.set_optimizer()
    
    def set_optimizer(self):
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        self.optimizer = MultipleOptimizer([optimizer])
        self.lr_scheduler = MultipleSchedulers([lr_scheduler])
    
    def set_log(self, epoch, num_batchs):
        meters, _ = super().set_log(epoch, num_batchs)
        meters['kl_losses'] = AverageMeter('KL_Loss', ':.4f')
        
        progress = ProgressMeter(num_batchs, meters=meters.values(),
                                prefix=f'Epoch[{epoch}] Batch')
        return meters, progress

    def compute_kl_loss(self, out1, out2):
        pred1_log = F.log_softmax(out1/self.T, dim=1)
        pred2 = F.softmax(out2/self.T, dim=1)
        return self.criterion_kl(pred1_log, pred2)

    def calculate_loss(self, x, y):
        outputs_1, outputs_2 = self.backbone(x), self.backbone2(x)
        loss_ce1 = self.criterion_ce(outputs_1, y)
        loss_ce2 = self.criterion_ce(outputs_2, y)

        loss_kl1 = self.compute_kl_loss(outputs_2, outputs_1)
        loss_kl2 = self.compute_kl_loss(outputs_1, outputs_2)

        loss_ce = loss_ce1 + loss_ce2
        loss_kl = (self.T**2)*loss_kl1 + (self.T**2)*loss_kl2
        loss_logit = loss_ce + loss_kl

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_logit':loss_logit}

    def update_optimizer(self, results):
        loss_logit = results['loss_logit']

        self.optimizer.zero_grad()
        loss_logit.backward()
        self.optimizer.step()

    def update_log(self, results, meters, size, end):
        meters['losses'].update(results['loss_ce'].item(), size)
        meters['kl_losses'].update(results['loss_kl'].item(), size)
        meters['batch_time'].update(time.time() - end)

        return meters

class AFD(BaseMethod):
    def __init__(self, args, backbone, backbone2):
        super(BaseMethod, self).__init__()
        self.args = args
        self.T = args.t
        self.backbone = backbone
        self.backbone2 = backbone2
        self.D_1 = self._make_discriminator()
        self.D_2 = self._make_discriminator()
        ## parameters
        self.backbones_parameters = list(backbone.parameters()) + list(backbone2.parameters())
        self.D_parameters = list(self.D_1.parameters()) + list(self.D_2.parameters())
        self.set_optimizer()

    def set_log(self, epoch, num_batchs):
        meters, _ = super().set_log(epoch, num_batchs)
        meters['kl_losses'] = AverageMeter('KL_Loss', ':.4f')
        meters['G_losses'] = AverageMeter('G_Loss', ':.4f')
        meters['D_losses'] = AverageMeter('D_Loss', ':.4f')
        
        progress = ProgressMeter(num_batchs, meters=meters.values(),
                                prefix=f'Epoch[{epoch}] Batch')
        return meters, progress

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
        optimizer_logit = torch.optim.SGD(self.backbones_parameters,   lr=0.1, momentum=0.9, weight_decay=1e-4)
        optimizer_feat_G = torch.optim.Adam(self.backbones_parameters, lr=2e-5,              weight_decay=0.1)
        optimizer_feat_D = torch.optim.Adam(self.D_parameters,         lr=2e-5,              weight_decay=0.1)
        lr_scheduler_logit = torch.optim.lr_scheduler.MultiStepLR(optimizer_logit,   milestones=[100, 150], gamma=0.1)
        lr_scheduler_feat_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_feat_G, milestones=[50, 100],  gamma=0.1)
        lr_scheduler_feat_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_feat_D, milestones=[50, 100],  gamma=0.1)
        self.optimizer = MultipleOptimizer([optimizer_logit, optimizer_feat_G, optimizer_feat_D])
        self.lr_scheduler = MultipleSchedulers([lr_scheduler_logit, lr_scheduler_feat_G, lr_scheduler_feat_D])

    def forward(self, x):
        return self.backbone(x)
    
    def compute_kl_loss(self, out1, out2):
        pred1_log = F.log_softmax(out1/self.T, dim=1)
        pred2 = F.softmax(out2/self.T, dim=1)
        return self.criterion_kl(pred1_log, pred2)

    def compute_D_loss(self, D, feat1, feat2):
        d_out1 = D(feat1.detach()).view(-1, 1)
        d_out2 = D(feat2.detach()).view(-1, 1)
        return torch.mean((1 - d_out1)**2 + (d_out2)**2, dim=0)

    def compute_G_loss(self, D, feat):
        return torch.mean((1 - D(feat))**2, dim=0)
    
    def calculate_loss(self, x, y):
        outputs_1, feat_1 = self.backbone(x, return_feat=True)
        outputs_2, feat_2 = self.backbone2(x, return_feat=True)
        loss_ce1 = self.criterion_ce(outputs_1, y)
        loss_ce2 = self.criterion_ce(outputs_2, y)

        loss_kl1 = self.compute_kl_loss(outputs_2, outputs_1)
        loss_kl2 = self.compute_kl_loss(outputs_1, outputs_2)

        loss_ce = loss_ce1 + loss_ce2
        loss_kl = (self.T**2)*loss_kl1 + (self.T**2)*loss_kl2
        loss_logit = loss_ce + loss_kl
        loss_feat_D = 0.5*(self.compute_D_loss(self.D_1, feat_1, feat_2) + self.compute_D_loss(self.D_2, feat_2, feat_1))
        loss_feat_G = 0.5*(self.compute_G_loss(self.D_1, feat_1) + self.compute_G_loss(self.D_2, feat_2))

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_logit':loss_logit, 
                'loss_feat_G':loss_feat_G , 'loss_feat_D':loss_feat_D}

    def update_optimizer(self, results):
        loss_logit = results['loss_logit']
        loss_feat_G = results['loss_feat_G']
        loss_feat_D = results['loss_feat_D']

        ## optimizers 0: logit, 1: G, 2: D
        # logit + feat_G
        self.optimizer.optimizers[0].zero_grad()
        self.optimizer.optimizers[1].zero_grad()
        (loss_logit + loss_feat_G).backward()
        self.optimizer.optimizers[0].step()
        self.optimizer.optimizers[1].step()
        # feat_D
        self.optimizer.optimizers[2].zero_grad()
        loss_feat_D.backward()
        self.optimizer.optimizers[2].step()

    def update_log(self, results, meters, size, end):
        meters['losses'].update(results['loss_ce'].item(), size)
        meters['kl_losses'].update(results['loss_kl'].item(), size)
        meters['G_losses'].update(results['loss_feat_G'].item(), size)
        meters['D_losses'].update(results['loss_feat_D'].item(), size)
        meters['batch_time'].update(time.time() - end)

        return meters

class SelfKD_AFD(AFD):
    def __init__(self, args, backbone):
        super(BaseMethod, self).__init__()
        self.args = args
        self.T = args.t
        self.P = args.p
        self.backbone = backbone
        self.D_1 = self._make_discriminator()
        self.D_2 = self._make_discriminator()
        ## parameters
        self.backbones_parameters = list(backbone.parameters())
        self.D_parameters = list(self.D_1.parameters()) + list(self.D_2.parameters())
        self.set_optimizer()

    def make_output(self, x):
        net = self.backbone
        x = net.avgpool(x)
        x = torch.flatten(x, 1)
        out = net.fc(x)

        return out

    def make_feats(self, x):
        net = self.backbone
        output, feat = net(x, return_feat=True)
        feats_dropout = [F.dropout2d(feat, p=self.P) for _ in range(2)]

        return output, feats_dropout

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

    def calculate_loss(self, x, y):
        output_wo_dropout, feats = self.make_feats(x)
        # feats = self.make_feats_with_multiple_dropout(x)
        outputs_1, outputs_2 = [self.make_output(feats[j]) for j in range(2)]
        # loss_ce = self.criterion_ce(output_wo_dropout, y)
        loss_ce = 0.5*(self.criterion_ce(outputs_1, y) + self.criterion_ce(outputs_2, y))

        loss_kl1 = self.compute_kl_loss(outputs_2, outputs_1)
        loss_kl2 = self.compute_kl_loss(outputs_1, outputs_2)

        loss_kl = (self.T**2)*loss_kl1 + (self.T**2)*loss_kl2
        loss_logit = loss_ce + loss_kl
        loss_feat_D = 0.5*(self.compute_D_loss(self.D_1, feats[0], feats[1]) + self.compute_D_loss(self.D_2, feats[1], feats[0]))
        loss_feat_G = 0.5*(self.compute_G_loss(self.D_1, feats[0]) + self.compute_G_loss(self.D_2, feats[1]))

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_logit':loss_logit, 
                'loss_feat_G':loss_feat_G , 'loss_feat_D':loss_feat_D}

    def update_optimizer(self, results):
        loss_logit = results['loss_logit']
        loss_feat_G = results['loss_feat_G']
        loss_feat_D = results['loss_feat_D']

        ## optimizers 0: logit, 1: G, 2: D
        # logit + feat_G
        self.optimizer.optimizers[0].zero_grad()
        self.optimizer.optimizers[1].zero_grad()
        (loss_logit + loss_feat_G).backward()
        self.optimizer.optimizers[0].step()
        self.optimizer.optimizers[1].step()
        # feat_D
        self.optimizer.optimizers[2].zero_grad()
        loss_feat_D.backward()
        self.optimizer.optimizers[2].step()

    def update_log(self, results, meters, size, end):
        meters['losses'].update(results['loss_ce'].item(), size)
        meters['kl_losses'].update(results['loss_kl'].item(), size)
        meters['G_losses'].update(results['loss_feat_G'].item(), size)
        meters['D_losses'].update(results['loss_feat_D'].item(), size)
        meters['batch_time'].update(time.time() - end)

        return meters

class SelfKD_KL(DML):
    def __init__(self, args, backbone):
        super(BaseMethod, self).__init__()
        self.T = args.t
        self.P = args.p
        self.backbone = backbone
        ## parameters
        self.set_optimizer()

    def make_output(self, x):
        net = self.backbone
        x = net.avgpool(x)
        x = torch.flatten(x, 1)
        out = net.fc(x)

        return out

    def make_feats(self, x):
        net = self.backbone
        output, feat = net(x, return_feat=True)
        feats_dropout = [F.dropout2d(feat, p=self.P) for _ in range(2)]

        return output, feats_dropout

    def calculate_loss(self, x, y):
        output_wo_dropout, feats = self.make_feats(x)
        # feats = self.make_feats_with_multiple_dropout(x)
        outputs_1, outputs_2 = [self.make_output(feats[j]) for j in range(2)]
        loss_ce = self.criterion_ce(output_wo_dropout, y)
        # loss_ce = 0.5*(self.criterion_ce(outputs_1, y) + self.criterion_ce(outputs_2, y))

        loss_kl1 = self.compute_kl_loss(outputs_2, outputs_1)
        loss_kl2 = self.compute_kl_loss(outputs_1, outputs_2)

        loss_kl = (self.T**2)*loss_kl1 + (self.T**2)*loss_kl2
        loss_logit = loss_ce + loss_kl

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_logit':loss_logit}

    def update_optimizer(self, results):
        loss_logit = results['loss_logit']

        self.optimizer.zero_grad()
        loss_logit.backward()
        self.optimizer.step()

    def update_log(self, results, meters, size, end):
        meters['losses'].update(results['loss_ce'].item(), size)
        meters['kl_losses'].update(results['loss_kl'].item(), size)
        meters['batch_time'].update(time.time() - end)

        return meters

class SelfKD_KL_once(SelfKD_KL):
    def __init__(self, args, backbone):
        super().__init__(args, backbone)

    def calculate_loss(self, x, y):
        output_wo_dropout, feats = self.make_feats(x)
        # feats = self.make_feats_with_multiple_dropout(x)
        outputs_1, outputs_2 = [self.make_output(feats[j]) for j in range(2)]
        loss_ce = self.criterion_ce(output_wo_dropout, y)
        # loss_ce = 0.5*(self.criterion_ce(outputs_1, y) + self.criterion_ce(outputs_2, y))

        loss_kl1 = self.compute_kl_loss(outputs_1, outputs_2)
        loss_kl = (self.T**2)*loss_kl1

        loss_logit = loss_ce + loss_kl

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_logit':loss_logit}

class SelfKD_KL_Delay(SelfKD_KL):
    def __init__(self, args, backbone):
        super().__init__(args, backbone)
    
    def calculate_loss_wo_dropout(self, x, y):
        outputs = self.forward(x)
        loss = self.criterion_ce(outputs, y)
        loss_kl = torch.tensor(0, dtype=torch.float32)
        if torch.cuda.is_available(): loss_kl = loss_kl.cuda()

        return {'loss_ce':loss, 'loss_kl':loss_kl, 'loss_logit':loss}

    def train_loop(self, dataloader, epoch, freq=10):
        self.train()
        meters, progress = self.set_log(epoch, len(dataloader))
        end = time.time()
        for i, (x, y) in enumerate(dataloader):
            meters['data_time'].update(time.time() - end)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            
            # calculate loss
            if epoch > 20:
                results = self.calculate_loss(x, y)
            else:
                results = self.calculate_loss_wo_dropout(x, y)

            # back-propagation
            self.update_optimizer(results)

            # log
            meters = self.update_log(results, meters, x.size(0), end)
            end = time.time()

            if (i%freq) == 0:
                progress.display(i)

        ## lr schedulder
        self.lr_scheduler.step()
        return [meter for meter in meters.values() if 'Loss' in meter.name]

class SelfKD_KL_Multi(SelfKD_KL):
    def __init__(self, args, backbone):
        super().__init__(args, backbone)
        self.num_multi = 2

    def make_feats(self, x):
        net = self.backbone
        output, feat = net(x, return_feat=True)
        feats_dropout = [F.dropout2d(feat, p=self.P) for _ in range(self.num_multi)]

        return output, feats_dropout

    def calculate_loss(self, x, y):
        output_wo_dropout, feats = self.make_feats(x)
        # feats = self.make_feats_with_multiple_dropout(x)
        outputs = [self.make_output(feats[j]) for j in range(self.num_multi)]
        loss_ce = self.criterion_ce(output_wo_dropout, y)
        # loss_ce = 0.5*(self.criterion_ce(outputs_1, y) + self.criterion_ce(outputs_2, y))

        loss_kl = torch.tensor(0, dtype=torch.float32)
        if torch.cuda.is_available(): loss_kl = loss_kl.cuda()
        for j in range(self.num_multi):
            for k in range(self.num_multi):
                loss_kl += self.compute_kl_loss(outputs[j], outputs[k])
        loss_kl *= (self.T**2)
        loss_logit = loss_ce + loss_kl/self.num_multi

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_logit':loss_logit}

class SelfKD_KL_ExclusiveDropout(SelfKD_KL):
    def __init__(self, args, backbone):
        super().__init__(args, backbone)

    def make_output(self, x):
        net = self.backbone
        x = net.avgpool(x)
        x = torch.flatten(x, 1)
        out = net.fc(x)

        return out

    def make_feats(self, x):
        net = self.backbone
        output, feat = net(x, return_feat=True)
        dropout_idxs = torch.randint(low=1, high=int(1/self.P), size=[feat.size(0), feat.size(1)])
        dp_factors = [(dropout_idxs!=(j+1)).to(torch.float32).reshape(-1, feat.size(1), 1, 1) for j in range(2)]
        if torch.cuda.is_available(): dp_factors = [dp_factors[j].cuda() for j in range(2)]
        dropout_scale = 1/(1-self.P)
        feats_dropout = [dropout_scale * feat * dp_factors[j] for j in range(2)]

        return output, feats_dropout

    def calculate_loss(self, x, y):
        output_wo_dropout, feats = self.make_feats(x)
        # feats = self.make_feats_with_multiple_dropout(x)
        outputs_1, outputs_2 = [self.make_output(feats[j]) for j in range(2)]
        loss_ce = self.criterion_ce(output_wo_dropout, y)
        # loss_ce = 0.5*(self.criterion_ce(outputs_1, y) + self.criterion_ce(outputs_2, y))

        loss_kl1 = self.compute_kl_loss(outputs_2, outputs_1)
        loss_kl2 = self.compute_kl_loss(outputs_1, outputs_2)

        loss_kl = (self.T**2)*loss_kl1 + (self.T**2)*loss_kl2
        loss_logit = loss_ce + loss_kl

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_logit':loss_logit}


class CS_KD(SelfKD_KL):
    def __init__(self, args, backbone):
        super().__init__(args, backbone)

    def calculate_loss(self, x, y):
        batch_size = x.size(0)

        y_ = y[:batch_size//2]
        outputs = self.backbone(x[:batch_size//2])
        loss_ce = self.criterion_ce(outputs, y_)

        with torch.no_grad():
            outputs_cls = self.backbone(x[batch_size//2:])
        loss_kl = self.compute_kl_loss(outputs, outputs_cls.detach())
        loss_kl *= self.T**2

        loss_logit = loss_ce + loss_kl

        return {'loss_ce':loss_ce, 'loss_kl':loss_kl, 'loss_logit':loss_logit}

"""
def kl_loss_compute(logits1, logits2):
    
    pred1 = tf.nn.softmax(logits1)
    pred2 = tf.nn.softmax(logits2)
    loss = tf.reduce_mean(tf.reduce_sum(pred2 * tf.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

    return loss
"""