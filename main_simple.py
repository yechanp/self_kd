from models.models import *
from models.resnet import resnet18
from dataset import dataset_cifar


train_dataset, test_dataset = dataset_cifar('cifar10')
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                            shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                            shuffle=False, num_workers=1)                                            
backbone = resnet18(num_classes=10)
model = BaseTempelet(backbone)
model.cuda()

model.train_loop(trainloader, 1)
model.evaluation(testloader)
