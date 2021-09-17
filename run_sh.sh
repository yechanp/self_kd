#                      exp_name  seed    dataset     backbone    optim   lr        wd
sh run_multiple_exp.sh Exp       41      CIFAR100    resnet18    adam    0.001     5e-4 &&
sh run_multiple_exp.sh Exp       41      CIFAR100    resnet18    adam    0.0001    1e-4 &&
sh run_multiple_exp.sh Exp       41      CIFAR100    resnet18    adam    0.01      1e-4