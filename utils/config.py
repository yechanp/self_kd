import os
import yaml
import shutil
from utils.utils import Logger

class Config(object):
    """ default param setting
    """

    def __init__(self, args) -> None:

        # training setting
        self.exp_name: str = args.exp_name
        self.gpu: str = args.gpu
        self.num_workers: int = args.num_workers
        self.seed: int = args.seed
        self.resume: str = args.resume

        self.dataset: str = args.dataset
        self.method: str = args.method
        self.backbone: str = args.backbone
        self.epochs: int = args.epochs
        self.t: float = args.t
        self.p: float = args.p
        self.lam_sdd: float = args.lam_sdd
        self.lam_kd: float = args.lam_kd
        self.detach: bool = args.detach
        self.aug: bool = args.aug
        self.force: bool = args.force

        assert len(self.__dict__) == len(args.__dict__), "Check Arguments"

        self.prefetch: int = 2

        # hyper-params
        self.hyper_param = {
            'dataset': '',
            'method': '',
            'batch_size': 'B',
        }
        
        if not 'Base' in self.method:
            self.hyper_param.update({
                't': 'T',
                'lam_kd': 'Lambda_SelfKD',
            })

        if 'Base_Dropout' in self.method:
            if 'v2' in self.method:
                self.hyper_param.update({
                    'lam_kd': 'Lambda_Dropout',
                })
        elif 'Dropout' in self.method:
            self.hyper_param.update({
                'p': 'P',
                'detach': 'Detach',
                'lam_sdd': 'Lambda_SDD',
            })

        if 'resnet18' not in self.backbone:
            self.hyper_param.update({
                'backbone': '',
            })

        self.hyper_param.update({
            'seed': 'SEED',
        })

        self.build()

    def __str__(self) -> str:
        _str = "==== params setting ====\n"
        for k, v in self.__dict__.items():
            if k == 'args': continue
            _str += f"{k} : {v}\n"
        return _str

    def set_params(self, params: dict) -> None:
        for k, v in params.items():
            self.__setattr__(k, v)

    def build(self):
        """ 
        expname 재정의 
        method, dataset, backbone 일치
        log/tb dir 만들기
        """
        # Backbone
        if 'BYOT' in self.method and 'byot' not in self.backbone:
            self.backbone = 'byot_' + self.backbone
        if 'CIFAR' in self.dataset and 'cifar' not in self.backbone:
            self.backbone = self.backbone + '_cifar'

        # Batch Size
        if 'CIFAR' in self.dataset:
            self.batch_size = 128
        else:
            self.batch_size = 32

        for k, v in self.hyper_param.items():
            self.exp_name += f"_{v}{self.__getattribute__(k)}"

        if self.exp_name[0] == '_': self.exp_name = self.exp_name[1:]

        self.save_folder = os.path.join('saved_models', self.exp_name)
        self.tb_folder = os.path.join('tb_results', self.exp_name)

        if os.path.exists(self.save_folder) or os.path.exists(self.tb_folder):
            print(f"Current Experiment is : {self.exp_name}")
            if not self.force:
                isdelete = input("delete exist exp dir (y/n): ")
                if isdelete == "y":
                    if os.path.exists(self.save_folder): shutil.rmtree(self.save_folder) 
                    if os.path.exists(self.tb_folder):   shutil.rmtree(self.tb_folder) 
                elif isdelete == "n":
                    raise FileExistsError
                else:
                    raise FileExistsError
            else:
                print("#"*50)
                print("Run by force")
                print("#"*50)

        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(self.tb_folder, exist_ok=True)
        ## logfile
        self.logfile = os.path.join(self.save_folder, 'log.txt')

        self.save()

    def save(self) -> None:
        """
        attribute들 
        실험 dir에 yaml 파일로 저장.
        """
        yaml_path = os.path.join(self.save_folder, "params.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f)
