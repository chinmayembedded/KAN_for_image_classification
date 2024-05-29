# datasets will be put here
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
import numpy as np
from torch.utils.data import random_split

random.seed(42)
np.random.seed(42)

class DataPipeline:

    def __init__(self, **configs):

        self.configs = configs

    
    def get_dataloader(self):
        dataset_name = self.configs['dataset_name'].lower()

        if dataset_name == 'cifar10':
            return self._load_cifar10()
        elif dataset_name == 'tiny_imagenet':
            return self._load_tiny_imagenet()
        elif dataset_name == 'cifar100':
            return self._load_cifar100()
        
    
    def _load_cifar10(self):

        transforms_train = T.Compose([
            T.Resize((224,224)),
            T.ToTensor()]) # add any extra transforms for training
        transforms_test = T.Compose([
            T.Resize((224,224)),
            T.ToTensor()])
        full_trainset = datasets.CIFAR10(root=self.configs['datapath'], train=True, download=True, transform=transforms_train)
        testset = datasets.CIFAR10(root=self.configs['datapath'], train=False, download=True, transform=transforms_test)

        # calculating the splits
        train_size = int((1-val_split) * len(full_trainset))
        val_size = len(full_trainset) - train_size

        trainset, valset = random_split(full_trainset, [train_size, val_size])
        train_load = DataLoader(trainset, batch_size=self.configs['batch_size'], shuffle=True, num_workers=self.configs['num_workers'], pin_memory=self.configs['pin_memory'])
        val_load = DataLoader(valset, batch_size=self.configs['batch_size'], shuffle=False, num_workers=self.configs['num_workers'], pin_memory=self.configs['pin_memory'])
        testset_load = DataLoader(testset, batch_size=1, shuffle=False, num_workers=self.configs['num_workers'], pin_memory=self.configs['pin_memory'])

        return train_load, val_load, testset_load


    def _load_tiny_imagenet(self):
        # different loading process. Tiny imagenet is available from hugging face datasets.
        pass


    def _load_cifar100(self):

        transforms_train = T.Compose([T.ToTensor()]) # add any extra transforms for training
        transforms_test = T.Compose([T.ToTensor()])
        trainset = datasets.CIFAR100(root=self.configs['datapath'], train=True, download=True, transform=transforms_train)
        testset = datasets.CIFAR100(root=self.configs['datapath'], train=False, download=True, transform=transforms_test)
        train_load = DataLoader(trainset, batch_size=self.configs['batch_size'], shuffle=True, num_workers=self.configs['num_workers'], pin_memory=self.configs['pin_memory'])
        testset_load = DataLoader(testset, batch_size=1, shuffle=False, num_workers=self.configs['num_workers'], pin_memory=self.configs['pin_memory'])

        return train_load, testset_load
    
# testing pipeline
params = None
import yaml
with open('/app/kan_classifiers/kan_classifiers/configs/exp1.yaml', 'r') as config_file:
    params = yaml.load(config_file, Loader=yaml.FullLoader)

# EXPERIMENT
exp_name = params['experiment']['exp_name']

# MODEL
model_name = params['model']['model_name']

# OPTIMIZATION
epoch = params['optimization']['epoch']
lr = params['optimization']['lr']

# DATASET
dataset_name = params['dataset']['dataset_name']
datapath = params['dataset']['datapath']
batch_size = params['dataset']['batch_size']
val_split = params['dataset']['val_split']
num_workers = params['dataset']['num_workers']
pin_memory = params['dataset']['pin_memory']

# LOGGING
log_folder = params['logging']['folder']

data_pipeline = DataPipeline(
    dataset_name=dataset_name,
    datapath=datapath, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    pin_memory=pin_memory, 
    val_split=val_split
    )

cifar10_train, cifar10_val, cifar10_test = data_pipeline.get_dataloader()
from kan_classifiers.models.sota.vit import CustomViT

model = CustomViT(out_features=10, hidden_features=256).to('cuda')
for data, label in cifar10_train:
    data = data.to('cuda')
    label = label.to('cuda')
    pred = model(data)

