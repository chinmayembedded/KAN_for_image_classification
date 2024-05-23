# datasets will be put here
import numpy as np
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader

class DataPipeline:

    def __init__(self, configs):

        self.configs = configs

    
    def get_dataloader(self, dataset_name:str='cifar', batch_size:int=64, val_size:float=0.2, shuffle:bool=True, num_workers:int=1):
        dataset_name = dataset_name.lower()

        if dataset_name == 'cifar10':
            return self._load_cifar10(batch_size=batch_size, shuffle=shuffle, val_size=val_size, num_workers=num_workers)
        elif dataset_name == 'tiny_imagenet':
            return self._load_tiny_imagenet(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        elif dataset_name == 'cifar100':
            return self._load_cifar100(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def _get_train_val_loader(train_data, val_size):
        train_length = len(train_data)
        indices = list(range(len(train_data)))
        split = int(np.floor(val_size * train_length))

        np.random.shuffle(indices)

        train_idx = indices[split:]
        valid_idx = indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(valid_idx)
        return train_sampler, validation_sampler


    
    def _load_cifar10(self, batch_size:int, shuffle:bool, val_size:float, num_workers:int):

        transforms_train = T.Compose([T.ToTensor()]) # add any extra transforms for training
        transforms_test = T.Compose([T.ToTensor()])
        trainset = datasets.CIFAR10(root=self.configs['cifar10']['data_path'], train=True, download=True, transform=transforms_train)
        testset = datasets.CIFAR10(root=self.configs['cifar10']['data_path'], train=False, download=True, transform=transforms_test)
        train_sampler, val_sampler = self._get_train_val_loader(trainset, val_size)
        
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=train_sampler)
        val_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=val_sampler)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader


    def _load_tiny_imagenet(self):
        # different loading process. Tiny imagenet is available from hugging face datasets.
        pass


    def _load_cifar100(self, batch_size:int, shuffle:bool, num_workers:int):

        transforms_train = T.Compose([T.ToTensor()]) # add any extra transforms for training
        transforms_test = T.Compose([T.ToTensor()])
        trainset = datasets.CIFAR10(root=self.configs['cifar100']['data_path'], train=True, download=True, transform=transforms_train)
        testset = datasets.CIFAR10(root=self.configs['cifar100']['data_path'], train=False, download=True, transform=transforms_test)
        train_load = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        testset_load = DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

        return train_load, testset_load
    

configs = {
    'cifar10': {'data_path':'/app/kan_classifiers/kan_classifiers/data/cifar10'},
    'cifar100': {'data_path':'/app/kan_classifiers/kan_classifiers/data/cifar100'},
    'tiny_imagenet': {'data_path':'/app/kan_classifiers/kan_classifiers/data/tiny_imagenet'}
}

# testing pipeline
data_pipeline = DataPipeline(configs=configs)

cifar10_train, cifar10_test = data_pipeline.get_dataloader('CIFAR10')
