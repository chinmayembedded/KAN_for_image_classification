import numpy as np 
import argparse
from config import Params 
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from src.model.models import FNet, convNet

def weight_init_normal(m):
    classname=m.__class__.__name__
    if classname.find('Linear')!=-1:
        n = m.in_features
        y = (1.0/np.sqrt(n))
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    if Params.DATASET_NAME == "CIFAR":
        train_data = datasets.CIFAR10('data',train=True,download=True,transform=transform_train)
        test_data = datasets.CIFAR10('data',train=False,download=True,transform=transform_test)
        train_length = len(train_data)
        indices = list(range(len(train_data)))
        split = int(np.floor(Params.valid_size * train_length))

        np.random.shuffle(indices)

        train_idx = indices[split:]
        valid_idx = indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, 
            num_workers=Params.num_workers, 
            batch_size=Params.batch_size, 
            sampler=train_sampler
        )
    valid_loader = DataLoader(train_data,
            num_workers=Params.num_workers, 
            batch_size=Params.batch_size, 
            sampler=validation_sampler
        )
    test_loader = DataLoader(test_data,
        shuffle=True,
        num_workers=Params.num_workers, 
        batch_size=Params.batch_size
        )

    criterion = nn.CrossEntropyLoss()
    
    model_1 = FNet()
    model_2 = convNet()
 
    model_1.apply(weight_init_normal),model_2.apply(weight_init_normal)

    if torch.cuda.is_available():
        model_1.cuda()
        model_2.cuda()

    m1_loss, m1_acc = trainNet(model_1, 0.01, train_loader, valid_loader)
    m2_loss, m2_acc = trainNet(model_2, 0.01, train_loader, valid_loader)


    # Loading the model from the lowest validation loss 
    model_1.load_state_dict(torch.load('FNet_model.pth'))
    model_2.load_state_dict(torch.load('convNet_model.pth'))

