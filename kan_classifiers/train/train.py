import torch
import numpy as np 
import argparse
from config import Params 
import torch.nn as nn
import torch.nn.functional as F
from kan_classifiers.trainer.trainer import CifarTrainer
from kan_classifiers.data.data import DataPipeline
import torch.optim as optim



def weight_init_normal(m):
    classname=m.__class__.__name__
    if classname.find('Linear')!=-1:
        n = m.in_features
        y = (1.0/np.sqrt(n))
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    if Params.DATASET_NAME == "cifar10":
        data_pipeline = DataPipeline(configs=Params.configs)
        train_loader, val_loader, test_loader = data_pipeline.get_dataloader(
            dataset_name='cifar10', batch_size=Params.batch_size, num_workers=Params.num_workers)

    
    # Model architecture
    if Params.MODEL_NAME == "ConvNet":
        from kan_classifiers.models.sota.base_cnn import ConvNet
        model = ConvNet()
        model.apply(weight_init_normal)

    # Loss and optmizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if torch.cuda.is_available():
        model.cuda()

    if Params.DATASET_NAME == "cifar10":
        trainer = CifarTrainer()

    trainer.train_loop(train_loader, val_loader, model, optimizer, criterion, Params.model_path, Params.n_epochs)

    model.load_state_dict(torch.load(Params.model_path))

    trainer.evaluate_model(test_loader, model, criterion, Params.batch_size)

