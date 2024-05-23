import numpy as np 
import argparse
from config import Params 
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
# from kan_classifiers.models.sota import FNet, convNet
from kan_classifiers.data.data import data_pipeline


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
        train_loader, val_loader, test_loader = data_pipeline.get_dataloader('cifar10')
    print(train_loader)

    criterion = nn.CrossEntropyLoss()
    
    if Params.MODEL_NAME == "ConvNet":
        from kan_classifiers.models.sota.base_cnn import ConvNet
        model = ConvNet()
        model.apply(weight_init_normal)

    
    # model_1 = FNet()
    # model_2 = convNet()
 
    # model_1.apply(weight_init_normal),model_2.apply(weight_init_normal)

    if torch.cuda.is_available():
        model_1.cuda()
        model_2.cuda()

    # m1_loss, m1_acc = trainNet(model_1, 0.01, train_loader, valid_loader)
    # m2_loss, m2_acc = trainNet(model_2, 0.01, train_loader, valid_loader)


    # # Loading the model from the lowest validation loss 
    # model_1.load_state_dict(torch.load('FNet_model.pth'))
    # model_2.load_state_dict(torch.load('convNet_model.pth'))

