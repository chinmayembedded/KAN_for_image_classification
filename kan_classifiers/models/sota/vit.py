import torch.nn as nn
from transformers import AutoModel
from ..classifiers.mlp import MLP

class CustomViT(nn.Module):
    def __init__(self, out_features:int, hidden_features:int, model_name:str='facebook/dinov2-base'):
        super(CustomViT, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.head = MLP(in_features=self.backbone.config.hidden_size, hidden_features=hidden_features, out_features=out_features)
    
    def forward(self, x):
        output = self.backbone(x)
        last_hidden_states = output.last_hidden_state[:, 0, :]
        output = self.head(last_hidden_states)
        return output



        
