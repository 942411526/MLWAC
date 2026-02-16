import os
import argparse
import time
import pdb
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from vint_train.models.vint.self_attention import MultiLayerDecoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Distance(nn.Module):

    def __init__(self, distance_encoder, 
                       dist_pred_net):
        super(Distance, self).__init__()


        self.distance_encoder = distance_encoder

        self.dist_pred_net = dist_pred_net


      

    def forward(self, **kwargs):
      

        return self.dist_pred_net(self.distance_encoder(obs_img=kwargs["obs_img"],goal_img=kwargs["goal_img"]))
        


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):

        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output



