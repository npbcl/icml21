import copy as cpy
import gzip
import math
import os
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as tod
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utils import *

class DynamicDNN(nn.Module):

    def __init__(self, in_dim, hidden_size, out_dim):
        super(DynamicDNN, self).__init__()
        self.size = [in_dim] + hidden_size + [out_dim]
        self.module_list = nn.ModuleList([])
        self.module_last = nn.ModuleList([])
        for i in range(len(self.size)-2):
            self.module_list.append(nn.Linear(self.size[i],self.size[i+1]))
        self.add_last()
    def forward(self, input):
        x = input.view(-1, self.size[0])
        N, D = x.shape
        for module in self.module_list:
            x = module(x)
            x = F.leaky_relu(x)
        
        outs = []
        for module in self.module_last:
            outs.append(module(x).view(N,1))
        
        x = torch.cat(outs, dim = 1)
        return F.softmax(x, dim = -1)

    def add_last(self):
        self.module_last.append(nn.Linear(self.size[-2],self.size[-1]))
