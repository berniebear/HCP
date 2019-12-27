import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable as V

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, layers=2, act='leaky_relu'):
        super(Net, self).__init__()
        self.layers = layers
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        self.drop =torch.nn.Dropout(p=0.3)
        if act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU()
        else:
            assert(0)

    def forward(self, x):
        x = self.drop(x)
        if self.layers == 1:
            x = self.predict0(x)  
        elif self.layers == 2:
            x = self.act(self.hidden1(x)) 
            x = self.predict(x)  
        elif self.layers == 3:
            x = self.act(self.hidden1(x)) 
            o = x
            x = self.act(self.hidden2(x)) + o # with residual
            x = self.predict(x)  
        return x
