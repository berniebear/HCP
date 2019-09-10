import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, layers=2, dropout=0.5, leaky=True):
        super(Net, self).__init__()
        self.layers = layers
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(dropout)
        if leaky == True:
            self.act = torch.nn.ReLU()
        else:
            self.act = torch.nn.LeakyReLU()
            #self.act = torch.nn.Tanh()

    def forward(self, x):
        x = self.act(self.hidden1(x))  
        if self.layers == 3:
            x = self.act(self.hidden2(x))
            #x = self.act(self.hidden3(x))      
        x = self.predict(x)  
        return x
