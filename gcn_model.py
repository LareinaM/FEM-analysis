import torch.nn as nn
import torch.nn.functional as F
import gcn_layer as conv
from gcn_utils import *
import numpy as np
import torch


class Net(nn.Module):
    def __init__(self):
        """
        :param out_feature: 1/3, depending on whether target is stress/displacement
        :param n: #vertices
        """
        super(Net, self).__init__()
        # (n, 7)
        self.conv1 = conv.GraphConvolution(7, 1120, k=6)  # -> (n, 1120)
        self.conv2 = conv.GraphConvolution(1120, 560, k=5)  # -> (n, 560)
        self.conv3 = conv.GraphConvolution(560, 280, k=4)  # -> (n, 280)
        self.conv4 = conv.GraphConvolution(280, 140, k=3)  # -> (n, 140)

        self.fc1 = nn.Linear(in_features=140, out_features=140)  # -> (n, 140)
        self.fc2 = nn.Linear(in_features=140, out_features=4)  # -> (n, 4)

    def forward(self, x, adj):
        """
        :param adj: n*n, adjacency matrix
        :param x: n*7, feature matrix
        """
        n = len(x)
        L = laplacian(adj, n)

        # output of conv1
        x = self.conv1(x, 6, L)
        x = F.relu(x, inplace=False)  # -> (n, 1120)

        # output of conv2
        x = self.conv2(x, 5, L)
        x = F.relu(x)  # -> (n, 560)

        # output of conv3
        x = self.conv3(x, 4, L)
        x = F.relu(x)  # -> (n, 280)

        # output of conv4
        x = self.conv4(x, 3, L)
        x = F.relu(x)  # -> (n, 140)

        x = self.fc1(x)
        x = self.fc2(x)
        return x.float()


"""
models = Net()
params = list(models.parameters())
print(len(params))
for item in params:
    print(item.shape)
"""