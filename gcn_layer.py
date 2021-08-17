import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from gcn_utils import *
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(k, in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, k, L):
        """
        :param adj: n*n, adjacency matrix
        :param x: n*in, feature matrix
        :param k: Cheb order
        """
        n = len(x)

        #L = laplacian(adj, n) # -> (n, n)
        Xt = cheby_basis_eval(L, x, k)  # -> k * n * in
        # -> n * out
        return (filter_basis(Xt, self.weight)).float()