import numpy as np
from numpy import linalg as LA
import torch
import scipy.sparse as sp
import argparse
import sys

def laplacian(W, n):
    adj = W.numpy()
    I = np.identity(n)
    In = torch.from_numpy(I)

    d = adj.sum(axis=0)
    d = 1 / np.sqrt(d)
    dim = adj.shape[0]
    D = sp.coo_matrix((dim, dim))
    D.setdiag(d)  # <scipy.sparse.coo.coo_matrix>
    D_t = torch.from_numpy(D.toarray())  # <tensor>

    # float
    w = W #.to(torch.double)

    # Laplacian matrix
    tmp = torch.mm(D_t, w)
    L = In - torch.mm(tmp, D_t)


    value, vector = LA.eig(L)
    lambda_max = max(value)
    """
    if normalized:
        assert lmax <= 2
        lmax = 2
    """
    L = 2 * L / lambda_max - I
    return L.float()

def cheby_basis_eval(L, x, K):
    """
    :param L: n*n, tensor
    :param x: n*m, feature matrix, tensor
    :param k: Cheb order
    :return: T_k*X where T_k are the Chebyshev polynomials of order up to K
    time: O(KMN)
    Xt ∈ R(K * n * m), return g(L)
    """
    n, m = x.shape
    x = x.float()
    # float
    Xt = torch.empty(K, n, m)
    #Xt = Xt.to(torch.double)
    #x = x.to(torch.double)

    # Xt(0) = x
    Xt[0, ...] = x
    # Xt_1 = Lx
    if K > 1:
        Xt[1, ...] = torch.mm(L, x)
    # Xt_k = 2 L Xt_k-1 - Xt_k-2
    for k in range(2, K):
        Xt[k, ...] = 2 * torch.mm(L, Xt[k - 1, ...].clone()) - Xt[k - 2, ...].clone()

    return Xt

def filter_basis(Xt, c):
    """
    :param Xt: (k, n, m)
    :param c: (k, m, out), params
    :return: (n, out), prediction
    """
    k, n, m = Xt.shape
    _, _, out = c.shape
    y = torch.zeros(n, out)
    for i in range(k):
        #            n * m   m * out
        mul = torch.mm(Xt[i], c[i])
        y = y + mul
    return y

class ArgParser(argparse.ArgumentParser):
    """ ArgumentParser with better error message"""
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint

def save_checkpoint(epoch, model, optimizer, psnr=0, opt=None, path=None, best=False):
    psnr1 = format(psnr, '.4f')
    if path is None:
        path = "checkpoints/checkpoint.best" if best else "checkpoints/cp_p={psnr}".format(psnr=psnr1)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model if best else model.state_dict(),
            "optimizer_state_dict": optimizer if best else optimizer.state_dict(),
            "opt": opt,
            "psnr": psnr
        },
        path,
    )

def save_checkpoint_f(epoch, model, optimizer, psnr=0, opt=None, path=None, best=False):
    psnr1 = format(psnr, '.4f')
    if path is None:
        path = "checkpoints/f+checkpoint.best" if best else "checkpoints/f+cp_p={psnr}".format(psnr=psnr1)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model if best else model.state_dict(),
            "optimizer_state_dict": optimizer if best else optimizer.state_dict(),
            "opt": opt,
            "psnr": psnr
        },
        path,
    )

class AvgMetric(object):
    def __init__(self, total):
        self.acc = 0.0
        self.total = total

    def add(self, value):
        self.acc += value

    def clear(self):
        self.acc = 0

    def average(self, count=None):
        if count:
            return self.acc / count
        else:
            return self.acc / self.total
