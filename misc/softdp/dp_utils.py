import numpy as np
import torch

from itertools import product
from torch import log, exp
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


class VarTable():
    def __init__(self, dims, dtype=torch.float, device=device):
        self.dims = dims
        d1, d2, d_rest = dims[0], dims[1], dims[2:]

        self.vars = []
        for i in range(d1):
            self.vars.append([])
            for j in range(d2):
                var = torch.zeros(d_rest).to(dtype).to(device)
                self.vars[i].append(var)

    def __getitem__(self, pos):
        i, j = pos
        return self.vars[i][j]

    def __setitem__(self, pos, new_val):
        i, j = pos
        if self.vars[i][j].sum() != 0:
            assert False, "This cell has already been assigned. There must be a bug somwhere."
        else:
            self.vars[i][j] = self.vars[i][j] + new_val

    def show(self):
        device, dtype = self[0, 0].device, self[0, 0].dtype
        mat = torch.zeros((self.d1, self.d2, self.d3)).to().to(dtype).to(device)
        for dims in product([range(d) for d in self.dims]):
            i, j, rest = dims[0], dims[1], dims[2:]
            mat[dims] = self[i, j][rest]
        return mat


def minGamma(inputs, gamma=1, keepdim=True):
    """ continuous relaxation of min defined in the D3TW paper"""
    if type(inputs) == list:
        if inputs[0].shape[0] == 1:
            inputs = torch.cat(inputs)
        else:
            inputs = torch.stack(inputs, dim=0)

    if gamma == 0:
        minG = inputs.min(dim=0, keepdim=keepdim)
    else:
        # log-sum-exp stabilization trick
        zi = (-inputs / gamma)
        max_zi = zi.max()
        log_sum_G = max_zi + log(exp(zi - max_zi).sum(dim=0, keepdim=keepdim) + 1e-5)
        minG = -gamma * log_sum_G
    return minG


def minProb(inputs, gamma=1, keepdim=True):
    if type(inputs) == list:
        if inputs[0].shape[0] == 1:
            inputs = torch.cat(inputs)
        else:
            inputs = torch.stack(inputs, dim=0)

    if gamma == 0:
        minP = inputs.min(dim=0, keepdim=keepdim)
    else:
        probs = F.softmax(-inputs / gamma, dim=0)
        minP = (probs * inputs).sum(dim=0, keepdim=keepdim) 
    return minP


def traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)