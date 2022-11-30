import numpy as np
import torch
from numba import jit
from torch.autograd import Function
import random

def traceback_soda_matrix(path, N, M, add_missing=True):
    """
    N = Number of target boxes
    M = Number of proposal boxes
    """
    
    def get_pairs(temp_path, i, j):
        p = np.where(temp_path[i][:j+1] == 2)[0]
        if i != 0 and len(p) == 0:
            return get_pairs(temp_path, i-1, j)
        elif i == 0 or p[-1] == 0:
            if len(p) == 0:
                return []
            return [(i, p[-1])]
        else:
            return get_pairs(temp_path, i-1, p[-1]-1) + [(i, p[-1])]
    
    pairs = get_pairs(path, N-1, M-1)
    if len(pairs) == 0:
        pairs = [(i,i) for i in range(N)]
        
    if len(pairs) != N and add_missing:
        # import pdb
        # pdb.set_trace()
        pairs = np.array(pairs)
        missing_targets = set(np.array(range(N))) - set(pairs[:, 0])
        missing_preds = set(np.array(range(M))) - set(pairs[:, 1])
        random_preds = random.sample(missing_preds, len(missing_targets))
        missing = np.array([[j,i] for i,j in zip(random_preds, missing_targets)])
        pairs = np.append(pairs, missing, 0)
        return pairs[:,1], pairs[:,0]
    return np.array(pairs)[:,1], np.array(pairs)[:,0]


@jit(nopython=True)
def compute_softsoda(D, gamma):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    # R = np.ones((B, N + 2, M + 2)) * np.inf
    R = np.zeros((B, N + 2, M + 2))
    path = - np.ones((B, N + 2, M + 2))
    R[:, 0, 0] = 0
    for k in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                r0 = -(R[k, i - 1, j - 1] + D[k, i - 1, j - 1]) / gamma
                r1 = -R[k, i - 1, j] / gamma
                r2 = -R[k, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + \
                    np.exp(r2 - rmax)
                
                R[k, i, j] = - gamma * (np.log(rsum) + rmax)
                max_index = -1
                if r2 > max(r0, r1):
                    max_index = 1
                elif r1 > max(r0, r2):
                    max_index = 0
                elif r0 > max(r1, r2):
                    max_index = 2
                path[k, i, j] = max_index
                # print(r0,r1,r2,rmax,rsum, softmin, D[k, i - 1, j - 1])
    
    return R, path


@jit(nopython=True)
def compute_softsoda_backward(D_, R, gamma):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                a0 = (R[k, i + 1, j] - R[k, i, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] -
                      D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + \
                    E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]


class _SoftSODA(Function):

    @staticmethod
    def forward(ctx, D, gamma):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        R, path = compute_softsoda(D_, g_)
        R = torch.Tensor(R).to(dev).type(dtype)
        path = torch.Tensor(path).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma)
        return R[:, -2, -2], path #, R

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        E = torch.Tensor(compute_softsoda_backward(
            D_, R_, g_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None


def box_iou(boxes1, boxes2):
    area1 = boxes1[:, 1] - boxes1[:, 0]
    area2 = boxes2[:, 1] - boxes2[:, 0]
    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M,2]
    inter = (rb - lt).clamp(min=0)  # [N,M,2]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-5)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 1:] >= boxes1[:, :1]).all()
    assert (boxes2[:, 1:] >= boxes2[:, :1]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    area = (rb - lt).clamp(min=0)  # [N,M,2]
    giou = iou - (area - union) / (area + 1e-5)
    return -giou
    #return -iou

class SoftSODA(torch.nn.Module):
    def __init__(self, gamma=1.0, normalize=False):
        super(SoftSODA, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.soda_matrix = None
        self.cost_matrix = None
        self.path = None
        self.func_soda = _SoftSODA.apply

    # def calc_distance_matrix(self, x, y):
    #     batch = x.size(0)
    #     out = []
    #     for b in range(batch):
    #       out.append(generalized_box_iou(x[b], y[b]))
    #     return torch.stack(out, dim=0)

    def forward(self, D_xy):

        out_xy, path = self.func_soda(D_xy, self.gamma)
        self.path = path
        # self.soda_matrix = dp
        result = out_xy
        return -result.squeeze(0)

    # def forward(self, x, y):
    #     assert len(x.shape) == len(y.shape)
    #     squeeze = False
    #     if len(x.shape) < 3:
    #         x = x.unsqueeze(0)
    #         y = y.unsqueeze(0)
    #         squeeze = True
    #     if self.normalize:
    #         D_xy = self.calc_distance_matrix(x, y)
    #         out_xy, path = self.func_soda(D_xy, self.gamma)
    #         self.path = path
    #         D_xx = self.calc_distance_matrix(x, x)
    #         out_xx, _ = self.func_soda(D_xx, self.gamma)
    #         D_yy = self.calc_distance_matrix(y, y)
    #         out_yy, _ = self.func_soda(D_yy, self.gamma)
    #         result = out_xy - 1/2 * (out_xx + out_yy)  # distance
    #     else:
    #         D_xy = self.calc_distance_matrix(x, y)
    #         out_xy, path = self.func_soda(D_xy, self.gamma)
    #         self.path = path
    #         result = out_xy  # discrepancy
    #     return result.squeeze(0) if squeeze else result