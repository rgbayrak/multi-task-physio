import torch
from torch import nn


def pearsonr(X, Y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y1 : 1D torch.Tensor
    y2 : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        # >>> x = np.random.randn(100)
        # >>> y = np.random.randn(100)
        # >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        # >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        # >>> np.allclose(sp_corr, th_corr)
    """
    total = 0
    total_zeros = 0
    for batch_idx in range(X.shape[0]):
        x = X[batch_idx].squeeze()
        y = Y[batch_idx].squeeze()
        # print(x.shape)
        # print(y.shape)
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x.sub(mean_x)
        ym = y.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        r_val = r_num / r_den
        # if r_num == 0:
        #     r_val = torch.tensor(0.0).cuda()
        total += r_val

    return 1 - total/X.shape[0]

def corrcoef(x):
    """
    Mimics `np.corrcoef`

    Arguments
    ---------
    x : 2D torch.Tensor

    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)

    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013

    Example:
        # >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        # >>> np_corr = np.corrcoef(x)
        # >>> th_corr = corrcoef(torch.from_numpy(x))
        # >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
