import torch
import numpy as np

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def inverse_transform(self, data):
        return (data - self.mean) / self.std

    def transform(self, data):
        return (data * self.std) + self.mean


class StandardScaler:

    def fit_transform(self, y):
        self.mean = np.mean(y)
        self.std = np.std(y) + 1e-4
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        return y * self.std + self.mean

    def transform(self, y):
        return (y - self.mean) / self.std


class MaxScaler:

    def fit_transform(self, y):
        self.max = np.max(y)
        return y / self.max

    def inverse_transform(self, y):
        return y * self.max

    def transform(self, y):
        return y / self.max


class MeanScaler:

    def fit_transform(self, y):
        self.mean = np.mean(y)
        return y / self.mean

    def inverse_transform(self, y):
        return y * self.mean

    def transform(self, y):
        return y / self.mean


class LogScaler:

    def fit_transform(self, y):
        return np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y)

    def transform(self, y):
        return np.log1p(y)



class UnitGaussianScaler(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianScaler, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianScaler(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianScaler, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeScaler(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeScaler, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x
    