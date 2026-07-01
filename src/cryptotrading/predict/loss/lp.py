import torch

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def norm(self, x):
        return torch.norm(x, self.p, 1)

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*self.norm(x.view(num_examples,-1) - y.view(num_examples,-1))

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = self.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1))
        y_norms = self.norm(y.reshape(num_examples,-1))
        ratio_norms = diff_norms/y_norms

        if self.reduction:
            if self.size_average:
                return torch.mean(ratio_norms)
            else:
                return torch.sum(ratio_norms)

        return ratio_norms

    def __call__(self, x, y):
        return self.rel(x, y)