"""
Loss functions for Pyt
"""

import torch
import torch.nn as nn
import numpy as np


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


def RSE(ypred, ytrue):
    rse = np.sqrt(np.square(ypred - ytrue).sum()) / \
          np.sqrt(np.square(ytrue - ytrue.mean()).sum())
    return rse


def quantile_loss(ytrue, ypred, qs):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    '''
    L = np.zeros_like(ytrue)
    for i, q in enumerate(qs):
        yq = ypred[:, :, i]
        diff = yq - ytrue
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()


def gaussian_likelihood_loss(z, mu, sigma):
    '''
    Gaussian Liklihood Loss
    Args:
    z (tensor): true observations, shape (num_ts, num_periods)
    mu (tensor): mean, shape (num_ts, num_periods)
    sigma (tensor): standard deviation, shape (num_ts, num_periods)
    likelihood:
    (2 pi sigma^2)^(-1/2) exp(-(z - mu)^2 / (2 sigma^2))
    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - (z - mu)^2 / (2 sigma^2)
    '''
    negative_likelihood = torch.log(sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + 6
    return negative_likelihood.mean()


def negative_binomial_loss(ytrue, mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)
    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))
    minimize loss = - log l_{nb}
    Note: torch.lgamma: log Gamma function
    '''
    batch_size, seq_len = ytrue.size()
    likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
                 - 1. / alpha * torch.log(1 + alpha * mu) \
                 + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    return - likelihood.mean()

def SMAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel() + 1e-4
    mean_y = (ytrue + ypred) / 2.
    return np.mean(np.abs((ytrue - ypred) \
                          / mean_y))


def MAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel() + 1e-4
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred) \
                          / ytrue))

class GuassianLikelihoodLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.float:
        '''
        Gaussian Liklihood Loss
        Args:
        z (tensor): true observations, shape (num_ts, num_periods)
        mu (tensor): mean, shape (num_ts, num_periods)
        sigma (tensor): standard deviation, shape (num_ts, num_periods)
        likelihood:
        (2 pi sigma^2)^(-1/2) exp(-(z - mu)^2 / (2 sigma^2))
        log likelihood:
        -1/2 * (log (2 pi) + 2 * log (sigma)) - (z - mu)^2 / (2 sigma^2)
        '''
        negative_likelihood = torch.log(sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + 6
        return negative_likelihood.mean()

class NegativeBinomialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ytrue: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.float:
        '''
        Negative Binomial Sample
        Args:
        ytrue (array like)
        mu (array like)
        alpha (array like)
        maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                    - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))
        minimize loss = - log l_{nb}
        Note: torch.lgamma: log Gamma function
        '''
        batch_size, seq_len = ytrue.size()
        likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
                    - 1. / alpha * torch.log(1 + alpha * mu) \
                    + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
        return - likelihood.mean()


class QuantileLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true: torch.Tensor, pred: torch.Tensor, qs: list[float]) -> torch.float:
        '''
        Quantile loss version 2
        Args:
        true (batch_size, output_horizon)
        pred (batch_size, output_horizon, num_quantiles)
        qs (list[float]): List of quantiles to use
        '''
        L = torch.zeros_like(true)
        for i, q in enumerate(qs):
            yq = pred[:, :, i]
            diff = yq - true
            L += torch.max(q * diff, (q - 1) * diff)
        return L.mean()

class WMAPELoss(nn.Module):
    def __init__(self):
        super(WMAPELoss, self).__init__()

    def forward(
        self, 
        insample: torch.Tensor,
        freq: int,
        forecast: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None
    ) -> torch.float:
        if weights is None:
            weights = torch.ones_like(forecast)

        if mask is None:
            mask = torch.ones_like(forecast)

        numerator = torch.sum(torch.abs(forecast - target) * weights * mask)
        denominator = torch.sum(torch.abs(target) * weights * mask)

        wmape = divide_no_nan(numerator, denominator)

        return torch.mean(wmape * mask)

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(
        self, 
        insample: torch.Tensor,
        freq: int,
        forecast: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return torch.mean(torch.abs((forecast - target) * weights))


class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(
        self, 
        insample: torch.Tensor,
        freq: int,
        forecast: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                          torch.abs(forecast.data) + torch.abs(target.data)) * mask)


class MASELoss(nn.Module):
    def __init__(self):
        super(MASELoss, self).__init__()

    def forward(
        self, 
        insample: torch.Tensor, 
        freq: int,
        forecast: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.float:
        """ 
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return torch.mean(torch.abs(target - forecast) * masked_masep_inv)



class PSLoss(nn.Module):
    def __init__(self, pred_layer, patch_len_threshold: int = 128):
        """
        Initialize the PSLoss class.

        Args:
            pred_layer (nn.Module): The model to be used for gradient-based dynamic weighting.
            patch_len_threshold (int, optional): The maximum patch length. Defaults to 128.
        """
        super(PSLoss, self).__init__()
        self.pred_layer = pred_layer
        self.patch_len_threshold = patch_len_threshold
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def create_patches(
        self, 
        x: torch.Tensor, 
        patch_len: int, 
        stride: int
    ) -> torch.Tensor:
        """
        Create patches from the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, L].
            patch_len (int): Length of each patch.
            stride (int): Stride for patching.

        Returns:
            torch.Tensor: Patches of shape [B, C, num_patches, patch_len].
        """
        x = x.permute(0, 2, 1) # [B, C, L] -> [B, L, C]
        B, C, L = x.shape
        
        num_patches = (L - patch_len) // stride + 1
        patches = x.unfold(2, patch_len, stride)
        patches = patches.reshape(B, C, num_patches, patch_len)
        
        return patches

    def fouriour_based_adaptive_patching(
        self, 
        true: torch.Tensor, 
        pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform fourier-based adaptive patching.

        Args:
            true (torch.Tensor): True values of shape [B, L].
            pred (torch.Tensor): Predicted values of shape [B, L].

        Returns:
            torch.Tensor: True patches of shape [B, C, num_patches, patch_len].
            torch.Tensor: Predicted patches of shape [B, C, num_patches, patch_len].
        """
        # Get patch length an stride
        true_fft = torch.fftorch.rfft(true, dim=1)
        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        frequency_list[:1] = 0.0
        top_index = torch.argmax(frequency_list)
        period = (true.shape[1] // top_index)
        patch_len = min(period // 2, self.patch_len_threshold)
        stride = patch_len // 2
        
        # Patching
        true_patch = self.create_patches(true, patch_len, stride=stride)
        pred_patch = self.create_patches(pred, patch_len, stride=stride)

        return true_patch, pred_patch
    
    def patch_wise_structural_loss(
        self, 
        true_patch: torch.Tensor, 
        pred_patch: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate patch-wise structural loss.

        Args:
            true_patch (torch.Tensor): True patches of shape [B, C, num_patches, patch_len].
            pred_patch (torch.Tensor): Predicted patches of shape [B, C, num_patches, patch_len].

        Returns:
            torch.Tensor: Patch-wise structural loss.
        """
        # Calculate mean
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)
        
        # Calculate variance and standard deviation
        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var)
        pred_patch_std = torch.sqrt(pred_patch_var)
        
        # Calculate Covariance
        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1, keepdim=True)
        
        # 1. Calculate linear correlation loss
        patch_linear_corr = (true_pred_patch_cov + 1e-5) / (true_patch_std * pred_patch_std + 1e-5)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        # 2. Calculate variance
        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()
        
        # 3. Mean loss
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()
        
        return linear_corr_loss, var_loss, mean_loss
    
    def gradient_based_dynamic_weighting(
        self, 
        true: torch.Tensor, 
        pred: torch.Tensor, 
        corr_loss: torch.Tensor, 
        var_loss: torch.Tensor, 
        mean_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform gradient-based dynamic weighting.

        Args:
            true (torch.Tensor): True values of shape [B, L].
            pred (torch.Tensor): Predicted values of shape [B, L].
            corr_loss (torch.Tensor): Correlation loss.
            var_loss (torch.Tensor): Variance loss.
            mean_loss (torch.Tensor): Mean loss.

        Returns:
            torch.Tensor: Gradient-based dynamic weighting.
        """
        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = torch.mean(true, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var)
        pred_std = torch.sqrt(pred_var)
        true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        linear_sim = (1.0 + linear_sim) * 0.5
        var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)
   
        # Gradiant based dynamic weighting
        corr_gradient = torch.autograd.grad(corr_loss, self.pred_layer.parameters(), create_graph=True)[0]
        var_gradient = torch.autograd.grad(var_loss, self.pred_layer.parameters(), create_graph=True)[0]
        mean_gradient = torch.autograd.grad(mean_loss, self.pred_layer.parameters(), create_graph=True)[0]
        gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0

        aplha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        beta =  gradiant_avg.norm().detach() /  var_gradient.norm().detach()
        gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()
        
        return aplha, beta, gamma

    def forward(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate the PS loss.

        Args:
            true (torch.Tensor): True values of shape [B, L].
            pred (torch.Tensor): Predicted values of shape [B, L].

        Returns:
            torch.Tensor: PS loss.
        """
        # Fourior based adaptive patching
        true_patch, pred_patch = self.fouriour_based_adaptive_patching(true, pred)
        
        # Pacth-wise structural loss
        corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)
        
        # Gradient based dynamic weighting
        alpha, beta, gamma = self.gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss)

        # Final PS loss
        loss = alpha * corr_loss + beta * var_loss + gamma * mean_loss
        
        return loss



