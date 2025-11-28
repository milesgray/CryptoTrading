# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for Pyt.
"""

import torch as t
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


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)



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
        x: t.Tensor, 
        patch_len: int, 
        stride: int
    ) -> t.Tensor:
        """
        Create patches from the input tensor.

        Args:
            x (t.Tensor): Input tensor of shape [B, C, L].
            patch_len (int): Length of each patch.
            stride (int): Stride for patching.

        Returns:
            t.Tensor: Patches of shape [B, C, num_patches, patch_len].
        """
        x = x.permute(0, 2, 1) # [B, C, L] -> [B, L, C]
        B, C, L = x.shape
        
        num_patches = (L - patch_len) // stride + 1
        patches = x.unfold(2, patch_len, stride)
        patches = patches.reshape(B, C, num_patches, patch_len)
        
        return patches

    def fouriour_based_adaptive_patching(
        self, 
        true: t.Tensor, 
        pred: t.Tensor
    ) -> t.Tensor:
        """
        Perform fourier-based adaptive patching.

        Args:
            true (t.Tensor): True values of shape [B, L].
            pred (t.Tensor): Predicted values of shape [B, L].

        Returns:
            t.Tensor: True patches of shape [B, C, num_patches, patch_len].
            t.Tensor: Predicted patches of shape [B, C, num_patches, patch_len].
        """
        # Get patch length an stride
        true_fft = t.fft.rfft(true, dim=1)
        frequency_list = t.abs(true_fft).mean(0).mean(-1)
        frequency_list[:1] = 0.0
        top_index = t.argmax(frequency_list)
        period = (true.shape[1] // top_index)
        patch_len = min(period // 2, self.patch_len_threshold)
        stride = patch_len // 2
        
        # Patching
        true_patch = self.create_patches(true, patch_len, stride=stride)
        pred_patch = self.create_patches(pred, patch_len, stride=stride)

        return true_patch, pred_patch
    
    def patch_wise_structural_loss(
        self, 
        true_patch: t.Tensor, 
        pred_patch: t.Tensor
    ) -> t.Tensor:
        """
        Calculate patch-wise structural loss.

        Args:
            true_patch (t.Tensor): True patches of shape [B, C, num_patches, patch_len].
            pred_patch (t.Tensor): Predicted patches of shape [B, C, num_patches, patch_len].

        Returns:
            t.Tensor: Patch-wise structural loss.
        """
        # Calculate mean
        true_patch_mean = t.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = t.mean(pred_patch, dim=-1, keepdim=True)
        
        # Calculate variance and standard deviation
        true_patch_var = t.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = t.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = t.sqrt(true_patch_var)
        pred_patch_std = t.sqrt(pred_patch_var)
        
        # Calculate Covariance
        true_pred_patch_cov = t.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1, keepdim=True)
        
        # 1. Calculate linear correlation loss
        patch_linear_corr = (true_pred_patch_cov + 1e-5) / (true_patch_std * pred_patch_std + 1e-5)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        # 2. Calculate variance
        true_patch_softmax = t.softmax(true_patch, dim=-1)
        pred_patch_softmax = t.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()
        
        # 3. Mean loss
        mean_loss = t.abs(true_patch_mean - pred_patch_mean).mean()
        
        return linear_corr_loss, var_loss, mean_loss
    
    def gradient_based_dynamic_weighting(
        self, 
        true: t.Tensor, 
        pred: t.Tensor, 
        corr_loss: t.Tensor, 
        var_loss: t.Tensor, 
        mean_loss: t.Tensor
    ) -> t.Tensor:
        """
        Perform gradient-based dynamic weighting.

        Args:
            true (t.Tensor): True values of shape [B, L].
            pred (t.Tensor): Predicted values of shape [B, L].
            corr_loss (t.Tensor): Correlation loss.
            var_loss (t.Tensor): Variance loss.
            mean_loss (t.Tensor): Mean loss.

        Returns:
            t.Tensor: Gradient-based dynamic weighting.
        """
        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = t.mean(true, dim=-1, keepdim=True)
        pred_mean = t.mean(pred, dim=-1, keepdim=True)
        true_var = t.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = t.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = t.sqrt(true_var)
        pred_std = t.sqrt(pred_var)
        true_pred_cov = t.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        linear_sim = (1.0 + linear_sim) * 0.5
        var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)
   
        # Gradiant based dynamic weighting
        corr_gradient = t.autograd.grad(corr_loss, self.pred_layer.parameters(), create_graph=True)[0]
        var_gradient = t.autograd.grad(var_loss, self.pred_layer.parameters(), create_graph=True)[0]
        mean_gradient = t.autograd.grad(mean_loss, self.pred_layer.parameters(), create_graph=True)[0]
        gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0

        aplha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        beta =  gradiant_avg.norm().detach() /  var_gradient.norm().detach()
        gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        gamma = gamma * t.mean(linear_sim * var_sim).detach()
        
        return aplha, beta, gamma

    def forward(self, true: t.Tensor, pred: t.Tensor) -> t.Tensor:
        """
        Calculate the PS loss.

        Args:
            true (t.Tensor): True values of shape [B, L].
            pred (t.Tensor): Predicted values of shape [B, L].

        Returns:
            t.Tensor: PS loss.
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