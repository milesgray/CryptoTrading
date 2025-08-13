import torch
from torch import nn

class PSLoss(nn.Module):
    def __init__(self, model, patch_len_threshold: int = 128):
        """
        Initialize the PSLoss class.

        Args:
            model (nn.Module): The model to be used for gradient-based dynamic weighting.
            patch_len_threshold (int, optional): The maximum patch length. Defaults to 128.
        """
        super(PSLoss, self).__init__()
        self.model = model
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
        true_fft = torch.fft.rfft(true, dim=1)
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
        corr_gradient = torch.autograd.grad(corr_loss, self.model.predict_layers[-1].parameters(), create_graph=True)[0]
        var_gradient = torch.autograd.grad(var_loss, self.model.predict_layers[-1].parameters(), create_graph=True)[0]
        mean_gradient = torch.autograd.grad(mean_loss, self.model.predict_layers[-1].parameters(), create_graph=True)[0]
        gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0

        aplha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        beta =  gradiant_avg.norm().detach() /  var_gradient.norm().detach()
        gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()
        
        return aplha, beta, gamma

    def __call__(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
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