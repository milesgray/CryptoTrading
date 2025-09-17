import torch
import torch.nn as nn

class AdaptiveKernelLayer(nn.Module):
    """KAN-inspired adaptive kernel layer"""
    def __init__(self, input_dim, output_dim, num_kernels=16):
        super(AdaptiveKernelLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_kernels = num_kernels
        
        # Learnable basis functions (kernels)
        self.kernels = nn.Parameter(torch.randn(num_kernels, input_dim))
        
        # Mixing weights
        self.mixer = nn.Linear(input_dim, num_kernels)
        
        # Output projection
        self.output_proj = nn.Linear(num_kernels, output_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Calculate kernel activations
        # Reshape x for broadcasting: (batch, seq, 1, input_dim)
        x_expanded = x.unsqueeze(2)
        
        # Reshape kernels for broadcasting: (1, 1, num_kernels, input_dim)
        kernels_expanded = self.kernels.unsqueeze(0).unsqueeze(0)
        
        # Compute distances: (batch, seq, num_kernels)
        kernel_distances = torch.sum((x_expanded - kernels_expanded) ** 2, dim=-1)
        kernel_activations = torch.exp(-kernel_distances)
        
        # Get adaptive mixing weights: (batch, seq, num_kernels)
        mixing_weights = torch.softmax(self.mixer(x), dim=-1)
        
        # Apply weighted kernel activations
        weighted_activations = kernel_activations * mixing_weights
        
        # Project to output dimension
        output = self.output_proj(weighted_activations)
        return output