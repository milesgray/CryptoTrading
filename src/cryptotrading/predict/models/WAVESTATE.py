import torch
import torch.nn as nn

from ..layers.WaveletTransform import WaveletTransformLayer
from ..layers.AdaptiveKernel import AdaptiveKernelLayer
from ..layers.StateSpace import StateSpaceLayer
from ..layers.Diffusion import DiffusionBlock
from ..layers.SelfAttention import SelfAttention


class WAVESTATE(nn.Module):
    """WAVElet-enhanced State Space Transformer with Adaptive Temporal Encoding
    
    An advanced time series prediction model that combines wavelet transforms,
    state space models, adaptive kernels, and diffusion principles."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, seq_len=20, num_layers=3):
        super(WAVESTATE, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        # Feature extraction layers
        self.wavelet_layer = WaveletTransformLayer(hidden_dim, hidden_dim)
        
        # Main processing blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': SelfAttention(hidden_dim, num_heads=4),
                'adaptive_kernel': AdaptiveKernelLayer(hidden_dim, hidden_dim),
                'state_space': StateSpaceLayer(hidden_dim),
                'diffusion': DiffusionBlock(hidden_dim)
            })
            self.layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Confidence predictor (uncertainty quantification)
        self.confidence_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, noise_level=None):
        # Default noise level
        if noise_level is None:
            noise_level = torch.zeros(x.shape[0], device=x.device)
        
        # Initial projection and add positional encoding
        x = self.input_proj(x)
        seq_len = min(x.shape[1], self.positional_encoding.shape[1])
        x[:, :seq_len] = x[:, :seq_len] + self.positional_encoding[:, :seq_len]
        
        # Multi-scale feature extraction
        x = self.wavelet_layer(x)
        
        # Process through main layers
        for layer in self.layers:
            # Apply attention
            attn_out = layer['attention'](x)
            x = x + attn_out
            
            # Apply adaptive kernel
            kernel_out = layer['adaptive_kernel'](x)
            x = x + kernel_out
            
            # Apply state space transformation
            ss_out = layer['state_space'](x)
            x = x + ss_out
            
            # Apply diffusion block
            x = layer['diffusion'](x, noise_level)
        
        # Project to output dimension
        prediction = self.output_proj(x)
        confidence = torch.sigmoid(self.confidence_proj(x))
        
        return prediction, confidence
