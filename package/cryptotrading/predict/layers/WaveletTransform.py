
class WaveletTransformLayer(nn.Module):
    """Wavelet transform layer for multi-scale feature extraction"""
    def __init__(self, input_dim, output_dim):
        super(WaveletTransformLayer, self).__init__()
        self.high_pass = nn.Conv1d(input_dim, output_dim // 2, kernel_size=2, stride=2)
        self.low_pass = nn.Conv1d(input_dim, output_dim // 2, kernel_size=2, stride=2)
        self.reconstruct = nn.Linear(output_dim, output_dim)
        
    def forward(self, x):
        # Reshape for 1D convolution
        x_reshaped = x.transpose(1, 2)  # (batch_size, channels, seq_len)
        
        # Multi-scale decomposition
        high_freq = self.high_pass(x_reshaped)
        low_freq = self.low_pass(x_reshaped)
        
        # Concatenate frequency components
        multi_scale = torch.cat([high_freq, low_freq], dim=1)
        
        # Reshape back
        multi_scale = multi_scale.transpose(1, 2)  # (batch_size, seq_len/2, output_dim)
        
        # Reconstruction
        output = self.reconstruct(multi_scale)
        return output
