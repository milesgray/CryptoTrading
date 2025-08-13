
class DiffusionBlock(nn.Module):
    """Diffusion-inspired block with noise level conditioning"""
    def __init__(self, hidden_dim):
        super(DiffusionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.noise_embed = nn.Linear(1, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x, noise_level):
        # Noise level conditioning
        noise_embed = self.noise_embed(noise_level.unsqueeze(-1))
        x = self.norm1(x + noise_embed)
        
        # Feed-forward with residual connection
        x = x + self.ff(self.norm2(x))
        return x
