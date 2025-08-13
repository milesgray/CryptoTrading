
class StateSpaceLayer(nn.Module):
    """Mamba-inspired selective state space layer"""
    def __init__(self, hidden_dim):
        super(StateSpaceLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.B = nn.Linear(hidden_dim, hidden_dim)
        self.C = nn.Linear(hidden_dim, hidden_dim)
        self.D = nn.Parameter(torch.zeros(hidden_dim))
        
        # Input-dependent parameters
        self.A_proj = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs = []
        
        # Input-dependent A parameter (selective state space)
        A_input = torch.sigmoid(self.A_proj(x))
        A = torch.exp(self.A.unsqueeze(0).unsqueeze(0) * A_input)
        
        # Sequential processing (can be parallelized with custom CUDA kernels)
        for t in range(seq_len):
            # Update state with input-dependent dynamics
            h = A[:, t] * h + self.B(x[:, t])
            
            # Generate output
            y = self.C(h) + self.D + x[:, t]  # Skip connection
            outputs.append(y.unsqueeze(1))
        
        # Combine outputs
        output = torch.cat(outputs, dim=1)
        return self.projection(output)