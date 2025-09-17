import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TemporalFlowNetwork(nn.Module):
    """
    Temporal Flow Network (TFN): A revolutionary architecture combining:
    1. KAN's learnable activation functions via basis decomposition
    2. MAMBA's selective state space modeling with hardware-aware design
    3. Diffusion-inspired score matching for probabilistic predictions
    4. Novel temporal flow dynamics with reversible transformations
    """
    
    def __init__(self, 
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 state_dim: int = 32,
                 num_basis: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        
        # Learnable basis functions (KAN-inspired)
        self.basis_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_basis)
            ) for _ in range(num_layers)
        ])
        
        # Learnable activation coefficients
        self.activation_coeffs = nn.ParameterList([
            nn.Parameter(torch.randn(num_basis, hidden_dim) * 0.1)
            for _ in range(num_layers)
        ])
        
        # State space parameters (MAMBA-inspired)
        self.A = nn.Parameter(torch.randn(num_layers, state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(num_layers, state_dim, hidden_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(num_layers, hidden_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.randn(num_layers, hidden_dim, hidden_dim) * 0.01)
        
        # Selective gating mechanism
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])
        
        # Temporal flow layers (novel component)
        self.flow_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Score network for diffusion-inspired prediction
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)  # Up/Down probabilities
        )
        
        # Temporal encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights using Xavier/He initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def compute_adaptive_activation(self, x, layer_idx):
        """KAN-inspired adaptive activation using learned basis functions"""
        basis_values = self.basis_net[layer_idx](x)
        basis_values = torch.softmax(basis_values, dim=-1)
        
        # Combine basis functions with learned coefficients
        activation = torch.einsum('...b,bh->...h', basis_values, self.activation_coeffs[layer_idx])
        return activation
    
    def state_space_layer(self, x, state, layer_idx):
        """MAMBA-inspired selective state space modeling"""
        batch_size = x.shape[0]
        
        # State transition
        A = torch.tanh(self.A[layer_idx])  # Ensure stability
        B = self.B[layer_idx]
        C = self.C[layer_idx]
        D = self.D[layer_idx]
        
        # Update state
        new_state = torch.matmul(state, A.T) + torch.matmul(x, B.T)
        
        # Compute output
        output = torch.matmul(new_state, C.T) + torch.matmul(x, D)
        
        # Apply selective gating
        gate = self.gates[layer_idx](x)
        output = gate * output + (1 - gate) * x
        
        return output, new_state
    
    def temporal_flow(self, x, layer_idx):
        """Novel temporal flow transformation with reversibility"""
        # Forward flow
        transformed = self.flow_transform[layer_idx](x)
        
        # Residual connection for reversibility
        output = x + transformed
        
        # Normalize to maintain gradient flow
        output = output / (1 + torch.norm(transformed, dim=-1, keepdim=True) * 0.1)
        
        return output
    
    def forward(self, x, time_diff=None):
        batch_size, seq_len, _ = x.shape
        
        # Initial projection
        h = self.input_proj(x)
        
        # Add temporal encoding if provided
        if time_diff is not None:
            time_encoding = self.time_encoder(time_diff.unsqueeze(-1))
            h = h + time_encoding
        
        # Initialize state
        state = torch.zeros(batch_size, self.state_dim, device=x.device)
        
        # Process through layers
        for layer_idx in range(self.num_layers):
            # Adaptive activation (KAN-inspired)
            h_activated = self.compute_adaptive_activation(x, layer_idx)
            h = h + h_activated
            
            # Process each time step
            outputs = []
            for t in range(seq_len):
                h_t = h[:, t, :]
                
                # State space transformation (MAMBA-inspired)
                h_t, state = self.state_space_layer(h_t, state, layer_idx)
                
                # Temporal flow transformation
                h_t = self.temporal_flow(h_t, layer_idx)
                
                # Dropout for regularization
                h_t = self.dropout(h_t)
                
                outputs.append(h_t)
            
            h = torch.stack(outputs, dim=1)
        
        # Final prediction using score network
        final_features = h[:, -1, :]  # Use last timestep
        logits = self.score_net(final_features)
        
        return logits


class PriceDataset(Dataset):
    def __init__(self, df, window_size=20, prediction_horizon=1):
        self.df = df.copy()
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
        # Calculate returns for better stationarity
        self.df['returns'] = df['price'].pct_change().fillna(0)
        
        # Calculate time differences (in hours, assuming datetime is in standard format)
        self.df['time_diff'] = df['datetime'].diff().dt.total_seconds() / 3600
        self.df['time_diff'].fillna(1, inplace=True)
        
        # Normalize features
        self.scaler = StandardScaler()
        self.df['returns_scaled'] = self.scaler.fit_transform(self.df[['returns']])
        
        # Create labels (1 for up, 0 for down)
        self.df['label'] = (self.df['price'].shift(-prediction_horizon) > self.df['price']).astype(int)
        
        # Remove invalid samples
        self.valid_indices = list(range(window_size, len(df) - prediction_horizon))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        # Get window of returns
        window_data = self.df.iloc[real_idx - self.window_size:real_idx]
        returns = torch.FloatTensor(window_data['returns_scaled'].values).unsqueeze(-1)
        
        # Get time differences
        time_diffs = torch.FloatTensor(window_data['time_diff'].values)
        
        # Get label
        label = torch.LongTensor([self.df.iloc[real_idx]['label']])
        
        return returns, time_diffs, label


class TemporalFlowTrainer:
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-5):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.criterion = nn.CrossEntropyLoss()
        
        # Additional loss for score matching (diffusion-inspired)
        self.score_matching_weight = 0.1
    
    def compute_score_matching_loss(self, logits, labels):
        """Diffusion-inspired score matching loss for better probability calibration"""
        probs = torch.softmax(logits, dim=-1)
        
        # Add noise to create diffusion process
        noise_level = 0.1
        noisy_probs = probs + torch.randn_like(probs) * noise_level
        noisy_probs = torch.clamp(noisy_probs, 1e-6, 1 - 1e-6)
        
        # Score matching objective
        score = torch.log(noisy_probs[:, 1] / noisy_probs[:, 0])
        target_score = (labels.float() - 0.5) * 2  # Map to [-1, 1]
        
        score_loss = torch.mean((score - target_score.squeeze()) ** 2)
        return score_loss
    
    def train_epoch(self, dataloader, device):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (returns, time_diffs, labels) in enumerate(dataloader):
            returns = returns.to(device)
            time_diffs = time_diffs.to(device)
            labels = labels.squeeze().to(device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(returns, time_diffs)
            
            # Compute losses
            ce_loss = self.criterion(logits, labels)
            score_loss = self.compute_score_matching_loss(logits, labels)
            
            # Combined loss
            loss = ce_loss + self.score_matching_weight * score_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        self.scheduler.step()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader, device):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for returns, time_diffs, labels in dataloader:
                returns = returns.to(device)
                time_diffs = time_diffs.to(device)
                labels = labels.squeeze().to(device)
                
                logits = self.model(returns, time_diffs)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy


def train_model(df, epochs=50, batch_size=32, window_size=20, learning_rate=1e-3):
    """
    Main training function for the Temporal Flow Network
    
    Args:
        df: DataFrame with 'datetime' and 'price' columns
        epochs: Number of training epochs
        batch_size: Batch size for training
        window_size: Size of lookback window
        learning_rate: Learning rate for optimizer
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = PriceDataset(df, window_size=window_size)
    
    # Split into train/val sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TemporalFlowNetwork(
        input_dim=1,
        hidden_dim=64,
        state_dim=32,
        num_basis=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = TemporalFlowTrainer(model, learning_rate=learning_rate)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, device)
        val_loss, val_acc = trainer.evaluate(val_loader, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_tfn_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Best Val Acc: {best_val_acc:.4f}")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    return model
