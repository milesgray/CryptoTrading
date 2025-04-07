"""
WAVESTATE: WAVElet-enhanced State Space Transformer with Adaptive Temporal Encoding

A revolutionary time series prediction model that fuses concepts from:
- Kolmogorov-Arnold Networks (KAN)
- State Space Models (Mamba)
- Diffusion Models
- Wavelet Transforms

Author: Claude, March 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TimeSeriesDataset(Dataset):
    """Dataset for time series with sliding window approach"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SelfAttention(nn.Module):
    """Multi-head self-attention mechanism with linear complexity"""
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        # State space gating for selective attention
        self.gate = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        queries = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        queries = queries.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Efficient attention with linear complexity using kernel approximation
        # Inspired by linear transformer and Performers
        queries = F.elu(queries) + 1
        keys = F.elu(keys) + 1
        
        # Key-value aggregation
        kv = torch.matmul(keys.transpose(-1, -2), values)  # (batch_size, num_heads, head_dim, head_dim)
        
        # Query-based retrieval
        attn_output = torch.matmul(queries, kv)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape and combine heads
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate(x))
        attn_output = gate * attn_output
        
        # Output projection
        output = self.out(attn_output)
        return output

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

class WAVESTATE(nn.Module):
    """WAVElet-enhanced State Space Transformer with Adaptive Temporal Encoding
    
    An advanced time series prediction model that combines wavelet transforms,
    state space models, adaptive kernels, and diffusion principles."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, seq_len=20, num_layers=3):
        super(TimeSeriesDiffusionModel, self).__init__()
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

def prepare_time_series_data(df, window_size=20, forecast_horizon=1):
    """
    Prepare time series data with sliding window approach
    
    Args:
        df: DataFrame with 'datetime' and 'price' columns
        window_size: Number of time steps to use for prediction
        forecast_horizon: Number of steps ahead to predict
        
    Returns:
        X, y arrays for training
    """
    # Sort by datetime to ensure sequential order
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Extract price series
    price_series = df['price'].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaled = scaler.fit_transform(price_series.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = [], []
    for i in range(len(price_scaled) - window_size - forecast_horizon + 1):
        X.append(price_scaled[i:i+window_size])
        # For binary classification (up/down)
        y.append(1 if price_scaled[i+window_size+forecast_horizon-1] > price_scaled[i+window_size-1] else 0)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Add features (you can expand this)
    # 1. Technical indicators
    X_with_features = []
    for i in range(len(X)):
        window = X[i]
        
        # Calculate additional features
        momentum = window[-1] - window[0]
        volatility = np.std(window)
        rsi_window = np.diff(window)
        rsi_up = np.sum(np.maximum(rsi_window, 0)) / window_size
        rsi_down = -np.sum(np.minimum(rsi_window, 0)) / window_size
        rsi = 100 - (100 / (1 + rsi_up / (rsi_down + 1e-10)))
        
        # Create feature vector
        features = np.column_stack((
            window.reshape(-1, 1),  # Raw price
            np.ones((window_size, 1)) * momentum,  # Momentum
            np.ones((window_size, 1)) * volatility,  # Volatility
            np.ones((window_size, 1)) * rsi,  # RSI
        ))
        X_with_features.append(features)
    
    X_with_features = np.array(X_with_features)
    
    return X_with_features, y, scaler

def train_model(df, epochs=30, batch_size=64, learning_rate=0.001):
    """
    Train the time series diffusion model
    
    Args:
        df: DataFrame with 'datetime' and 'price' columns
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        
    Returns:
        Trained model and scalers
    """
    # Prepare data
    X, y, scaler = prepare_time_series_data(df)
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze(-1)
    
    # Split into train/validation/test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = X.shape[2]  # Number of features
    window_size = X.shape[1]  # Sequence length
    model = WAVESTATE(input_dim=input_dim, hidden_dim=128, seq_len=window_size)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            # Generate noise levels for diffusion training (curriculum learning)
            noise_level = torch.rand(batch_X.shape[0]) * (1.0 - epoch / epochs)
            
            predictions, confidence = model(batch_X, noise_level)
            predictions = predictions[:, -1]  # Get prediction for the last timestep
            
            # Calculate loss
            loss = bce_loss(predictions, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate accuracy
            predicted_classes = (torch.sigmoid(predictions) > 0.5).float()
            train_correct += (predicted_classes == batch_y).sum().item()
            train_total += batch_y.size(0)
            
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions, confidence = model(batch_X)
                predictions = predictions[:, -1]  # Get prediction for the last timestep
                
                # Calculate loss
                loss = bce_loss(predictions, batch_y)
                val_loss += loss.item()
                
                # Calculate accuracy
                predicted_classes = (torch.sigmoid(predictions) > 0.5).float()
                val_correct += (predicted_classes == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        # Print statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Test phase
    model.eval()
    test_correct = 0
    test_total = 0
    test_predictions = []
    test_confidences = []
    test_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions, confidence = model(batch_X)
            predictions = predictions[:, -1]  # Get prediction for the last timestep
            confidence = confidence[:, -1]
            
            # Calculate accuracy
            predicted_classes = (torch.sigmoid(predictions) > 0.5).float()
            test_correct += (predicted_classes == batch_y).sum().item()
            test_total += batch_y.size(0)
            
            # Store predictions and targets
            test_predictions.append(predicted_classes.numpy())
            test_confidences.append(confidence.numpy())
            test_targets.append(batch_y.numpy())
    
    test_predictions = np.concatenate(test_predictions)
    test_confidences = np.concatenate(test_confidences)
    test_targets = np.concatenate(test_targets)
    
    test_acc = test_correct / test_total * 100
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    return model, scaler, test_predictions, test_confidences, test_targets

def predict_next_movement(model, df, scaler, window_size=20):
    """
    Predict if the next price movement will be up or down
    
    Args:
        model: Trained model
        df: DataFrame with 'datetime' and 'price' columns
        scaler: Fitted MinMaxScaler
        window_size: Size of the sliding window used for prediction
        
    Returns:
        prediction: 1 for up, 0 for down
        confidence: Confidence in the prediction
    """
    # Sort by datetime to ensure sequential order
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Extract the most recent window
    recent_prices = df['price'].values[-window_size:]
    
    # Scale the data
    recent_prices_scaled = scaler.transform(recent_prices.reshape(-1, 1)).flatten()
    
    # Calculate features
    momentum = recent_prices_scaled[-1] - recent_prices_scaled[0]
    volatility = np.std(recent_prices_scaled)
    rsi_window = np.diff(recent_prices_scaled)
    rsi_up = np.sum(np.maximum(rsi_window, 0)) / window_size
    rsi_down = -np.sum(np.minimum(rsi_window, 0)) / window_size
    rsi = 100 - (100 / (1 + rsi_up / (rsi_down + 1e-10)))
    
    # Create feature vector
    features = np.column_stack((
        recent_prices_scaled.reshape(-1, 1),  # Raw price
        np.ones((window_size, 1)) * momentum,  # Momentum
        np.ones((window_size, 1)) * volatility,  # Volatility
        np.ones((window_size, 1)) * rsi,  # RSI
    ))
    
    # Convert to PyTorch tensor
    X = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction, confidence = model(X)
        prediction = prediction[:, -1]  # Get prediction for the last timestep
        confidence = confidence[:, -1]
        
        # Convert to binary class
        prediction_class = 1 if torch.sigmoid(prediction).item() > 0.5 else 0
        confidence_value = confidence.item()
    
    return prediction_class, confidence_value

def visualize_predictions(df, test_predictions, test_targets, window_size=20):
    """
    Visualize the predictions against actual price movements
    
    Args:
        df: DataFrame with 'datetime' and 'price' columns
        test_predictions: Model predictions (0 or 1)
        test_targets: Actual targets (0 or 1)
        window_size: Size of the sliding window used for prediction
    """
    # Get dates corresponding to predictions
    dates = df['datetime'].values[window_size+1:]
    dates = dates[-len(test_predictions):]
    
    # Get prices
    prices = df['price'].values
    prices = prices[-len(test_predictions)-1:]  # Include one more for visualization
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot prices
    ax1.plot(dates, prices[1:], label='Price', color='blue')
    ax1.set_title('Price Movement and Predictions')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Plot predictions and actual movements
    ax2.plot(dates, test_targets, label='Actual (Up=1, Down=0)', color='green', marker='o', linestyle=':')
    ax2.plot(dates, test_predictions, label='Predicted', color='red', marker='x')
    ax2.set_title('Prediction vs Actual Movement (Up=1, Down=0)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Direction')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Sample data for demonstration
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    prices = np.sin(np.linspace(0, 15, 300)) * 10 + 100 + np.random.randn(300) * 2
    df = pd.DataFrame({
        'datetime': dates,
        'price': prices
    })
    
    # Train model
    model, scaler, test_predictions, test_confidences, test_targets = train_model(df, epochs=20)
    
    # Predict next movement
    prediction, confidence = predict_next_movement(model, df, scaler)
    
    print(f"Prediction for next price movement: {'UP' if prediction == 1 else 'DOWN'} with confidence {confidence:.2f}")
    
    # Visualize results
    fig = visualize_predictions(df, test_predictions, test_targets)
    plt.show()