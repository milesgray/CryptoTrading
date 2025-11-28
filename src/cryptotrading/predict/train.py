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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models import get_model
from dotdict import dotdict

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

def train_model(model, df, epochs=30, batch_size=64, learning_rate=0.001):
    """
    Train the time series diffusion model
    
    Args:
        model: The model to train
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

    model = get_model(dotdict({
        "input_dim": 1,
        "window_size": 20,
        "seq_len": 20,
        "d_model": 128,
        "num_layers": 4,
        "enc_in": 1,
        "dec_in": 1,
        "enc_out": 1,
        "dec_out": 1,
        "model": "WAVESTATE"
    }))
    
    # Train model
    model, scaler, test_predictions, test_confidences, test_targets = train_model(model, df, epochs=20)
    
    # Predict next movement
    prediction, confidence = predict_next_movement(model, df, scaler)
    
    print(f"Prediction for next price movement: {'UP' if prediction == 1 else 'DOWN'} with confidence {confidence:.2f}")
    
    # Visualize results
    fig = visualize_predictions(df, test_predictions, test_targets)
    plt.show()