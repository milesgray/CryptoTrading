"""
Memory-Efficient Trainer with ResNet-MLP Architecture.
"""
from typing import List

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.1, norm=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.LayerNorm(dim_hidden) if norm else nn.Identity(),
            nn.GELU(),  # GELU > ReLU for financial time series usually
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
            nn.LayerNorm(dim_out) if norm else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    """Dense Residual Block for feature interaction"""

    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.1, norm=True):
        super().__init__()
        self.net = MLP(dim_in, dim_hidden, dim_out, dropout=dropout, norm=norm)

    def forward(self, x):
        return x + self.net(x)


class RobustPressurePredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        head_dim: int = 64,
        dropout: float = 0.15,
        use_residual: bool = True,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # Deep Residual Feature Extractor
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim // 2, hidden_dim, dropout=dropout)
            for hidden_dim in hidden_dims
        ])

        # Independent Heads
        self.buy_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], head_dim), 
            nn.GELU(), 
            nn.Linear(head_dim, 1), 
            nn.Sigmoid()
        )

        self.sell_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], head_dim), 
            nn.GELU(), 
            nn.Linear(head_dim, 1), 
            nn.Sigmoid()
        )

        self.total_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], head_dim), 
            nn.GELU(), 
            nn.Linear(head_dim, 1), 
            nn.Tanh()
        )

        # Uncertainty Head (predicts Log Variance)
        self.uncertainty_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)

        buy = self.buy_head(h)
        sell = self.sell_head(h)
        total = self.total_head(h)

        # Aleatoric Uncertainty
        log_var = self.uncertainty_head(h)

        return {
            "buy_pressure": buy,
            "sell_pressure": sell,
            "total_pressure": total,
            "uncertainty_log_var": log_var,
        }


class PressurePredictor(nn.Module):
    """
    Neural network to predict buy/sell pressure from order book features.

    Architecture improvements:
    - Residual connections for deeper network
    - Layer normalization instead of BatchNorm (more stable)
    - Separate uncertainty head
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64, 32],
        head_dim: int = 64,
        dropout: float = 0.15,
        use_residual: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual

        # Encoder with residual connections
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            if use_residual:
                layer = ResidualBlock(prev_dim, hidden_dim, hidden_dim, dropout=dropout)
            else:
                layer = MLP(prev_dim, hidden_dim, hidden_dim, dropout=dropout)
            layers.append(layer)

            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Three output heads
        self.buy_pressure_head = nn.Sequential(
            nn.Linear(prev_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Less dropout in heads
            nn.Linear(head_dim, 1),
            nn.Sigmoid(),
        )

        self.sell_pressure_head = nn.Sequential(
            nn.Linear(prev_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, 1),
            nn.Sigmoid(),
        )

        self.total_pressure_head = nn.Sequential(
            nn.Linear(prev_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        buy_pressure = self.buy_pressure_head(features)
        sell_pressure = self.sell_pressure_head(features)
        total_pressure = self.total_pressure_head(features)

        return {
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "total_pressure": total_pressure,
        }


def get_model(config):
    if config.model_type == "robust":
        return RobustPressurePredictor(config.input_dim, hidden_dims=config.hidden_dims, head_dim=config.head_dim)
    elif config.model_type == "pressure":
        return PressurePredictor(config.input_dim, hidden_dims=config.hidden_dims, head_dim=config.head_dim)