"""
Contrastive Learning Encoder for Trade Setup Embeddings

Uses a CNN architecture with triplet loss to learn representations where:
- Similar trade outcomes (both profitable longs, both profitable shorts) cluster together
- Different outcomes (profitable vs unprofitable, long vs short) are pushed apart

The learned embeddings capture "trade setup patterns" that can be matched in real-time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import IntEnum

from cryptotrading.predict.layers.S3 import S3

class TradeOutcome(IntEnum):
    """Trade outcome categories for contrastive learning"""
    LOSS_LARGE = 0      # < -5%
    LOSS_SMALL = 1      # -5% to -1%
    FLAT = 2            # -1% to +1%
    PROFIT_SMALL = 3    # +1% to +5%
    PROFIT_LARGE = 4    # > +5%


class TradeDirection(IntEnum):
    """Trade direction"""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class TradeSetup:
    """A single trade setup with its outcome"""
    price_window: np.ndarray      # Normalized price history before entry
    direction: TradeDirection     # Long or Short
    profit_pct: float            # Realized profit percentage
    leverage: float              # Leverage used
    hold_duration: int           # How long position was held
    entry_idx: int               # Index in original price array
    timestamp: Optional[float] = None
    
    @property
    def outcome(self) -> TradeOutcome:
        """Categorize the trade outcome"""
        if self.profit_pct < -0.05:
            return TradeOutcome.LOSS_LARGE
        elif self.profit_pct < -0.01:
            return TradeOutcome.LOSS_SMALL
        elif self.profit_pct < 0.01:
            return TradeOutcome.FLAT
        elif self.profit_pct < 0.05:
            return TradeOutcome.PROFIT_SMALL
        else:
            return TradeOutcome.PROFIT_LARGE
    
    @property
    def outcome_label(self) -> int:
        """Combined label: direction + outcome for contrastive grouping"""
        # 10 classes: 5 outcomes x 2 directions
        dir_offset = 0 if self.direction == TradeDirection.LONG else 5
        return dir_offset + int(self.outcome)


class PriceWindowEncoder(nn.Module):
    """
    CNN encoder for price windows.
    
    Architecture:
    - 1D Convolutions to capture local patterns
    - Global pooling to handle variable-length inputs
    - Projection head for contrastive learning
    
    Input: (batch, window_size) - normalized log returns
    Output: (batch, embedding_dim) - L2 normalized embeddings
    """
    
    def __init__(
        self,
        window_size: int = 100,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_conv_layers: int = 4,
        dropout: float = 0.1,
        use_s3: bool = True
    ):
        super().__init__()
        
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        self.use_s3 = use_s3
        if use_s3:
            self.s3_layers = S3(num_layers=3, initial_num_segments=4, shuffle_vector_dim=1, segment_multiplier=2)
        
        # Initial projection from 1D to hidden channels
        self.input_proj = nn.Conv1d(1, hidden_dim // 4, kernel_size=7, padding=3)
        
        # Stack of conv layers with residual connections
        self.conv_layers = nn.ModuleList()
        channels = hidden_dim // 4
        
        for i in range(num_conv_layers):
            out_channels = min(channels * 2, hidden_dim)
            self.conv_layers.append(
                ConvBlock(channels, out_channels, kernel_size=5, dropout=dropout)
            )
            channels = out_channels
        
        # Multi-scale pooling
        self.pool_sizes = [1, 2, 4, 8]  # Different temporal scales
        pool_output_dim = channels * len(self.pool_sizes)
        
        # Projection head (MLP for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(pool_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size) - normalized price returns
            
        Returns:
            (batch, embedding_dim) - L2 normalized embeddings
        """
        # Add channel dimension: (batch, 1, window_size)
        x = x.unsqueeze(1)

        if self.use_s3:
            x = self.s3_layers(x)
        
        # Initial projection
        x = self.input_proj(x)
        x = F.gelu(x)
        
        # Conv layers with residuals
        for conv in self.conv_layers:
            x = conv(x)
        
        # Multi-scale pooling
        pooled = []
        for pool_size in self.pool_sizes:
            if pool_size == 1:
                # Global average pool
                pooled.append(x.mean(dim=2))
            else:
                # Adaptive pool then flatten
                p = F.adaptive_avg_pool1d(x, pool_size)
                pooled.append(p.mean(dim=2))
        
        # Concatenate pooled features
        x = torch.cat(pooled, dim=1)
        
        # Project to embedding space
        x = self.projection(x)
        
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def encode(self, price_window: np.ndarray) -> np.ndarray:
        """
        Encode a single price window (numpy interface for inference).
        
        Args:
            price_window: (window_size,) array of normalized returns
            
        Returns:
            (embedding_dim,) array
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(price_window).float().unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            emb = self(x)
            return emb.cpu().numpy()[0]


class ConvBlock(nn.Module):
    """Residual conv block with batch norm and dropout"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if dimensions change
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = F.gelu(x + residual)
        return x


class TripletLoss(nn.Module):
    """
    Triplet loss for contrastive learning.
    
    Pulls together embeddings with same outcome label,
    pushes apart embeddings with different outcome labels.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: (batch, embedding_dim)
            positive: (batch, embedding_dim) - same outcome as anchor
            negative: (batch, embedding_dim) - different outcome
            
        Returns:
            Scalar loss
        """
        # Cosine distances (embeddings are already L2 normalized)
        pos_dist = 1 - (anchor * positive).sum(dim=1)  # 0 = identical, 2 = opposite
        neg_dist = 1 - (anchor * negative).sum(dim=1)
        
        # Triplet loss with margin
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss (SupCon).
    
    More powerful than triplet loss - uses all positive/negative pairs in batch.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embedding_dim) - L2 normalized
            labels: (batch,) - outcome labels
            
        Returns:
            Scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Compute all pairwise similarities
        similarity = torch.mm(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float().to(device)
        
        # Remove self-similarity from positives
        mask_pos = mask_pos - torch.eye(batch_size, device=device)
        
        # For numerical stability
        similarity_max, _ = similarity.max(dim=1, keepdim=True)
        similarity = similarity - similarity_max.detach()
        
        # Compute log softmax
        exp_sim = torch.exp(similarity)
        
        # Mask out self-similarity
        mask_self = torch.eye(batch_size, device=device)
        exp_sim = exp_sim * (1 - mask_self)
        
        # Denominator: sum over all negatives + positives (excluding self)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Average over positive pairs
        num_positives = mask_pos.sum(dim=1)
        
        # Handle case where sample has no positives
        num_positives = torch.clamp(num_positives, min=1)
        
        loss = -(mask_pos * log_prob).sum(dim=1) / num_positives
        return loss.mean()


class TradeSetupDataset(torch.utils.data.Dataset):
    """Dataset for training the encoder"""
    
    def __init__(self, setups: List[TradeSetup], augment: bool = True):
        self.setups = setups
        self.augment = augment
        
        # Group by outcome label for efficient triplet mining
        self.label_to_indices = {}
        for i, setup in enumerate(setups):
            label = setup.outcome_label
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)
        
        self.labels = list(self.label_to_indices.keys())
    
    def __len__(self):
        return len(self.setups)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        setup = self.setups[idx]
        x = torch.from_numpy(setup.price_window).float()
        
        if self.augment:
            x = self._augment(x)
        
        return x, setup.outcome_label
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Data augmentation for price windows"""
        # Random scaling (simulate different volatility regimes)
        if torch.rand(1) < 0.5:
            scale = torch.empty(1).uniform_(0.8, 1.2)
            x = x * scale
        
        # Add small noise
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        # Random shift (simulate different price levels)
        if torch.rand(1) < 0.3:
            shift = torch.empty(1).uniform_(-0.02, 0.02)
            x = x + shift
        
        return x
    
    def get_triplet(self, anchor_idx: int) -> Tuple[int, int, int]:
        """Get indices for triplet: anchor, positive, negative"""
        anchor_label = self.setups[anchor_idx].outcome_label
        
        # Get positive (same label)
        pos_indices = self.label_to_indices[anchor_label]
        pos_idx = anchor_idx
        while pos_idx == anchor_idx:
            pos_idx = pos_indices[torch.randint(len(pos_indices), (1,)).item()]
        
        # Get negative (different label)
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = self.labels[torch.randint(len(self.labels), (1,)).item()]
        neg_indices = self.label_to_indices[neg_label]
        neg_idx = neg_indices[torch.randint(len(neg_indices), (1,)).item()]
        
        return anchor_idx, pos_idx, neg_idx


def normalize_price_window(prices: np.ndarray) -> np.ndarray:
    """
    Normalize a price window to log returns.
    
    This makes the representation scale-invariant and stationary.
    
    Args:
        prices: Raw price array
        
    Returns:
        Log returns array (length = len(prices) - 1)
    """
    log_prices = np.log(prices + 1e-8)
    returns = np.diff(log_prices)
    
    # Standardize
    mean = returns.mean()
    std = returns.std() + 1e-8
    normalized = (returns - mean) / std
    
    return normalized.astype(np.float32)


def extract_trade_setups(
    prices: np.ndarray,
    timestamps: np.ndarray,
    oracle_actions: np.ndarray,
    oracle_leverages: np.ndarray,
    window_size: int = 100,
    min_profit_threshold: float = 0.0  # Include all trades, filter later if needed
) -> List[TradeSetup]:
    """
    Extract trade setups from oracle-labeled data.
    
    Args:
        prices: Price array
        timestamps: Timestamp array
        oracle_actions: Actions from DP oracle (0=hold, 1=long, 2=short, 3=close)
        oracle_leverages: Leverage values from oracle
        window_size: How much history before entry to include
        min_profit_threshold: Minimum profit to include trade (set to 0 to include all)
        
    Returns:
        List of TradeSetup objects
    """
    setups = []
    n = len(prices)
    
    # Find all entry points
    entry_indices = np.where((oracle_actions == 1) | (oracle_actions == 2))[0]
    
    for entry_idx in entry_indices:
        action = oracle_actions[entry_idx]
        direction = TradeDirection.LONG if action == 1 else TradeDirection.SHORT
        entry_price = prices[entry_idx]
        leverage = oracle_leverages[entry_idx]
        
        # Find exit: look for CLOSE (3) or next entry (1, 2)
        exit_idx = entry_idx + 1
        while exit_idx < n:
            if oracle_actions[exit_idx] in [1, 2, 3]:  # New entry or close
                break
            exit_idx += 1
        
        # If we hit a new entry, the exit is the bar before
        if exit_idx < n and oracle_actions[exit_idx] in [1, 2]:
            exit_idx = exit_idx  # Exit at the new entry point (position flips)
        
        if exit_idx >= n:
            exit_idx = n - 1
        
        exit_price = prices[exit_idx]
        
        # Calculate profit
        if direction == TradeDirection.LONG:
            profit_pct = (exit_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - exit_price) / entry_price
        
        # Extract price window before entry
        # Allow partial windows for early trades (pad with edge values)
        window_start = max(0, entry_idx - window_size)
        price_window = prices[window_start:entry_idx]
        
        if len(price_window) < 2:
            # Need at least 2 prices to compute returns
            continue
        
        normalized_window = normalize_price_window(price_window)
        
        # Pad to target size if needed
        target_len = window_size - 1  # -1 because returns are diff
        if len(normalized_window) < target_len:
            pad_size = target_len - len(normalized_window)
            normalized_window = np.pad(normalized_window, (pad_size, 0), mode='edge')
        
        setup = TradeSetup(
            price_window=normalized_window,
            direction=direction,
            profit_pct=profit_pct,
            leverage=leverage,
            hold_duration=exit_idx - entry_idx,
            entry_idx=entry_idx,
            timestamp=timestamps[entry_idx] if timestamps is not None else None
        )
        
        if abs(profit_pct) >= min_profit_threshold:
            setups.append(setup)
    
    return setups
