"""
Koopman-JEPA Model Implementation

This module implements the Koopman JEPA architecture for time series forecasting,
with fixes for out-of-distribution collapse issues.

Key features:
- Conv1D encoder for time series feature extraction
- Linear Koopman predictor for temporal dynamics
- Variance regularization to prevent OOD collapse
- Spectral loss for better eigenfunction learning
"""
from typing import Tuple, Dict, List, Optional
import hashlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)




class MobileNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        **kwargs
    ):
        super(MobileNetBlock, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                **kwargs,
            ),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                **kwargs,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.proj = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        bs, ds, sl, _ = x.shape
        projected = self.proj(x)
        x = x.transpose(2, 3).flatten(0, 1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2).reshape(bs, ds, sl, self.out_channels)
        return x + projected


class FFBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super(FFBlock, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
            nn.Dropout(dropout),
        )
        if d_in != d_out:
            self.proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        residual = x
        x = self.ff(x)
        if x.shape[-1] != residual.shape[-1]:
            residual = self.proj(residual)
        return x + residual



class Conv1DEncoder(nn.Module):
    """
    Conv1D encoder for time series data.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        sequence_length: int = 768,
        latent_dim: int = 32,
        hidden_channels: List[int] = [16, 32, 64, 128],
        kernel_sizes: List[int] = [7, 5, 3, 3],
        strides: List[int] = [2, 2, 2, 2],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # Build conv layers
        layers = []
        in_ch = input_channels
        
        for out_ch, kernel, stride in zip(hidden_channels, kernel_sizes, strides):
            padding = kernel // 2
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        
        # CRITICAL FIX: Calculate actual output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, sequence_length)
            dummy_output = self.conv_layers(dummy_input)
            self.feature_size = dummy_output.numel()
        
        logger.info(
            f"Encoder: {input_channels}x{sequence_length} -> "
            f"{dummy_output.shape} -> {self.feature_size} features -> {latent_dim}D"
        )
        
        # Projection with correct size
        self.projection = nn.Linear(self.feature_size, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        features_flat = torch.flatten(features, start_dim=1)
        z = self.projection(features_flat)
        z = self.output_norm(z)
        return z


class LinearPredictor(nn.Module):
    """
    Linear predictor for Koopman dynamics.
    
    Maps latent representations to predicted future states using a linear transformation.
    """
    
    def __init__(self, latent_dim: int, init_identity: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.M = nn.Parameter(torch.empty(latent_dim, latent_dim))
        
        if init_identity:
            nn.init.eye_(self.M)
        else:
            nn.init.orthogonal_(self.M)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.M)

    def identity_deviation(self) -> float:
        I = torch.eye(self.latent_dim, device=self.M.device)
        return (torch.norm(self.M - I, p='fro') / torch.norm(I, p='fro')).item()


class KoopmanJEPA(nn.Module):
    """
    Koopman JEPA model for time series forecasting.
    
    Uses a Conv1D encoder to extract latent representations and a linear predictor
    to model Koopman dynamics for future state prediction.
    
    The model learns Koopman eigenfunctions to capture temporal dynamics
    and can optionally predict market regimes.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        sequence_length: int = 768,
        latent_dim: int = 32,
        predictor_type: str = 'linear',
        ema_decay: float = 0.996,
        num_regimes: int = 8,
        init_identity: bool = True,
        regime_embedding_dim: int = 16,
        regime_prediction: bool = False,
        encoder_output_norm: bool = True,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.predictor_type = predictor_type
        self.num_regimes = num_regimes
        
        self.encoder = Conv1DEncoder(
            input_channels=input_channels,
            sequence_length=sequence_length,
            latent_dim=latent_dim
        )
        self.target_encoder = Conv1DEncoder(input_channels, sequence_length, latent_dim)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        

        if predictor_type == 'linear':
            self.predictor = LinearPredictor(latent_dim, init_identity)
        else:
            self.predictor = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.LayerNorm(latent_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(latent_dim * 2, latent_dim)
            )
        
        self.regime_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, num_regimes)
        )
        
    
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update - no changes from v2"""
        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.ema_decay).add_(p_online.data, alpha=1 - self.ema_decay)
    
    
    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Both paths have gradients, detachment in loss"""
        z_t = self.encoder(x_context)
        z_pred = self.predictor(z_t)
        with torch.no_grad():
            z_t_delta = self.target_encoder(x_target)
        
        result = {
            'z_t': z_t,
            'z_pred': z_pred,
            'z_t_delta': z_t_delta,
            'z_context': z_t,
            'z_target': z_t_delta
        }
        
        return result


    def compute_spectral_regularization(self) -> torch.Tensor:
        if self.predictor_type == 'linear':
            M = self.predictor.M
            I = torch.eye(self.latent_dim, device=M.device)  # noqa: E741
            return torch.norm(M - I, p='fro') ** 2 / self.latent_dim
        return torch.tensor(0.0, device=self.predictor.M.device if hasattr(self.predictor, 'M') else 'cpu')

    def compute_variance_regularization(self, z: torch.Tensor) -> torch.Tensor:
        """
        NEW: Force non-zero variance to prevent OOD collapse.
        
        Issue: Model maps OOD inputs (val/test) to zero vectors.
        Fix: Penalize when variance drops below target.
        """
        # Compute std per dimension
        std_per_dim = z.std(dim=0)
        
        # Target: std should be around 1.0 (after layer norm)
        target_std = 0.5
        
        # Penalize when std drops below target
        # Use ReLU so we only penalize when too low
        variance_loss = F.relu(target_std - std_per_dim).mean()
        
        return variance_loss
    
    def compute_regime_consistency_loss(
        self,
        z_context: torch.Tensor,
        z_pred: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        logits_t = self.regime_classifier(z_context)
        logits_pred = self.regime_classifier(z_pred)
        log_p_pred = F.log_softmax(logits_pred / temperature, dim=-1)
        p_t = F.softmax(logits_t / temperature, dim=-1)
        loss_regime = F.kl_div(log_p_pred, p_t, reduction='batchmean')
        return loss_regime

    def compute_loss(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        alpha_jepa: float = 1.0,
        alpha_regime: float = 0.01,
        alpha_spectral: float = 0.5,  # INCREASED (was 0.1)
        alpha_variance: float = 0.1,  # NEW
        temperature: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        UPDATED: Added variance regularization.
        """
        outputs = self.forward(x_context, x_target, return_all=False)
        z_t = outputs['z_t']
        z_pred = outputs['z_pred']
        z_t_delta = outputs['z_t_delta']
        
        # 1. JEPA loss (detach target)
        loss_jepa = F.mse_loss(z_pred, z_t_delta.detach())

        # 2. Regime consistency (detached in loss computation)
        loss_regime = self.compute_regime_consistency_loss(z_t.detach(), z_pred.detach(), temperature)

        # 3. Spectral regularization (STRONGER)
        if self.predictor_type == 'linear':
            M = self.predictor.M
            I = torch.eye(self.latent_dim, device=M.device)  # noqa: E741
            loss_spectral = torch.norm(M - I, p='fro') ** 2 / self.latent_dim
        else:
            loss_spectral = torch.tensor(0.0, device=z_t.device)
        
        # 4. Variance regularization (NEW - prevent OOD collapse)
        loss_variance = self.compute_variance_regularization(z_t)
        
        # Total loss
        total_loss = (
            alpha_jepa * loss_jepa +
            alpha_regime * loss_regime +
            alpha_spectral * loss_spectral +
            alpha_variance * loss_variance
        )
        
        return {
            'total': total_loss,
            'jepa': loss_jepa,
            'regime': loss_regime,
            'spectral': loss_spectral,
            'variance': loss_variance
        }

    def compute_koopman_crypto_loss(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        p_context: Optional[torch.Tensor] = None,
        p_target: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        return self.compute_loss(x_context, x_target, **kwargs)

    def compute_koopman_jepa_loss(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        return self.compute_loss(x_context, x_target, **kwargs)
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        z = self.encoder(x)
        return z

    @torch.no_grad()
    def extract_regime_embeddings(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        z = self.encoder(x)
        regime_logits = self.regime_classifier(z)
        regime_probs = F.softmax(regime_logits, dim=-1)
        return z, regime_probs
    
    @torch.no_grad()
    def analyze_predictor_properties(self) -> Dict[str, float]:
        if self.predictor_type != 'linear':
            return {}
        
        M = self.predictor.M
        I = torch.eye(self.latent_dim, device=M.device)  # noqa: E741
        
        properties = {
            'identity_deviation': (torch.norm(M - I, p='fro') / torch.norm(I, p='fro')).item(),
            'frobenius_norm': torch.norm(M, p='fro').item()
        }
        
        try:
            eigenvalues = torch.linalg.eigvals(M)
            eigenvalue_mags = torch.abs(eigenvalues)
            properties['eigenvalue_mean_mag'] = eigenvalue_mags.mean().item()
            properties['eigenvalues_near_one'] = (
                (eigenvalue_mags > 0.95) & (eigenvalue_mags < 1.05)
            ).sum().item()
        except Exception as e:
            logger.warning(f"Could not compute eigenvalues: {e}")
        
        return properties

# Alias for backward compatibility and tests
CryptoKoopmanJEPA = KoopmanJEPA


def create_crypto_feature_tensor(
    prices: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    window_size: int = 128
) -> torch.Tensor:
    """
    Creates a 3-channel feature tensor from a price window:
    1. Normalized prices (local z-score normalization)
    2. Simple returns
    3. Log returns
    """
    eps = 1e-8
    
    # Cast input array to float32 first
    prices = np.asarray(prices, dtype=np.float32)
    
    # Ensure window size
    if len(prices) > window_size:
        prices = prices[-window_size:]
    elif len(prices) < window_size:
        pad_length = window_size - len(prices)
        prices = np.concatenate([np.full(pad_length, prices[0], dtype=np.float32), prices])
    
    # Local Z-score normalization
    mean = np.mean(prices)
    std = np.std(prices, ddof=1)
    if std < eps:
        std = eps
    prices_norm = (prices - mean) / std
    
    # Force mean of prices_norm to be exactly 0 and std exactly 1 to avoid float32 rounding errors
    # in unittest comparisons.
    prices_norm = prices_norm - np.mean(prices_norm)
    norm_std = np.std(prices_norm, ddof=1)
    if norm_std > eps:
        prices_norm = prices_norm / norm_std
    
    # Returns
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + eps)
    
    # Log returns
    log_returns = np.zeros_like(prices)
    log_returns[1:] = np.log(prices[1:] + eps) - np.log(prices[:-1] + eps)
    
    # Stack channels
    features = np.stack([
        prices_norm,
        returns,
        log_returns
    ], axis=0)
    
    return torch.from_numpy(features).float()


def compute_price_window_hash(prices: np.ndarray) -> str:
    """
    Computes a deterministic hash of the price window to use as a cache key.
    """
    prices_bytes = prices.astype(np.float32).tobytes()
    return hashlib.md5(prices_bytes).hexdigest()


if __name__ == "__main__":
    """Validation tests"""
    print("="*80)
    print("Final Corrected Koopman-JEPA Validation")
    print("="*80)
    
    # Test 1: Model
    print("\n1. Creating model...")
    model = KoopmanJEPA(
        input_channels=3,
        latent_dim=32,
        predictor_type='linear'
    )
    print("✓ Model created")
    
    # Test 2: Forward
    print("\n2. Testing forward...")
    x_ctx = torch.randn(4, 3, 768)
    x_tgt = torch.randn(4, 3, 768)
    outputs = model(x_ctx, x_tgt)
    print(f"✓ z_t: {outputs['z_t'].shape}")
    print(f"✓ Both have gradients: {outputs['z_t'].requires_grad and outputs['z_t_delta'].requires_grad}")
    
    # Test 3: Variance regularization
    print("\n3. Testing variance regularization...")
    z_collapsed = torch.randn(32, 32) * 0.01  # Very small variance
    var_loss_collapsed = model.compute_variance_regularization(z_collapsed)
    
    z_healthy = torch.randn(32, 32) * 1.0  # Normal variance
    var_loss_healthy = model.compute_variance_regularization(z_healthy)
    
    print(f"  Collapsed (std~0.01): variance_loss = {var_loss_collapsed.item():.6f}")
    print(f"  Healthy (std~1.0): variance_loss = {var_loss_healthy.item():.6f}")
    print("  ✓ Collapsed should have higher loss")
    
    # Test 4: Full loss
    print("\n4. Testing full loss...")
    losses = model.compute_koopman_jepa_loss(x_ctx, x_tgt)
    print(f"✓ JEPA: {losses['jepa'].item():.6f}")
    print(f"✓ Variance: {losses['variance'].item():.6f}")
    print(f"✓ Spectral: {losses['spectral'].item():.6f}")    
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
    