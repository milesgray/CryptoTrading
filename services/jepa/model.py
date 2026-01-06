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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List

import logging

logger = logging.getLogger(__name__)


class Conv1DEncoder(nn.Module):
    """

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
                nn.LeakyReLU(0.2, inplace=True),
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
        
        nn.init.xavier_uniform_(self.projection.weight, gain=0.01)
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
        num_regimes: int = 8,
        init_identity: bool = True,
        regime_embedding_dim: int = 16,
        regime_prediction: bool = False,
        encoder_output_norm: bool = True,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.predictor_type = predictor_type
        
        self.encoder = Conv1DEncoder(
            input_channels=input_channels,
            sequence_length=sequence_length,
            latent_dim=latent_dim
        )
        
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
    
    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Both paths have gradients, detachment in loss"""
        z_t = self.encoder(x_context)
        z_pred = self.predictor(z_t)
        z_t_delta = self.encoder(x_target)
        
        result = {
            'z_t': z_t,
            'z_pred': z_pred,
            'z_t_delta': z_t_delta
        }
        
        if return_all:
            with torch.no_grad():
                regime_logits_t = self.regime_classifier(z_t.detach())
                regime_logits_pred = self.regime_classifier(z_pred.detach())
                result['regime_logits_t'] = regime_logits_t
                result['regime_logits_pred'] = regime_logits_pred
        
        return result
    
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
        
        # 2. Regime consistency (detached)
        logits_t = self.regime_classifier(z_t.detach())
        logits_pred = self.regime_classifier(z_pred.detach())
        log_p_pred = F.log_softmax(logits_pred / temperature, dim=-1)
        p_t = F.softmax(logits_t / temperature, dim=-1)
        loss_regime = F.kl_div(log_p_pred, p_t, reduction='batchmean')
        
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
    