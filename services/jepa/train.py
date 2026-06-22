"""
Updated Training Script for Corrected Koopman-JEPA

Key fixes:
1. Global normalization (fit on train, apply to all)
2. Proper dataset splitting (no data leakage)
3. Multi-channel features
4. Monitoring for collapse detection
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Optional
from pathlib import Path
from tqdm import tqdm
import json
from cryptotrading.predict.utils.augment import rnd_augment, sequence_augment
from .model import KoopmanJEPA
from .data import CryptoPriceDataset, GlobalPriceNormalizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JEPATrainer:
    """
    CORRECTED: Trainer with collapse monitoring and proper logging.
    """
    
    def __init__(
        self,
        model: KoopmanJEPA,
        train_dataset: CryptoPriceDataset,
        val_dataset: CryptoPriceDataset,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_mixed_precision: bool = True,
        save_dir: Optional[Path | str] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if device == 'cuda' else False,
            persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if device == 'cuda' else False
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=learning_rate * 0.01
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'predictor_properties': [],
            'embedding_stats': []
        }

        self.save_dir = Path(save_dir) if save_dir else None
    
    def train_epoch(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        augment_rounds: int = 2,
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        if loss_weights is None:
            loss_weights = {
                'alpha_jepa': 1.0,
                'alpha_spectral': 0.1,
                'alpha_variance': 0.1 
            }
        
        epoch_losses = {
            'total': 0.0,
            'jepa': 0.0,
            'spectral': 0.0,
            'variance': 0.0
        }
        
        # Track embedding statistics
        all_embeddings = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(pbar):
            x_context = batch['x_context']
            x_target = batch['x_target']
            
            # Apply random augmentations
            if augment_rounds > 0:
                context_seq, augmentations = rnd_augment(rounds=augment_rounds)
                x_context = sequence_augment(x_context, context_seq, augmentations)
                x_target = sequence_augment(x_target, context_seq, augmentations)
                if isinstance(x_context, np.ndarray):
                    x_context = torch.from_numpy(x_context)
                if isinstance(x_target, np.ndarray):
                    x_target = torch.from_numpy(x_target)
            x_context = x_context.to(self.device, non_blocking=True).float()
            x_target = x_target.to(self.device, non_blocking=True).float()
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    losses = self.model.compute_loss(
                        x_context, x_target, **loss_weights
                    )
                
                self.scaler.scale(losses['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.model.compute_loss(
                    x_context, x_target, **loss_weights
                )
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.model.update_target_encoder()

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Collect embeddings for statistics (every 10 batches)
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    outputs = self.model(x_context, x_target)
                    all_embeddings.append(outputs['z_t'].cpu())
            
            # Update progress bar
            pbar.set_postfix({
                'total': f"{losses['total'].item():.4f}",
                'jepa': f"{losses['jepa'].item():.4f}",
                'spectral': f"{losses['spectral'].item():.4f}",
                'variance': f"{losses['variance'].item():.4f}"
            })
        
        # Average losses
        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        # Compute embedding statistics
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            embedding_stats = {
                'mean': all_embeddings.mean().item(),
                'std': all_embeddings.std().item(),
                'min': all_embeddings.min().item(),
                'max': all_embeddings.max().item()
            }
        else:
            embedding_stats = {}
        
        return epoch_losses, embedding_stats
    
    @torch.no_grad()
    def validate(
        self,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        if loss_weights is None:
            loss_weights = {
                'alpha_jepa': 1.0,
                'alpha_spectral': 0.1,
                'alpha_variance': 0.1   # NEW
            }
        
        epoch_losses = {
            'total': 0.0,
            'jepa': 0.0,
            'spectral': 0.0,
            'variance': 0.0
        }
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            x_context = batch['x_context'].to(self.device, non_blocking=True)
            x_target = batch['x_target'].to(self.device, non_blocking=True)
            
            losses = self.model.compute_loss(
                x_context, x_target, **loss_weights
            )
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
        
        n_batches = len(self.val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def check_for_collapse(self) -> Dict[str, float]:
        """
        CRITICAL: Check for representation collapse.
        
        Warning signs:
        - Variance < 0.1 (collapsed to point)
        - Loss < 0.01 (too perfect, suspicious)
        - Val/train ratio > 10 (overfitting to artifacts)
        """
        self.model.eval()
        
        all_embeddings = []
        
        # Sample from validation set
        for i, batch in enumerate(self.val_loader):
            if i >= 5:  # Check first 5 batches
                break
            
            x_context = batch['x_context'].to(self.device)
            x_target = batch['x_target'].to(self.device)
            
            outputs = self.model(x_context, x_target)
            all_embeddings.append(outputs['z_t'].cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Compute statistics
        std_per_dim = all_embeddings.std(dim=0)
        mean_std = std_per_dim.mean().item()
        min_std = std_per_dim.min().item()
        max_std = std_per_dim.max().item()
        
        # Covariance analysis
        cov = torch.cov(all_embeddings.T)
        diag = torch.eye(cov.size(0), dtype=bool)
        off_diag = cov[~diag]
        mean_off_diag_cov = off_diag.abs().mean().item()
        
        stats = {
            'mean_std': mean_std,
            'min_std': min_std,
            'max_std': max_std,
            'mean_off_diag_cov': mean_off_diag_cov
        }
        
        # WARNINGS
        if mean_std < 0.1:
            logger.warning(f"⚠ COLLAPSE DETECTED: Variance = {mean_std:.6f} (should be > 0.3)")
        
        if min_std < 0.01:
            logger.warning(f"⚠ Dimension collapsed: min_std = {min_std:.6f}")
        
        return stats
    
    def fit(
        self,
        num_epochs: int,
        loss_weights: Optional[Dict[str, float]] = None,
        save_dir: Optional[Path] = None,
        validate_every: int = 1,
        check_collapse_every: int = 5,
        augment_rounds: int = 2,
    ):
        """
        Train with comprehensive monitoring.
        """
        save_dir = save_dir if save_dir else self.save_dir
        
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*80}")
            
            # Train
            train_losses, embedding_stats = self.train_epoch(loss_weights, augment_rounds)
            self.history['train_loss'].append(train_losses)
            
            logger.info(
                f"Train - Total: {train_losses['total']:.6f}, "
                f"JEPA: {train_losses['jepa']:.6f}, "
                f"Spectral: {train_losses['spectral']:.6f}"
            )
            
            if embedding_stats:
                logger.info(
                    f"Embeddings - Mean: {embedding_stats['mean']:.6f}, "
                    f"Std: {embedding_stats['std']:.6f}"
                )
                self.history['embedding_stats'].append(embedding_stats)
            
            # Validate
            if (epoch + 1) % validate_every == 0:
                val_losses = self.validate(loss_weights)
                self.history['val_loss'].append(val_losses)
                
                logger.info(
                    f"Val   - Total: {val_losses['total']:.6f}, "
                    f"JEPA: {val_losses['jepa']:.6f}, "
                )
                
                # Check val/train ratio
                if train_losses['total'] > 0:
                    ratio = val_losses['total'] / train_losses['total']
                    logger.info(f"Val/Train ratio: {ratio:.2f}")
                    
                    if ratio > 10:
                        logger.warning(
                            f"⚠ Val/Train ratio too high ({ratio:.2f})! "
                            "Possible overfitting to data artifacts."
                        )
                
                # Predictor analysis
                props = self.model.analyze_predictor_properties()
                self.history['predictor_properties'].append(props)
                
                if props:
                    logger.info(
                        f"Predictor - Identity dev: {props['identity_deviation']:.6f}, "
                        f"Frobenius norm: {props['frobenius_norm']:.6f}"
                    )
                    
                    if 'eigenvalues_near_one' in props:
                        logger.info(f"           Eigenvalues near 1: {props['eigenvalues_near_one']}")
                
                # Check for collapse
                if (epoch + 1) % check_collapse_every == 0:
                    collapse_stats = self.check_for_collapse()
                    logger.info(
                        f"Representation - Mean std: {collapse_stats['mean_std']:.6f}, "
                        f"Min std: {collapse_stats['min_std']:.6f}"
                    )
                
                # Save best model
                if save_dir and val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_loss': train_losses,
                        'val_loss': val_losses,
                        'embedding_stats': embedding_stats,
                        'history': self.history
                    }
                    
                    torch.save(checkpoint, save_dir / 'best_model.pt')
                    logger.info(f"✓ Saved best model (val_loss={best_val_loss:.6f})")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save periodic checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }
                torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        logger.info("\n" + "="*80)
        logger.info(f"Training complete! Best val loss: {best_val_loss:.6f}")
        logger.info("="*80)
        
        # Save final history
        if save_dir:
            with open(save_dir / 'training_history.json', 'w') as f:
                # Convert to JSON-serializable format
                history_json = {
                    'train_loss': self.history['train_loss'],
                    'val_loss': self.history['val_loss'],
                    'embedding_stats': self.history['embedding_stats']
                }
                json.dump(history_json, f, indent=2)

    def load(self, name: str = 'best_model', folder_path: Optional[Path] = None):
        folder_path = folder_path if folder_path else self.save_dir
        path = folder_path / f'{name}.pt'
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.embeddings = checkpoint.get('embeddings', None)

    def save(self, name: str = 'best_model', folder_path: Optional[Path] = None, **kwargs):
        folder_path = folder_path if folder_path else self.save_dir
        if folder_path is not None:
            folder_path = Path(folder_path)
            folder_path.mkdir(exist_ok=True, parents=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'embeddings': self.embeddings
        }, folder_path / f'{name}.pt')

def create_datasets_from_price_client(
    price_client,
    token: str,
    window: int = 768,    
    stride: int = 50,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    days: int = 30,
    page_size: int = 5000,
    augment_train: bool = True
):
    """
    CORRECTED: Create datasets with proper global normalization.
    
    Critical: Fit normalizer on training set ONLY.
    """
    logger.info(f"Loading historical prices for {token}...")
    
    # Load all available data
    historical_prices = price_client.load_historical_prices(
        token,
        days=days,
        page_size=page_size
    )
    
    if historical_prices is None or len(historical_prices) == 0:
        raise ValueError(f"No historical data available for {token}")
    
    # Extract prices and timestamps
    prices = historical_prices[:, 0]  # First column is price
    timestamps = historical_prices[:, 1]  # Second column is timestamp
    
    logger.info(f"Loaded {len(prices)} price points")
    logger.info(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Split data
    n = len(prices)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_prices = prices[:train_end]
    train_timestamps = timestamps[:train_end]
    
    val_prices = prices[train_end:val_end]
    val_timestamps = timestamps[train_end:val_end]
    
    test_prices = prices[val_end:]
    test_timestamps = timestamps[val_end:]
    
    logger.info(f"Split: train={len(train_prices)}, val={len(val_prices)}, test={len(test_prices)}")
    
    # CRITICAL: Fit normalizer on TRAINING SET ONLY
    logger.info("Fitting normalizer on training set...")
    normalizer = GlobalPriceNormalizer()
    normalizer.fit(train_prices)
    
    logger.info(f"Normalizer: mean={normalizer.price_mean:.2f}, std={normalizer.price_std:.2f}")
    
    # Create datasets (all use same normalizer)
    train_dataset = CryptoPriceDataset(
        train_prices, train_timestamps, normalizer,
        window=window,
        stride=stride,
        mode='train',
        augment=augment_train
    )
    
    val_dataset = CryptoPriceDataset(
        val_prices, val_timestamps, normalizer,
        window=window, stride=stride,
        mode='val',
        augment=False
    )
    
    test_dataset = CryptoPriceDataset(
        test_prices, test_timestamps, normalizer,
        window=window, stride=stride,
        mode='test',
        augment=False
    )
    
    return train_dataset, val_dataset, test_dataset, normalizer


if __name__ == "__main__":
    """
    Example usage
    """
    print("="*80)
    print("Corrected Koopman-JEPA Training")
    print("="*80)
    
    # This would use your actual price client
    # For demo, we'll use synthetic data
    
    # Generate synthetic multi-regime data
    print("\nGenerating synthetic data...")
    n_samples = 100000
    
    # Create price series with regime changes
    prices = []
    for regime in range(10):  # 10 regimes
        regime_length = n_samples // 10
        
        if regime % 3 == 0:
            # Bull market
            trend = np.cumsum(np.random.randn(regime_length) * 100 + 50)
        elif regime % 3 == 1:
            # Bear market
            trend = np.cumsum(np.random.randn(regime_length) * 100 - 50)
        else:
            # Sideways
            trend = np.cumsum(np.random.randn(regime_length) * 50)
        
        prices.append(trend + 50000 + regime * 1000)
    
    prices = np.concatenate(prices).astype(np.float32)
    timestamps = np.arange(len(prices), dtype=np.float32)
    
    print(f"Generated {len(prices)} prices across 10 regimes")
    
    # Split data
    n = len(prices)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)
    
    train_prices = prices[:train_end]
    val_prices = prices[train_end:val_end]
    test_prices = prices[val_end:]
    
    # CRITICAL: Fit normalizer on training only
    print("\nFitting normalizer on training set...")
    normalizer = GlobalPriceNormalizer()
    normalizer.fit(train_prices)
    print(f"Normalizer: mean={normalizer.price_mean:.2f}, std={normalizer.price_std:.2f}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CryptoPriceDataset(
        train_prices, timestamps[:train_end], normalizer,
        mode='train'
    )
    
    val_dataset = CryptoPriceDataset(
        val_prices, timestamps[train_end:val_end], normalizer,
        mode='val'
    )
    
    # Create model
    print("\nCreating model...")
    model = KoopmanJEPA(
        input_channels=3,  # price + returns + log_returns
        sequence_length=768,
        latent_dim=32,
        predictor_type='linear',
        init_identity=True
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = JEPATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=64,
        learning_rate=1e-4,
        device=device
    )
    
    # Train
    print("\nStarting training...")
    save_dir = Path('./jepa_corrected_checkpoints')
    
    trainer.fit(
        num_epochs=15,
        loss_weights={
            'alpha_jepa': 1.0,
            'alpha_regime': 0.01,  # Small!
            'alpha_spectral': 0.1
        },
        save_dir=save_dir,
        validate_every=1,
        check_collapse_every=5
    )
    
    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {save_dir}")