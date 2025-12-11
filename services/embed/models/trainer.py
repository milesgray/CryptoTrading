"""
Training script for the contrastive trade setup encoder.

Uses supervised contrastive learning to train embeddings where:
- Profitable longs cluster together
- Profitable shorts cluster together  
- Losses are separated from wins
- Different outcome magnitudes are distinguished
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Optional, Dict, List
from tqdm import tqdm
import json

from .encoder import (
    PriceWindowEncoder,
    SupervisedContrastiveLoss,
    TripletLoss,
    TradeSetupDataset,
    TradeSetup,
    extract_trade_setups
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncoderTrainer:
    """Trainer for the trade setup encoder"""
    
    def __init__(
        self,
        window_size: int = 100,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.07,
        use_supcon: bool = True,  # Use SupCon vs Triplet
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.device = device
        self.use_supcon = use_supcon
        
        # Initialize model
        self.encoder = PriceWindowEncoder(
            window_size=window_size - 1,  # -1 because we use returns
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Loss function
        if use_supcon:
            self.criterion = SupervisedContrastiveLoss(temperature=temperature)
        else:
            self.criterion = TripletLoss(margin=0.3)
        
        # Optimizer with warmup
        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = None  # Set during training
        
        self.train_losses = []
        self.val_losses = []
    
    def train(
        self,
        train_setups: List[TradeSetup],
        val_setups: Optional[List[TradeSetup]] = None,
        num_epochs: int = 100,
        save_path: Optional[Path] = None,
        patience: int = 10
    ) -> Dict:
        """
        Train the encoder on trade setups.
        
        Args:
            train_setups: Training data
            val_setups: Validation data (optional)
            num_epochs: Number of training epochs
            save_path: Where to save best model
            patience: Early stopping patience
            
        Returns:
            Training history dict
        """
        # Create datasets
        train_dataset = TradeSetupDataset(train_setups, augment=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        val_loader = None
        if val_setups:
            val_dataset = TradeSetupDataset(val_setups, augment=False)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Learning rate scheduler with warmup
        num_training_steps = num_epochs * len(train_loader)
        warmup_steps = min(1000, num_training_steps // 10)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.defaults['lr'],
            total_steps=num_training_steps,
            pct_start=warmup_steps / num_training_steps
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Training on {len(train_setups)} setups for {num_epochs} epochs")
        logger.info(f"Batch size: {self.batch_size}, Device: {self.device}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        self.save(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            elif save_path and (epoch + 1) % 10 == 0:
                self.save(save_path)
            
            # Logging
            log_msg = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" - Val Loss: {val_loss:.4f}"
            logger.info(log_msg)
        
        # Save final model
        if save_path:
            self.save(save_path)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch"""
        self.encoder.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for batch_x, batch_labels in pbar:
            batch_x = batch_x.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            embeddings = self.encoder(batch_x)
            
            # Compute loss
            if self.use_supcon:
                loss = self.criterion(embeddings, batch_labels)
            else:
                # For triplet loss, need to reorganize batch
                loss = self._compute_triplet_loss(embeddings, batch_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def _validate(self, loader: DataLoader) -> float:
        """Validate the model"""
        self.encoder.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_labels in loader:
                batch_x = batch_x.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                embeddings = self.encoder(batch_x)
                
                if self.use_supcon:
                    loss = self.criterion(embeddings, batch_labels)
                else:
                    loss = self._compute_triplet_loss(embeddings, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _compute_triplet_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss with hard mining"""
        batch_size = embeddings.shape[0]
        
        # Compute all pairwise distances
        distances = 1 - torch.mm(embeddings, embeddings.T)
        
        # Create masks
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float()
        mask_neg = (labels != labels.T).float()
        
        # Remove self from positives
        mask_pos = mask_pos - torch.eye(batch_size, device=self.device)
        
        # Hard positive mining: furthest positive
        pos_distances = distances * mask_pos
        pos_distances[mask_pos == 0] = -1
        hardest_pos, _ = pos_distances.max(dim=1)
        
        # Hard negative mining: closest negative
        neg_distances = distances * mask_neg
        neg_distances[mask_neg == 0] = 2  # Max distance
        hardest_neg, _ = neg_distances.min(dim=1)
        
        # Triplet loss
        margin = 0.3
        loss = torch.relu(hardest_pos - hardest_neg + margin)
        
        # Only count valid triplets
        valid = (hardest_pos > -1) & (hardest_neg < 2)
        if valid.sum() > 0:
            return loss[valid].mean()
        return torch.tensor(0.0, device=self.device)
    
    def save(self, path: Path):
        """Save model and config"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.encoder.state_dict(), path / 'encoder.pt')
        
        # Save config
        config = {
            'window_size': self.window_size,
            'embedding_dim': self.embedding_dim,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from path"""
        path = Path(path)
        
        # Load config
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Recreate encoder with correct dimensions
        self.encoder = PriceWindowEncoder(
            window_size=config['window_size'] - 1,
            embedding_dim=config['embedding_dim']
        ).to(self.device)
        
        # Load weights
        self.encoder.load_state_dict(torch.load(path / 'encoder.pt', map_location=self.device))
        self.encoder.eval()
        
        logger.info(f"Model loaded from {path}")


def train_from_oracle_data(
    prices: np.ndarray,
    timestamps: np.ndarray,
    oracle_actions: np.ndarray,
    oracle_leverages: np.ndarray,
    window_size: int = 100,
    embedding_dim: int = 128,
    num_epochs: int = 100,
    save_path: Optional[str] = None,
    val_split: float = 0.1
) -> EncoderTrainer:
    """
    Convenience function to train encoder from oracle-labeled data.
    
    Args:
        prices: Price array
        timestamps: Timestamp array
        oracle_actions: Actions from DP oracle
        oracle_leverages: Leverage values
        window_size: Price history window
        embedding_dim: Output embedding dimension
        num_epochs: Training epochs
        save_path: Where to save model
        val_split: Validation set fraction
        
    Returns:
        Trained EncoderTrainer
    """
    # Extract trade setups
    logger.info("Extracting trade setups from oracle data...")
    setups = extract_trade_setups(
        prices=prices,
        timestamps=timestamps,
        oracle_actions=oracle_actions,
        oracle_leverages=oracle_leverages,
        window_size=window_size
    )
    
    logger.info(f"Found {len(setups)} trade setups")
    
    # Log distribution
    from collections import Counter
    label_dist = Counter(s.outcome_label for s in setups)
    logger.info(f"Label distribution: {dict(label_dist)}")
    
    # Split train/val
    np.random.shuffle(setups)
    split_idx = int(len(setups) * (1 - val_split))
    train_setups = setups[:split_idx]
    val_setups = setups[split_idx:] if val_split > 0 else None
    
    # Train
    trainer = EncoderTrainer(
        window_size=window_size,
        embedding_dim=embedding_dim
    )
    
    trainer.train(
        train_setups=train_setups,
        val_setups=val_setups,
        num_epochs=num_epochs,
        save_path=Path(save_path) if save_path else None
    )
    
    return trainer


# Evaluation utilities
def evaluate_embedding_quality(
    encoder: PriceWindowEncoder,
    setups: List[TradeSetup],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Evaluate embedding quality using clustering metrics.
    
    Returns:
        Dict with silhouette score, cluster purity, etc.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    encoder.eval()
    
    # Get all embeddings
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for setup in setups:
            x = torch.from_numpy(setup.price_window).float().unsqueeze(0).to(device)
            emb = encoder(x).cpu().numpy()[0]
            embeddings.append(emb)
            labels.append(setup.outcome_label)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Silhouette score (how well separated are clusters)
    sil_score = silhouette_score(embeddings, labels)
    
    # K-means clustering alignment
    n_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Cluster purity
    from scipy.stats import mode
    purity = 0
    for i in range(n_clusters):
        mask = cluster_labels == i
        if mask.sum() > 0:
            most_common = mode(labels[mask], keepdims=True).mode[0]
            purity += (labels[mask] == most_common).sum()
    purity /= len(labels)
    
    return {
        'silhouette_score': sil_score,
        'cluster_purity': purity,
        'num_samples': len(labels),
        'num_classes': n_clusters
    }