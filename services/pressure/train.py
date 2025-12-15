"""
Production-Ready Pressure Model Training

FIXES:
2. Label smoothing properly implemented
3. Removed incorrect consistency constraint
4. Feature standardization integrated
5. Uncertainty calibration added
6. Temporal split enforced
8. Optimized hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training"""
    model_type: str = "robust"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 128  # FIX #8: Reduced for better uncertainty
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # Slightly higher for regularization
    num_epochs: int = 100
    patience: int = 15

    # FIX #6: No random split - must use temporal
    use_temporal_split: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_gap: int = 10  # Gap between splits

    gradient_clip: float = 1.0

    # Loss weights (FIX #3: no consistency weight)
    buy_pressure_weight: float = 1.0
    sell_pressure_weight: float = 1.0
    total_pressure_weight: float = 2.0
    coherence_weight: float = 0.3  # Reduced - softer constraint
    direction_weight: float = 1.5

    # FIX #2: Label smoothing properly implemented
    use_label_smoothing: bool = True
    label_smoothing: float = 0.05

    # FIX #5: Uncertainty calibration
    use_mc_dropout: bool = True
    mc_samples: int = 20  # Increased for better uncertainty
    dropout_rate: float = 0.15  # Increased
    calibrate_uncertainty: bool = True

    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True

    def to_dict(self):
        return asdict(self)


class MemoryEfficientDataset(Dataset):
    """
    Does NOT load everything to GPU RAM in __init__.
    Uses numpy array references.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # Keep as numpy to save RAM, convert to tensor on-the-fly
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.features[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
        }


class PressureDataset(Dataset):
    """
    PyTorch dataset for pressure prediction.

    FIX #4: Now handles normalized features
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict],
        apply_label_smoothing: bool = False,
        smoothing: float = 0.05,
    ):
        """
        Args:
            features: (N, feature_dim) array - should already be normalized
            labels: (N, 3) array with [buy_pressure, sell_pressure, total_pressure]
            metadata: List of metadata dicts
            apply_label_smoothing: Whether to apply label smoothing
            smoothing: Amount of smoothing
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

        # FIX #2: Apply label smoothing
        if apply_label_smoothing:
            self.labels = self._smooth_labels(self.labels, smoothing)

        self.metadata = metadata

    def _smooth_labels(self, labels: torch.Tensor, smoothing: float) -> torch.Tensor:
        """
        Apply label smoothing to reduce overconfidence.

        For pressures in [0, 1]: p_smooth = (1 - α) * p + α * 0.5
        For total pressure in [-1, 1]: no smoothing (it's already continuous)
        """
        smoothed = labels.clone()

        # Smooth buy pressure (column 0)
        smoothed[:, 0] = (1 - smoothing) * labels[:, 0] + smoothing * 0.5

        # Smooth sell pressure (column 1)
        smoothed[:, 1] = (1 - smoothing) * labels[:, 1] + smoothing * 0.5

        # Total pressure (column 2) - no smoothing needed
        # It's already a continuous signed value

        return smoothed

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "metadata": self.metadata[idx],
        }


class ImprovedPressureLoss(nn.Module):
    """
    Improved loss function for pressure prediction.

    FIXES:
    - Issue #3: Removed incorrect buy + sell = 1 constraint
    - Now allows independent pressure magnitudes
    """

    def __init__(
        self,
        buy_weight: float = 1.0,
        sell_weight: float = 1.0,
        total_weight: float = 2.0,
        coherence_weight: float = 0.3,
        direction_weight: float = 1.5,
        use_huber: bool = True,
    ):
        super().__init__()
        self.buy_weight = buy_weight
        self.sell_weight = sell_weight
        self.total_weight = total_weight
        self.coherence_weight = coherence_weight
        self.direction_weight = direction_weight

        # Use Huber loss (more robust to outliers than MSE)
        if use_huber:
            self.regression_loss = nn.SmoothL1Loss()
        else:
            self.regression_loss = nn.MSELoss()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor):
        """
        Args:
            predictions: Dict with 'buy_pressure', 'sell_pressure', 'total_pressure'
            targets: (batch, 3) tensor with [buy, sell, total]
        """
        buy_pred = predictions["buy_pressure"].squeeze()
        sell_pred = predictions["sell_pressure"].squeeze()
        total_pred = predictions["total_pressure"].squeeze()

        buy_target = targets[:, 0]
        sell_target = targets[:, 1]
        total_target = targets[:, 2]

        # Regression losses for each component
        buy_loss = self.regression_loss(buy_pred, buy_target)
        sell_loss = self.regression_loss(sell_pred, sell_target)
        total_loss = self.regression_loss(total_pred, total_target)

        # FIX #3: REMOVED incorrect consistency constraint
        # Old (WRONG): buy + sell = 1
        # New (CORRECT): buy and sell are independent
        # Volatile markets can have high buy AND high sell
        # Stagnant markets can have low buy AND low sell

        # Coherence: total should roughly equal buy - sell
        # But make this a soft constraint, not hard
        implied_total = buy_pred - sell_pred
        coherence_loss = self.regression_loss(total_pred, implied_total.detach())

        # Direction accuracy: penalize sign mismatch
        direction_pred = torch.sign(total_pred)
        direction_target = torch.sign(total_target)
        direction_loss = (direction_pred != direction_target).float().mean()

        # Combine losses
        total = (
            self.buy_weight * buy_loss
            + self.sell_weight * sell_loss
            + self.total_weight * total_loss
            + self.coherence_weight * coherence_loss
            + self.direction_weight * direction_loss
        )

        return total, {
            "buy_loss": buy_loss.item(),
            "sell_loss": sell_loss.item(),
            "total_loss": total_loss.item(),
            "coherence_loss": coherence_loss.item(),
            "direction_loss": direction_loss.item(),
        }


# Loss function adjustment for Aleatoric Uncertainty
def gaussian_nll_loss(pred, target, log_var):
    precision = torch.exp(-log_var)
    return 0.5 * precision * (pred - target) ** 2 + 0.5 * log_var

class UncertaintyCalibrator:
    """
    Calibrate MC Dropout uncertainty estimates.

    FIX #5: Proper uncertainty calibration

    Raw MC Dropout std can be poorly calibrated.
    This uses temperature scaling on a validation set.
    """

    def __init__(self):
        self.temperature = 1.0

    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str,
        mc_samples: int = 20,
    ):
        """
        Find optimal temperature for uncertainty calibration.

        Uses Expected Calibration Error (ECE) on validation set.
        """
        logger.info("Calibrating uncertainty estimates...")

        # Collect predictions and uncertainties
        all_predictions = []
        all_uncertainties = []
        all_targets = []

        model.train()  # Enable dropout

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Collecting uncertainties"):
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)

                # MC Dropout samples
                samples = []
                for _ in range(mc_samples):
                    pred = model(features)
                    samples.append(pred["total_pressure"].cpu().numpy())

                samples = np.array(samples)
                mean = samples.mean(axis=0)
                std = samples.std(axis=0)

                all_predictions.append(mean)
                all_uncertainties.append(std)
                all_targets.append(labels[:, 2].cpu().numpy())  # total pressure

        all_predictions = np.concatenate(all_predictions)
        all_uncertainties = np.concatenate(all_uncertainties)
        all_targets = np.concatenate(all_targets)

        # Grid search for best temperature
        best_temp = 1.0
        best_ece = float("inf")

        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            scaled_std = all_uncertainties * temp
            ece = self._compute_ece(all_predictions, scaled_std, all_targets)

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self.temperature = best_temp
        logger.info(f"Optimal temperature: {best_temp:.2f}, ECE: {best_ece:.4f}")

    def _compute_ece(self, predictions, uncertainties, targets, n_bins=10):
        """
        Compute Expected Calibration Error.

        Measures if predicted uncertainties match actual errors.
        """
        errors = np.abs(predictions - targets)

        # Bin by uncertainty
        bin_boundaries = np.linspace(
            uncertainties.min(), uncertainties.max(), n_bins + 1
        )

        ece = 0.0
        for i in range(n_bins):
            mask = (uncertainties >= bin_boundaries[i]) & (
                uncertainties < bin_boundaries[i + 1]
            )

            if mask.sum() > 0:
                avg_confidence = 1 - uncertainties[mask].mean()
                avg_accuracy = 1 - errors[mask].mean()
                bin_weight = mask.sum() / len(uncertainties)

                ece += bin_weight * abs(avg_confidence - avg_accuracy)

        return ece

    def scale(self, uncertainty: float) -> float:
        """Apply temperature scaling"""
        return uncertainty * self.temperature


class PressureTrainer:
    """
    Trainer with all critical fixes.

    FIXES:
    - #4: Feature standardization
    - #5: Uncertainty calibration
    - #6: Temporal split enforced
    """

    def __init__(
        self,
        config: TrainingConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        from .model import get_model
        self.model  = get_model(config).to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss function (improved)
        self.criterion = ImprovedPressureLoss(
            buy_weight=config.buy_pressure_weight,
            sell_weight=config.sell_pressure_weight,
            total_weight=config.total_pressure_weight,
            coherence_weight=config.coherence_weight,
            direction_weight=config.direction_weight,
        )

        # Learning rate scheduler
        if config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                min_lr=config.min_lr,
                verbose=True,
            )
        else:
            self.scheduler = None

        # FIX #5: Uncertainty calibrator
        if config.calibrate_uncertainty:
            self.uncertainty_calibrator = UncertaintyCalibrator()
        else:
            self.uncertainty_calibrator = None

        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_metrics = {
            "buy_loss": 0,
            "sell_loss": 0,
            "total_loss": 0,
            "coherence_loss": 0,
            "direction_loss": 0,
        }

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss, metrics = self.criterion(predictions, labels)

            loss.backward()

            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            for key, value in metrics.items():
                total_metrics[key] += value

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        avg_metrics = {k: v / len(train_loader) for k, v in total_metrics.items()}

        return avg_loss, avg_metrics

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_metrics = {
            "buy_loss": 0,
            "sell_loss": 0,
            "total_loss": 0,
            "coherence_loss": 0,
            "direction_loss": 0,
            "direction_accuracy": 0,
        }

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)

                predictions = self.model(features)
                loss, metrics = self.criterion(predictions, labels)

                total_loss += loss.item()
                for key, value in metrics.items():
                    total_metrics[key] += value

                all_predictions.append(
                    {k: v.cpu().numpy() for k, v in predictions.items()}
                )
                all_targets.append(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        avg_metrics = {k: v / len(val_loader) for k, v in total_metrics.items()}

        # Calculate additional metrics
        all_predictions = {
            k: np.concatenate([p[k] for p in all_predictions])
            for k in all_predictions[0].keys()
        }
        all_targets = np.concatenate(all_targets)

        # Direction accuracy
        direction_pred = np.sign(all_predictions["total_pressure"].flatten())
        direction_target = np.sign(all_targets[:, 2])
        direction_accuracy = (direction_pred == direction_target).mean()
        avg_metrics["direction_accuracy"] = direction_accuracy

        # Magnitude correlation
        magnitude_correlation = np.corrcoef(
            np.abs(all_predictions["total_pressure"].flatten()),
            np.abs(all_targets[:, 2]),
        )[0, 1]
        avg_metrics["magnitude_correlation"] = magnitude_correlation

        return avg_loss, avg_metrics

    def predict_with_uncertainty(
        self, features: torch.Tensor, n_samples: int = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with calibrated uncertainty"""
        if n_samples is None:
            n_samples = self.config.mc_samples

        self.model.train()  # Enable dropout
        predictions = {"buy_pressure": [], "sell_pressure": [], "total_pressure": []}

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(features)
                for key in predictions.keys():
                    predictions[key].append(pred[key].cpu().numpy())

        # Compute mean and std
        results = {}
        for key, values in predictions.items():
            values = np.array(values)
            mean = values.mean(axis=0)
            std = values.std(axis=0)

            # Apply calibration
            if self.uncertainty_calibrator is not None:
                std = self.uncertainty_calibrator.scale(std)

            results[key] = (mean, std)

        return results

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config.to_dict(),
            "history": self.history,
        }

        if self.uncertainty_calibrator is not None:
            checkpoint["uncertainty_temperature"] = (
                self.uncertainty_calibrator.temperature
            )

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss={val_loss:.6f}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Main training loop"""
        logger.info(f"Starting training on {self.device}")

        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            # Log
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            logger.info(
                f"Val Direction Accuracy: {val_metrics['direction_accuracy']:.4f}"
            )

            # LR scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch + 1, val_loss, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # FIX #5: Calibrate uncertainty on validation set
        if (
            self.config.calibrate_uncertainty
            and self.uncertainty_calibrator is not None
        ):
            self.uncertainty_calibrator.calibrate(
                self.model, val_loader, self.device, self.config.mc_samples
            )

        logger.info(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")
        return self.history


def prepare_temporal_dataloaders(
    dataset_dict: Dict[str, np.ndarray], config: TrainingConfig, featurizer=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare dataloaders with TEMPORAL split (FIX #6).

    Critical: Must not randomly shuffle time-series data!
    """
    features = dataset_dict["features"]
    labels = dataset_dict["labels"]
    metadata = dataset_dict["metadata"]

    # FIX #4: Normalize features
    if featurizer is not None and hasattr(featurizer, "fit_normalizer"):
        # Fit on training data only (avoid leakage)
        train_size = int(len(features) * config.train_ratio)
        featurizer.fit_normalizer(features[:train_size])

        # Normalize all features
        features = featurizer.normalize(features)
        logger.info("Applied feature normalization")

    # FIX #6: Temporal split (not random!)
    from .oracle import TemporalDatasetSplitter

    splitter = TemporalDatasetSplitter()
    train_idx, val_idx, test_idx = splitter.split_temporal(
        len(features),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        gap=config.split_gap,
    )

    # Create datasets with label smoothing for training only
    train_dataset = PressureDataset(
        features[train_idx],
        labels[train_idx],
        [metadata[i] for i in train_idx],
        apply_label_smoothing=config.use_label_smoothing,
        smoothing=config.label_smoothing,
    )

    val_dataset = PressureDataset(
        features[val_idx],
        labels[val_idx],
        [metadata[i] for i in val_idx],
        apply_label_smoothing=False,  # No smoothing for validation
    )

    test_dataset = PressureDataset(
        features[test_idx],
        labels[test_idx],
        [metadata[i] for i in test_idx],
        apply_label_smoothing=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # Can shuffle within training set
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(
        f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
