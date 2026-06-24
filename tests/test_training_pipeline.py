import os
import pytest
import pandas as pd
import numpy as np
import torch
from cryptotrading.predict.utils.tools import dotdict

from cryptotrading.predict.data import data_provider
from cryptotrading.predict.models import get_model
from cryptotrading.predict.exp.forecast import ForecastExp


@pytest.fixture
def dummy_csv_path(tmp_path):
    """Generates a temporary CSV file with dummy price data for training tests.
    
    Generates 250 periods of data to ensure that training, validation, and test
    dataset splits all have positive lengths for sliding windows.
    """
    csv_path = tmp_path / "dummy_prices.csv"
    dates = pd.date_range(start='2025-01-01', periods=250, freq='h')
    prices = np.sin(np.linspace(0, 15, 250)) * 10 + 100 + np.random.normal(0, 0.5, 250)
    df = pd.DataFrame({
        'date': dates,
        'OT': prices  # OT is the standard target feature
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def basic_configs(dummy_csv_path, tmp_path):
    """Returns a basic configuration dictionary for forecasting experiments."""
    return dotdict({
        'task_name': 'short_term_forecast',
        'is_training': 1,
        'model_id': 'test_model',
        'model': 'Linear',
        'data': 'custom',
        'data_path': dummy_csv_path,
        'features': 'S',
        'target': 'OT',
        'scale': True,
        'timeenc': 0,
        'freq': 'h',
        'seq_len': 24,
        'label_len': 12,
        'pred_len': 12,
        'enc_in': 1,
        'dec_in': 1,
        'c_out': 1,
        'd_model': 16,
        'dropout': 0.0,
        'use_gpu': False,
        'gpu': 0,
        'use_multi_gpu': False,
        'train_epochs': 1,
        'batch_size': 4,
        'patience': 3,
        'learning_rate': 0.01,
        'checkpoints': str(tmp_path / "checkpoints"),
        'use_amp': False,
        'output_attention': False,
        'ps_loss_mode': 'none',
        'ib_loss_enabled': False,
        'lradj': 'type1',
        'use_tqdm': False,
        'des': 'test',
        'n_period': 2,
        'topm': 2,
        'embed': 'timeF',
        'distil': False
    })


def test_index_aware_dataset(basic_configs):
    """Verifies that the data provider yields 5-tuples containing the absolute index."""
    dataset, loader = data_provider(basic_configs, flag='train')
    assert len(dataset) > 0
    
    # Fetch first sample
    sample = dataset[0]
    assert len(sample) == 5
    
    seq_x, seq_y, seq_x_mark, seq_y_mark, index = sample
    assert isinstance(seq_x, np.ndarray)
    assert isinstance(seq_y, np.ndarray)
    assert isinstance(index, int)
    assert index == 0
    
    # Test loader batch unpacking
    batch = next(iter(loader))
    assert len(batch) == 5
    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index = batch
    assert batch_x.shape[0] == basic_configs.batch_size
    assert batch_index.shape[0] == basic_configs.batch_size
    
    # Shuffled batch indices should be unique and within valid dataset boundaries
    assert len(torch.unique(batch_index)) == basic_configs.batch_size
    assert torch.all(batch_index >= 0)
    assert torch.all(batch_index < len(dataset))


def test_standard_linear_training_step(basic_configs):
    """Verifies that the training runner can build and run a standard Linear model."""
    exp = ForecastExp(basic_configs)
    
    # Assert model was built successfully
    assert exp.model is not None
    assert isinstance(exp.model, torch.nn.Module)
    
    # Run 1 epoch of training
    setting = "test_linear_run"
    metrics = exp.train(setting)
    
    assert "vali_loss" in metrics
    assert "test_loss" in metrics
    assert os.path.exists(os.path.join(basic_configs.checkpoints, setting, "checkpoint.pth"))


def test_raft_retrieval_augmented_training_step(basic_configs):
    """Verifies that the training runner can build and run the RAFT model.

    This tests dataset pre-computation, index-based forward signatures,
    and loss backpropagation.
    """
    # Switch model to RAFT
    basic_configs.model = 'RAFT'
    exp = ForecastExp(basic_configs)
    
    assert exp.model is not None
    assert exp.model.__class__.__name__ == 'RAFT'
    
    # Run 1 epoch of training
    setting = "test_raft_run"
    metrics = exp.train(setting)
    
    assert "vali_loss" in metrics
    assert "test_loss" in metrics
    assert os.path.exists(os.path.join(basic_configs.checkpoints, setting, "checkpoint.pth"))
