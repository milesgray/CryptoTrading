"""
Data pipeline for generating and storing trade setup embeddings.

This module orchestrates:
1. Loading historical price data
2. Running the DP oracle to generate optimal labels
3. Extracting trade setups
4. Training the contrastive encoder
5. Generating embeddings for all setups
6. Storing everything in PostgreSQL with pgvector
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple    
import logging
import json

from cryptotrading.trade.oracle import DPOracle, LeveragedDPOracle
from .models.encoder import (
    PriceWindowEncoder, 
    TradeSetup, 
    extract_trade_setups
)
from .models.trainer import EncoderTrainer
from database.numpy_store import NumpyVectorStore, StoredTradeSetup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradePipeline:
    """
    End-to-end pipeline for trade setup embedding generation and storage.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        embedding_dim: int = 128,
        max_leverage: float = 20.0,
        transaction_cost: float = 0.001,
        store_path: str = "vector_store",
        oracle_type: str = "leveraged"
    ):
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        self.store_path = store_path
        
        # Components (initialized lazily)
        self.oracle: Optional[DPOracle | LeveragedDPOracle] = None
        self.encoder: Optional[PriceWindowEncoder] = None
        self.trainer: Optional[EncoderTrainer] = None
        self.store: Optional[NumpyVectorStore] = None
        
    def initialize_oracle(self):
        """Initialize the DP oracle"""
        if self.oracle_type == "leveraged":
            self.oracle = LeveragedDPOracle(
                max_leverage=self.max_leverage,
                transaction_cost=self.transaction_cost
            )
        else:
            self.oracle = DPOracle(
                max_leverage=self.max_leverage,
                transaction_cost=self.transaction_cost
            )
        logger.info("Oracle initialized")
        
    def initialize_encoder(self, model_path: Optional[Path] = None):
        """Initialize or load the encoder"""
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.encoder = PriceWindowEncoder(
            window_size=self.window_size - 1,
            embedding_dim=self.embedding_dim
        ).to(device)
        
        if model_path and model_path.exists():
            self.encoder.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            logger.info(f"Encoder loaded from {model_path}")
        else:
            logger.info("Encoder initialized with random weights")
        
        self.encoder.eval()
        
    def initialize_store(self):
        """Initialize vector store"""
        self.store = NumpyVectorStore(
            store_path=self.store_path,
            embedding_dim=self.embedding_dim
        )
        logger.info(f"Vector store initialized at {self.store_path}")
        
    def generate_oracle_labels(
        self,
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate oracle labels for price data"""
        if self.oracle is None:
            self.initialize_oracle()
        
        actions, leverages = self.oracle.compute_oracle_actions(prices)
        
        # Log statistics
        stats = self.oracle.get_statistics(prices)
        logger.info(f"Oracle statistics: {json.dumps(stats, indent=2)}")
        
        return actions, leverages
    
    def extract_setups(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray,
        actions: np.ndarray,
        leverages: np.ndarray
    ) -> List[TradeSetup]:
        """Extract trade setups from oracle-labeled data"""
        setups = extract_trade_setups(
            prices=prices,
            timestamps=timestamps,
            oracle_actions=actions,
            oracle_leverages=leverages,
            window_size=self.window_size
        )
        
        logger.info(f"Extracted {len(setups)} trade setups")
        
        # Log outcome distribution
        from collections import Counter
        outcomes = Counter(s.outcome.name for s in setups)
        logger.info(f"Outcome distribution: {dict(outcomes)}")
        
        return setups
    
    def train_encoder(
        self,
        setups: List[TradeSetup],
        num_epochs: int = 100,
        save_path: Optional[Path] = None
    ):
        """Train the contrastive encoder"""
        self.trainer = EncoderTrainer(
            window_size=self.window_size,
            embedding_dim=self.embedding_dim
        )
        
        # Split train/val
        np.random.shuffle(setups)
        split_idx = int(len(setups) * 0.9)
        train_setups = setups[:split_idx]
        val_setups = setups[split_idx:]
        
        history = self.trainer.train(
            train_setups=train_setups,
            val_setups=val_setups,
            num_epochs=num_epochs,
            save_path=save_path
        )
        
        self.encoder = self.trainer.encoder
        
        return history
    
    def generate_embeddings(
        self,
        setups: List[TradeSetup]
    ) -> np.ndarray:
        """Generate embeddings for all setups"""
        import torch
        
        if self.encoder is None:
            raise ValueError("Encoder not initialized")
        
        self.encoder.eval()
        device = next(self.encoder.parameters()).device
        
        embeddings = []
        batch_size = 256
        
        for i in range(0, len(setups), batch_size):
            batch = setups[i:i + batch_size]
            windows = np.stack([s.price_window for s in batch])
            
            with torch.no_grad():
                x = torch.from_numpy(windows).float().to(device)
                emb = self.encoder(x).cpu().numpy()
            
            embeddings.append(emb)
        
        embeddings = np.vstack(embeddings)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        return embeddings
    
    def store_setups(
        self,
        setups: List[TradeSetup],
        embeddings: np.ndarray,
        prices: np.ndarray,
        timestamps: np.ndarray,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1s'
    ) -> List[int]:
        """Store setups and embeddings in vector store"""
        if self.store is None:
            self.initialize_store()
        
        stored_setups = []
        price_windows = []
        
        for setup in setups:
            # Get raw price window for visualization
            start_idx = max(0, setup.entry_idx - self.window_size)
            end_idx = setup.entry_idx
            raw_prices = prices[start_idx:end_idx]
            
            # Get entry and exit prices
            entry_price = float(prices[setup.entry_idx])
            exit_idx = min(setup.entry_idx + setup.hold_duration, len(prices) - 1)
            exit_price = float(prices[exit_idx])
            
            stored = StoredTradeSetup(
                id=0,  # Will be assigned by store
                direction=setup.direction,
                profit_pct=setup.profit_pct,
                leverage=setup.leverage,
                hold_duration=setup.hold_duration,
                entry_timestamp=float(timestamps[setup.entry_idx]) if timestamps is not None else float(setup.entry_idx),
                entry_price=entry_price,
                exit_price=exit_price,
                symbol=symbol,
                timeframe=timeframe,
                window_size=self.window_size
            )
            stored_setups.append(stored)
            price_windows.append(raw_prices)
        
        # Pad price windows to same length
        max_len = max(len(pw) for pw in price_windows) if price_windows else 0
        if max_len > 0:
            price_windows_padded = np.array([
                np.pad(pw, (max_len - len(pw), 0), mode='edge') 
                for pw in price_windows
            ])
        else:
            price_windows_padded = None
        
        ids = self.store.add_batch(embeddings, stored_setups, price_windows_padded)
        
        # Save to disk
        self.store.save()
        
        logger.info(f"Stored {len(ids)} setups in vector store")
        return ids
    
    def run_full_pipeline(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1s',
        train_epochs: int = 100,
        model_save_path: Optional[Path] = None
    ) -> dict:
        """
        Run the complete pipeline:
        1. Generate oracle labels
        2. Extract trade setups
        3. Train encoder
        4. Generate embeddings
        5. Store in vector store
        """
        logger.info("Starting full pipeline...")
        logger.info(f"Input: {len(prices)} prices, symbol={symbol}, timeframe={timeframe}")
        
        # Step 1: Oracle labels
        logger.info("Step 1: Generating oracle labels...")
        actions, leverages = self.generate_oracle_labels(prices, timestamps)
        
        # Step 2: Extract setups
        logger.info("Step 2: Extracting trade setups...")
        setups = self.extract_setups(prices, timestamps, actions, leverages)
        
        if len(setups) < 10:
            logger.warning(f"Only {len(setups)} setups found - may not be enough for training")
        
        # Step 3: Train encoder
        logger.info("Step 3: Training encoder...")
        if len(setups) >= 10:
            history = self.train_encoder(
                setups=setups,
                num_epochs=train_epochs,
                save_path=model_save_path
            )
        else:
            logger.warning("Skipping training due to insufficient data")
            self.initialize_encoder(model_save_path)
            history = None
        
        # Step 4: Generate embeddings
        logger.info("Step 4: Generating embeddings...")
        embeddings = self.generate_embeddings(setups)
        
        # Step 5: Store in vector store
        logger.info("Step 5: Storing in vector store...")
        self.initialize_store()
        ids = self.store_setups(
            setups=setups,
            embeddings=embeddings,
            prices=prices,
            timestamps=timestamps,
            symbol=symbol,
            timeframe=timeframe
        )
        
        # Get final stats
        stats = self.store.get_stats()
        
        logger.info("Pipeline complete!")
        
        return {
            'num_setups': len(setups),
            'num_stored': len(ids),
            'training_history': history,
            'store_stats': stats
        }
    
    def close(self):
        """Cleanup resources"""
        if self.store:
            self.store.save()


def run_pipeline_from_csv(
    csv_path: str,
    price_column: str = 'close',
    timestamp_column: str = 'timestamp',
    symbol: str = 'BTCUSDT',
    timeframe: str = '1m',
    **kwargs
):
    """
    Convenience function to run pipeline from CSV data.
    
    Args:
        csv_path: Path to CSV file
        price_column: Column name for prices
        timestamp_column: Column name for timestamps
        symbol: Trading symbol
        timeframe: Data timeframe
        **kwargs: Additional pipeline arguments
    """
    import pandas as pd
    
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    prices = df[price_column].values.astype(np.float64)
    
    if timestamp_column in df.columns:
        # Try to parse timestamps
        try:
            timestamps = pd.to_datetime(df[timestamp_column]).astype(np.int64) / 1e9
        except:  # noqa: E722
            timestamps = df[timestamp_column].values.astype(np.float64)
    else:
        timestamps = np.arange(len(prices), dtype=np.float64)
    
    pipeline = TradePipeline(**kwargs)
    
    try:
        result = pipeline.run_full_pipeline(
            prices=prices,
            timestamps=timestamps,
            symbol=symbol,
            timeframe=timeframe
        )
        return result
    finally:
        pipeline.close()


def run_pipeline_from_price_client(
    price_client,
    token: str,
    days: int = 5,
    **kwargs
):
    """
    Convenience function to run pipeline using PriceServerClient.
    
    Args:
        price_client: PriceServerClient instance
        token: Token symbol
        days: Days of historical data to load
        **kwargs: Additional pipeline arguments
    """
    logger.info(f"Loading {days} days of data for {token}")
    
    data = price_client.load_historical_prices(token, days=days)
    
    if data is None or len(data) == 0:
        raise ValueError(f"No data available for {token}")
    
    prices = data[:, 0].astype(np.float64)
    timestamps = data[:, 1].astype(np.float64)
    
    pipeline = TradePipeline(**kwargs)
    
    try:
        result = pipeline.run_full_pipeline(
            prices=prices,
            timestamps=timestamps,
            symbol=token,
            timeframe='1s'
        )
        return result
    finally:
        pipeline.close()


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trade Setup Embedding Pipeline")
    parser.add_argument('--csv', type=str, help='Path to CSV file with price data')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1m', help='Data timeframe')
    parser.add_argument('--window-size', type=int, default=100, help='Price window size')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--model-path', type=str, default='models/trained', help='Model save path')
    parser.add_argument('--store-path', type=str, default='vector_store', help='Vector store path')
    
    # Demo mode with synthetic data
    parser.add_argument('--demo', action='store_true', help='Run with synthetic demo data')
    parser.add_argument('--demo-length', type=int, default=10000, help='Length of demo data')
    
    args = parser.parse_args()
    
    def main():
        if args.demo:
            # Generate synthetic data
            logger.info(f"Generating {args.demo_length} synthetic price points")
            
            np.random.seed(42)
            
            # GBM simulation
            dt = 1.0 / (365 * 24 * 60)  # 1 minute
            mu = 0.0
            sigma = 0.5
            
            prices = [50000.0]
            for i in range(args.demo_length - 1):
                z = np.random.normal(0, 1)
                drift = (mu - 0.5 * sigma**2) * dt
                diffusion = sigma * np.sqrt(dt) * z
                new_price = prices[-1] * np.exp(drift + diffusion)
                prices.append(new_price)
            
            prices = np.array(prices)
            timestamps = np.arange(len(prices), dtype=np.float64) * 60  # 1 minute intervals
            
            pipeline = TradePipeline(
                window_size=args.window_size,
                embedding_dim=args.embedding_dim,
                store_path=args.store_path
            )
            
            try:
                result = pipeline.run_full_pipeline(
                    prices=prices,
                    timestamps=timestamps,
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    train_epochs=args.epochs,
                    model_save_path=Path(args.model_path) if args.model_path else None
                )
                
                logger.info(f"Pipeline result: {json.dumps(result, indent=2, default=str)}")
            finally:
                pipeline.close()
                
        elif args.csv:
            result = run_pipeline_from_csv(
                csv_path=args.csv,
                symbol=args.symbol,
                timeframe=args.timeframe,
                window_size=args.window_size,
                embedding_dim=args.embedding_dim,
                store_path=args.store_path
            )
            logger.info(f"Pipeline result: {json.dumps(result, indent=2, default=str)}")
        else:
            parser.print_help()
    
    main()
