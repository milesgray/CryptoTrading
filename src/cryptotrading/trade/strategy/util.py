from enum import IntEnum
from dataclasses import dataclass
from typing import Type
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class OracleAction(IntEnum):
    """Oracle action types"""
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3


@dataclass
class OracleTradeSegment:
    """A complete trade segment identified by the oracle"""
    start_idx: int
    end_idx: int
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    profit_pct: float
    leverage: float
    max_adverse_excursion: float  # Max drawdown during trade


def generate_oracle_labels(
    oracle_type: Type,
    prices: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    max_leverage: float = 20.0,
    transaction_cost: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, List[OracleTradeSegment]]:
    """
    Convenience function to generate oracle labels for a price series.
    
    Args:
        prices: Price array
        timestamps: Optional timestamp array
        max_leverage: Maximum leverage
        transaction_cost: Transaction cost as fraction
        
    Returns:
        actions: Array of OracleAction values
        leverages: Array of leverage values
        segments: List of trade segments
    """
    oracle = oracle_type(
        max_leverage=max_leverage,
        transaction_cost=transaction_cost
    )
    
    actions, leverages = oracle.compute_oracle_actions(prices)
    segments = oracle.extract_trade_segments(prices, timestamps)
    
    logger.info(f"Generated oracle labels for {len(prices)} price points")
    logger.info(f"Found {len(segments)} trade segments")
    
    stats = oracle.get_statistics(prices)
    logger.info(f"Oracle stats: {stats}")
    
    return actions, leverages, segments
