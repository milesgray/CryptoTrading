from enum import IntEnum
from dataclasses import dataclass

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
