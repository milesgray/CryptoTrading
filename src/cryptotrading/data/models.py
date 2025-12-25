from datetime import datetime    
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from typing import Tuple
import warnings

class OrderBookSnapshot(BaseModel):
    """Single order book snapshot with validation"""
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, volume), ...]
    asks: List[Tuple[float, float]]
    mid_price: float

    @classmethod
    def from_mongodb_doc(cls, doc: Dict[str, Any]) -> 'OrderBookSnapshot':
        """Create a OrderBookSnapshot from a MongoDB document."""
        return cls(
            timestamp=doc['timestamp'].timestamp(),
            bids=sorted(doc['book']['bids'], key=lambda x: x[0], reverse=True),
            asks=sorted(doc['book']['asks'], key=lambda x: x[0]),
            mid_price=doc['metadata']['midpoint']
        )
    
    def __post_init__(self):
        """Validate order book data"""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate order book for data quality issues.
        
        Checks:
        - Non-empty sides
        - No crossed book (best bid < best ask)
        - No negative prices/volumes
        - Monotonic price levels
        """
        if not self.bids or not self.asks:
            raise ValueError("Order book has empty side")
        
        # Check for crossed book
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        
        if best_bid >= best_ask:
            raise ValueError(f"Crossed book: bid={best_bid} >= ask={best_ask}")
        
        # Check for negative values
        for price, vol in self.bids + self.asks:
            if price <= 0 or vol <= 0:
                raise ValueError(f"Invalid price/volume: {price}/{vol}")
        
        # Check bid prices are descending
        bid_prices = [p for p, _ in self.bids]
        if bid_prices != sorted(bid_prices, reverse=True):
            warnings.warn("Bid prices not monotonically descending")
        
        # Check ask prices are ascending
        ask_prices = [p for p, _ in self.asks]
        if ask_prices != sorted(ask_prices):
            warnings.warn("Ask prices not monotonically ascending")
        
        # Verify mid price
        expected_mid = (best_bid + best_ask) / 2
        if abs(self.mid_price - expected_mid) > 0.01 * expected_mid:
            warnings.warn(f"Mid price mismatch: {self.mid_price} vs {expected_mid}")
            self.mid_price = expected_mid
        
        return True

class TransformedOrderBookDataPoint(BaseModel):
    timestamp: datetime
    lowest_ask: float
    highest_bid: float
    midpoint: float
    spread: float
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
    
    @classmethod
    def from_mongodb_doc(cls, doc: Dict[str, Any]) -> 'TransformedOrderBookDataPoint':
        """Create a TransformedOrderBookDataPoint from a MongoDB document."""
        if 'metadata' in doc and 'lowest_ask' in doc['metadata']:
            # Handle documents where fields are nested under metadata
            return cls(
                timestamp=doc['timestamp'],
                lowest_ask=doc['metadata']['lowest_ask'],
                highest_bid=doc['metadata']['highest_bid'],
                midpoint=doc['metadata'].get('midpoint', (doc['metadata']['lowest_ask'] + doc['metadata']['highest_bid']) / 2),
                spread=doc['metadata'].get('spread', abs(doc['metadata']['lowest_ask'] - doc['metadata']['highest_bid']))
            )
        else:
            # Handle documents where fields are at the top level
            return cls(
                timestamp=doc['timestamp'],
                lowest_ask=doc['lowest_ask'],
                highest_bid=doc['highest_bid'],
                midpoint=doc.get('midpoint', (doc['lowest_ask'] + doc['highest_bid']) / 2),
                spread=doc.get('spread', abs(doc['lowest_ask'] - doc['highest_bid']))
            )

class TransformedOrderBookData(BaseModel):
    token: str
    points: list[TransformedOrderBookDataPoint]
    
class PaginatedResponse(BaseModel):
    data: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool

class PriceDataPoint(BaseModel):
    timestamp: datetime
    price: float
    book: dict
    token: str

    @classmethod
    def from_mongodb_doc(cls, doc: Dict[str, Any]) -> 'PriceDataPoint':
        """Create a PriceDataPoint from a MongoDB document."""
        return cls(
            timestamp=doc['timestamp'],
            price=doc['price'],
            book=doc['metadata']['book'],
            token=doc['metadata']['token']
        )
class PriceBucket(BaseModel):
    range: str
    avg_price: float
    volume: float

class PriceOutlier(BaseModel):
    price: float
    volume: float

class OrderBookSummaryData(BaseModel):
    volume: float
    bid_buckets: List[PriceBucket] = []
    ask_buckets: List[PriceBucket] = []
    bid_outliers: List[PriceOutlier] = []
    ask_outliers: List[PriceOutlier] = []

class LatestPriceData(BaseModel):
    price: float
    timestamp: datetime
    order_book: Optional[OrderBookSummaryData] = None

class CandlestickData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None  # Traditional candlestick volume (if you have it)
    exchange_count: Optional[float] = None
    order_book: Optional[OrderBookSummaryData] = None

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
