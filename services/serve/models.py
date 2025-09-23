from datetime import datetime    
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


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
