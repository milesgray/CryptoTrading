import logging
import datetime

from pymongo import ASCENDING

from cryptotrading.data.mongo import get_db
from cryptotrading.config import (
    PRICE_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

class PriceMongoAdapter:
    def __init__(self):
        self.db = get_db()

    async def initialize(self):
        self.collections = await self.db.list_collection_names()
        await self.init_price_collection()
     
    async def shutdown(self):
        pass

    async def init_price_collection(self):
        self.price_collection = self.db[PRICE_COLLECTION_NAME]
        
        # Create time series collection if it doesn't exist
        if PRICE_COLLECTION_NAME not in self.collections:
            try:
                await self.db.create_collection(
                    PRICE_COLLECTION_NAME,
                    timeseries={
                        'timeField': 'timestamp',
                        'metaField': 'price',
                        'granularity': 'seconds'
                    }
                )
            except:
                await self.db.create_collection(PRICE_COLLECTION_NAME)            
            # Create indexes for faster queries
            await self.price_collection.create_index([("timestamp", ASCENDING)])
            await self.price_collection.create_index([("metadata.symbol", ASCENDING)])

    async def store_price_data(
        self, 
        symbol: str, 
        index_price: float, 
        book: dict,            
        raw_data: list[dict], 
        verbose: bool = False
    ) -> None:
        """Store calculated index price and raw data in MongoDB time series collection"""
        timestamp = datetime.datetime.now(datetime.UTC)
        if verbose: logger.info(f"Storing price data! {symbol}: {index_price}")
        token = symbol.split("/")[0] if "/" in symbol else symbol
        # Store calculated index price
        index_doc = {
            "timestamp": timestamp,
            "metadata": {
                "token": token,
                "symbol": symbol,
                "book": book,                
                "type": "index_price",
            },
            "price": index_price,
            "exchanges_count": len(raw_data)
        }

        try:
            await self.price_collection.insert_one(index_doc)
            
            logger.debug(f"Stored price data for {symbol}: {index_price}")
        except Exception as e:
            logger.error(f"Failed to store price data: {str(e)}")
        