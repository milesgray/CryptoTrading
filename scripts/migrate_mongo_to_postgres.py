#!/usr/bin/env python3
"""
Migration script to move data from MongoDB to PostgreSQL.

This script helps migrate data from existing MongoDB collections to the new PostgreSQL database.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.database import Database as MongoDatabase

from cryptotrading.config import (
    MONGO_URI,
    DB_NAME,
    PRICE_COLLECTION_NAME,
    COMPOSITE_ORDER_BOOK_COLLECTION_NAME,
    EXCHANGE_ORDER_BOOK_COLLECTION_NAME,
    TRANSFORMED_ORDER_BOOK_COLLECTION_NAME,
    TWEET_COLLECTION_NAME,
    logger
)
from cryptotrading.data.postgres import (
    init_db,
    price_repo,
    order_book_repo,
    document_embedding_repo
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoToPostgresMigrator:
    """Handle migration of data from MongoDB to PostgreSQL."""
    
    def __init__(self, mongo_uri: str, db_name: str):
        """Initialize with MongoDB connection details."""
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.mongo_db: Optional[MongoDatabase] = None
    
    async def connect(self):
        """Establish connections to both MongoDB and PostgreSQL."""
        # Connect to MongoDB
        self.mongo_client = AsyncIOMotorClient(self.mongo_uri)
        self.mongo_db = self.mongo_client[self.db_name]
        
        # Initialize PostgreSQL connection
        await init_db()
        logger.info("Connected to both MongoDB and PostgreSQL")
    
    async def close(self):
        """Close all database connections."""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("Closed MongoDB connection")
    
    async def migrate_price_data(self, batch_size: int = 1000) -> int:
        """Migrate price data from MongoDB to PostgreSQL."""
        if not self.mongo_db:
            raise RuntimeError("MongoDB connection not established")
        
        collection = self.mongo_db[PRICE_COLLECTION_NAME]
        total_docs = await collection.count_documents({})
        logger.info(f"Found {total_docs} price documents to migrate")
        
        migrated_count = 0
        skip = 0
        
        while skip < total_docs:
            cursor = collection.find().skip(skip).limit(batch_size)
            batch = await cursor.to_list(length=batch_size)
            
            for doc in batch:
                try:
                    # Transform document to match PostgreSQL schema
                    price_data = {
                        'time': doc.get('timestamp') or doc.get('time') or datetime.utcnow(),
                        'symbol': doc.get('symbol', '').upper(),
                        'exchange': doc.get('exchange', '').lower(),
                        'open': float(doc.get('open', 0)),
                        'high': float(doc.get('high', 0)),
                        'low': float(doc.get('low', 0)),
                        'close': float(doc.get('close', 0)),
                        'volume': float(doc.get('volume', 0)),
                        'metadata': {
                            'source': 'migration',
                            'migrated_at': datetime.utcnow().isoformat(),
                            'mongo_id': str(doc.get('_id', ''))
                        }
                    }
                    
                    # Insert into PostgreSQL
                    await price_repo.insert(price_data)
                    migrated_count += 1
                    
                    # Log progress
                    if migrated_count % 100 == 0:
                        logger.info(f"Migrated {migrated_count}/{total_docs} price records")
                        
                except Exception as e:
                    logger.error(f"Error migrating price document {doc.get('_id')}: {e}")
            
            skip += batch_size
        
        logger.info(f"Completed price data migration. Migrated {migrated_count} records.")
        return migrated_count
    
    async def migrate_order_book_data(self, batch_size: int = 500) -> int:
        """Migrate order book data from MongoDB to PostgreSQL."""
        if not self.mongo_db:
            raise RuntimeError("MongoDB connection not established")
        
        # Migrate composite order book data
        collection = self.mongo_db[COMPOSITE_ORDER_BOOK_COLLECTION_NAME]
        total_docs = await collection.count_documents({})
        logger.info(f"Found {total_docs} order book documents to migrate")
        
        migrated_count = 0
        skip = 0
        
        while skip < total_docs:
            cursor = collection.find().skip(skip).limit(batch_size)
            batch = await cursor.to_list(length=batch_size)
            
            for doc in batch:
                try:
                    timestamp = doc.get('timestamp') or datetime.utcnow()
                    symbol = doc.get('symbol', '').upper()
                    
                    # Process bids
                    for bid in doc.get('bids', []):
                        order_data = {
                            'time': timestamp,
                            'symbol': symbol,
                            'exchange': 'composite',
                            'is_bid': True,
                            'price': float(bid.get('price', 0)),
                            'amount': float(bid.get('amount', 0)),
                            'metadata': {
                                'source': 'migration',
                                'migrated_at': datetime.utcnow().isoformat(),
                                'mongo_id': str(doc.get('_id', ''))
                            }
                        }
                        await order_book_repo.insert(order_data)
                    
                    # Process asks
                    for ask in doc.get('asks', []):
                        order_data = {
                            'time': timestamp,
                            'symbol': symbol,
                            'exchange': 'composite',
                            'is_bid': False,
                            'price': float(ask.get('price', 0)),
                            'amount': float(ask.get('amount', 0)),
                            'metadata': {
                                'source': 'migration',
                                'migrated_at': datetime.utcnow().isoformat(),
                                'mongo_id': str(doc.get('_id', ''))
                            }
                        }
                        await order_book_repo.insert(order_data)
                    
                    migrated_count += 1
                    
                    # Log progress
                    if migrated_count % 50 == 0:
                        logger.info(f"Migrated {migrated_count}/{total_docs} order book records")
                        
                except Exception as e:
                    logger.error(f"Error migrating order book document {doc.get('_id')}: {e}")
            
            skip += batch_size
        
        logger.info(f"Completed order book data migration. Migrated {migrated_count} records.")
        return migrated_count
    
    async def migrate_tweet_data(self, batch_size: int = 500) -> int:
        """Migrate tweet data from MongoDB to PostgreSQL."""
        if not self.mongo_db:
            raise RuntimeError("MongoDB connection not established")
        
        if not document_embedding_repo:
            logger.warning("pgvector is not enabled. Tweet data migration skipped.")
            return 0
            
        collection = self.mongo_db[TWEET_COLLECTION_NAME]
        total_docs = await collection.count_documents({})
        logger.info(f"Found {total_docs} tweet documents to migrate")
        
        migrated_count = 0
        skip = 0
        
        while skip < total_docs:
            cursor = collection.find().skip(skip).limit(batch_size)
            batch = await cursor.to_list(length=batch_size)
            
            for doc in batch:
                try:
                    # Transform document to match PostgreSQL schema
                    tweet_data = {
                        'id': str(doc.get('_id')),
                        'content': doc.get('text', ''),
                        'created_at': doc.get('created_at') or datetime.utcnow(),
                        'author_id': doc.get('author_id'),
                        'sentiment_score': float(doc.get('sentiment', {}).get('score', 0)),
                        'sentiment_magnitude': float(doc.get('sentiment', {}).get('magnitude', 0)),
                        'entities': doc.get('entities', {}),
                        'metadata': {
                            'source': 'migration',
                            'migrated_at': datetime.utcnow().isoformat(),
                            'mongo_id': str(doc.get('_id', ''))
                        }
                    }
                    
                    # Insert into PostgreSQL
                    await document_embedding_repo.insert(tweet_data)
                    migrated_count += 1
                    
                    # Log progress
                    if migrated_count % 100 == 0:
                        logger.info(f"Migrated {migrated_count}/{total_docs} tweet records")
                        
                except Exception as e:
                    logger.error(f"Error migrating tweet document {doc.get('_id')}: {e}")
            
            skip += batch_size
        
        logger.info(f"Completed tweet data migration. Migrated {migrated_count} records.")
        return migrated_count

async def main():
    """Run the migration process."""
    migrator = MongoToPostgresMigrator(MONGO_URI, DB_NAME)
    
    try:
        # Connect to databases
        await migrator.connect()
        
        # Run migrations
        await migrator.migrate_price_data()
        await migrator.migrate_order_book_data()
        await migrator.migrate_tweet_data()
        
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
    finally:
        # Clean up
        await migrator.close()

if __name__ == "__main__":
    asyncio.run(main())
