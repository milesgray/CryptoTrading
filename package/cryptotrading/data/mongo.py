
"""Database connection and session management."""
import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from dotenv import load_dotenv
load_dotenv()

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorClientSession, AsyncIOMotorDatabase


logger = logging.getLogger(__name__)

client: AsyncIOMotorClient | None = None
session: AsyncIOMotorClientSession | None = None

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "crypto_prices")
PRICE_COLLECTION_NAME = os.getenv("MONGO_PRICE_COLLECTION_NAME", "price_data")
ORDER_BOOK_COLLECTION_NAME = os.getenv("MONGO_ORDER_BOOK_COLLECTION_NAME", "order_book_data")

logger.info(f"MONGO URI loaded: {MONGO_URI}\nDB NAME: {DB_NAME}\nPRICE COLLECTION NAME: {PRICE_COLLECTION_NAME}\nORDER BOOK COLLECTION NAME: {ORDER_BOOK_COLLECTION_NAME}")

@asynccontextmanager
async def get_session(
    client: AsyncIOMotorClient,
) -> AsyncGenerator[AsyncIOMotorClientSession | None, None]:
    """Yield a MongoDB session with transaction management."""
    session = await client.start_session()
    try:
        await session.start_transaction()  # type: ignore[misc]
        yield session
        await session.commit_transaction()
    except Exception as e:
        await session.abort_transaction()
        raise e
    finally:
        await session.end_session()
        session = None  # type: ignore[assignment]


def get_client() -> AsyncIOMotorClient | None:
    """Get the MongoDB client instance."""
    if client is None:
        connect_to_mongo()
    return client


def connect_to_mongo(uri: str = MONGO_URI) -> None:
    """Connect to MongoDB using the provided URI."""
    global client
    if client is None:
        client = AsyncIOMotorClient(uri)
        logger.info("Connected to MongoDB.")
    else:
        logger.warning("MongoDB client is already connected.")


def close_mongo_connection() -> None:
    """Close the MongoDB connection."""
    global client
    if client:
        client.close()
        client = None  # ignore: type[assignment]
        logger.info("MongoDB connection closed.")
    else:
        logger.warning("MongoDB client is already closed.")


def get_db() -> AsyncIOMotorDatabase | None:
    """Yield a MongoDB database instance."""
    client = get_client()
    if client is None:
        logger.error("Failed to connect to MongoDB client.")
        raise ConnectionError("Failed to retrieve MongoDB client.")

    db = client.get_default_database(DB_NAME)
    # logger.info(f"Connected to database: {db.name}")
    return db    


def get_db_client() -> AsyncIOMotorClient | None:
    """Yield a MongoDB client instance."""
    client = get_client()
    if client is None:
        logger.error("Failed to retrieve MongoDB client.")
        raise ConnectionError("Failed to retrieve MongoDB client.")

    logger.info("Providing MongoDB client.")
    return client    
