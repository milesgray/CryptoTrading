"""
PostgreSQL adapter with TimescaleDB and pgvector support for the crypto trading application.

This module provides an async interface to PostgreSQL with the following features:
- Connection pooling for efficient database access
- Support for both regular and TimescaleDB hypertables
- Vector similarity search with pgvector
- Transaction management with async context managers
- Automatic schema initialization and migration
"""
import os
import logging
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import (
    Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast
)

import asyncpg
from asyncpg import Connection, Pool, create_pool
from asyncpg.pool import PoolAcquireContext
from asyncpg.types import Type as PgType
from pydantic import BaseModel
from typing_extensions import TypeVarTuple, Unpack

from cryptotrading.config import (
    POSTGRES_PARAMS,
    POSTGRES_POOL_MIN,
    POSTGRES_POOL_MAX,
    POSTGRES_USE_TIMESCALE,
    POSTGRES_USE_PGVECTOR,
    logger
)

# Type variables for generic typing
T = TypeVar('T', bound=BaseModel)
Ts = TypeVarTuple('Ts')

# Global connection pool
_pool: Optional[Pool] = None

# Custom JSON encoder for handling datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Custom JSON codec for asyncpg
class JSONBEncoder(asyncpg.TypeCodec):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = DateTimeEncoder()

    def encode(self, value):
        return json.dumps(value, cls=DateTimeEncoder).encode('utf-8')

    def decode(self, value, *args):
        return json.loads(value.decode('utf-8'))

# Custom type codecs for better type handling
class TypeCodec(asyncpg.TypeCodec):
    def __init__(self, oid: int, format_code: int, python_type: type, encoder, decoder):
        self.oid = oid
        self.format_code = format_code
        self.python_type = python_type
        self._encoder = encoder
        self._decoder = decoder

    def encode(self, value: Any) -> bytes:
        if value is None:
            return b'NULL'
        return self._encoder(value).encode('utf-8')

    def decode(self, value: bytes, *args) -> Any:
        if value is None:
            return None
        return self._decoder(value.decode('utf-8'))

# Initialize database connection pool
async def init_pool(**overrides) -> Pool:
    """Initialize the PostgreSQL connection pool."""
    global _pool
    
    if _pool is not None:
        return _pool
    
    # Merge default params with overrides
    params = {**POSTGRES_PARAMS, **overrides}
    
    # Initialize pool with connection settings
    _pool = await create_pool(
        min_size=POSTGRES_POOL_MIN,
        max_size=POSTGRES_POOL_MAX,
        **params,
        init=init_connection,
        command_timeout=60,  # 60 seconds command timeout
        max_inactive_connection_lifetime=300,  # 5 minutes
        max_queries=50000,  # Connections will be recycled after this many queries
        setup=setup_connection,
    )
    
    # Initialize database schema
    async with _pool.acquire() as conn:
        await init_schema(conn)
    
    return _pool

# Initialize a connection with custom type handlers
async def init_connection(conn: Connection):
    """Initialize a new database connection with custom type handlers."""
    # Register JSON codec
    await conn.set_type_codec(
        'jsonb',
        encoder=json.dumps,
        decoder=json.loads,
        schema='pg_catalog',
        format='text'
    )
    
    # Register timestamp with timezone codec
    await conn.set_type_codec(
        'timestamptz',
        encoder=lambda x: x.isoformat() if x else None,
        decoder=lambda x: datetime.fromisoformat(x) if x else None,
        schema='pg_catalog',
        format='text'
    )
    
    # Register vector type if pgvector is enabled
    if POSTGRES_USE_PGVECTOR:
        try:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            
            # Register vector type codec
            await conn.set_type_codec(
                'vector',
                encoder=lambda x: str(x) if x else None,
                decoder=lambda x: [float(v) for v in x[1:-1].split(',')] if x else None,
                schema='pg_catalog',
                format='text'
            )
        except Exception as e:
            logger.warning(f"Failed to initialize pgvector extension: {e}")

# Setup connection (called for each new connection in the pool)
async def setup_connection(conn: Connection):
    """Setup connection with custom settings."""
    # Set timezone to UTC
    await conn.execute('SET TIME ZONE UTC')
    
    # Set statement timeout to 30 seconds
    await conn.execute('SET statement_timeout = 30000')
    
    # Ensure search_path is set correctly
    await conn.execute('SET search_path TO public')

# Initialize database schema
async def init_schema(conn: Connection):
    """Initialize database schema and extensions."""
    # Enable TimescaleDB if configured
    if POSTGRES_USE_TIMESCALE:
        try:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE')
        except Exception as e:
            logger.warning(f"Failed to initialize TimescaleDB extension: {e}")
    
    # Create tables if they don't exist
    await conn.execute('''
    CREATE TABLE IF NOT EXISTS price_data (
        time TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        exchange TEXT NOT NULL,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume DOUBLE PRECISION,
        metadata JSONB,
        PRIMARY KEY (time, symbol, exchange)
    );
    
    CREATE TABLE IF NOT EXISTS order_book_data (
        time TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        exchange TEXT NOT NULL,
        is_bid BOOLEAN NOT NULL,
        price DOUBLE PRECISION NOT NULL,
        amount DOUBLE PRECISION NOT NULL,
        metadata JSONB,
        PRIMARY KEY (time, symbol, exchange, is_bid, price)
    );
    
    CREATE TABLE IF NOT EXISTS trade_data (
        id TEXT PRIMARY KEY,
        time TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        exchange TEXT NOT NULL,
        side TEXT NOT NULL,
        price DOUBLE PRECISION NOT NULL,
        amount DOUBLE PRECISION NOT NULL,
        cost DOUBLE PRECISION GENERATED ALWAYS AS (price * amount) STORED,
        fee_currency TEXT,
        fee_cost DOUBLE PRECISION,
        metadata JSONB
    );
    
    CREATE TABLE IF NOT EXISTS tweet_data (
        id TEXT PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL,
        author_id TEXT,
        text TEXT NOT NULL,
        sentiment_score DOUBLE PRECISION,
        sentiment_magnitude DOUBLE PRECISION,
        entities JSONB,
        metadata JSONB
    );
    
    -- Create vector extension table if pgvector is enabled
    {'CREATE TABLE IF NOT EXISTS document_embeddings ('
     'id TEXT PRIMARY KEY,'
     'content TEXT NOT NULL,'
     'embedding VECTOR(1536),'  -- Default dimension for OpenAI embeddings
     'metadata JSONB,'
     'created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()'
     ');' if POSTGRES_USE_PGVECTOR else ''}
    ''')
    
    # Convert regular tables to hypertables if TimescaleDB is enabled
    if POSTGRES_USE_TIMESCALE:
        try:
            await conn.execute('''
            SELECT create_hypertable('price_data', 'time', if_not_exists => TRUE);
            SELECT create_hypertable('order_book_data', 'time', if_not_exists => TRUE);
            SELECT create_hypertable('trade_data', 'time', if_not_exists => TRUE);
            
            -- Create appropriate indexes
            CREATE INDEX IF NOT EXISTS idx_price_data_symbol_time 
                ON price_data(symbol, time DESC);
                
            CREATE INDEX IF NOT EXISTS idx_order_book_data_symbol_time 
                ON order_book_data(symbol, time DESC);
                
            CREATE INDEX IF NOT EXISTS idx_trade_data_symbol_time 
                ON trade_data(symbol, time DESC);
                
            -- Create index on tweet data for faster lookups
            CREATE INDEX IF NOT EXISTS idx_tweet_data_created_at 
                ON tweet_data(created_at DESC);
                
            -- Create GIN index on JSONB columns for better query performance
            CREATE INDEX IF NOT EXISTS idx_price_data_metadata 
                ON price_data USING GIN (metadata);
                
            CREATE INDEX IF NOT EXISTS idx_trade_data_metadata 
                ON trade_data USING GIN (metadata);
                
            -- Add compression to historical data
            SELECT add_compression_policy('price_data', INTERVAL '7 days', if_not_exists => TRUE);
            SELECT add_compression_policy('order_book_data', INTERVAL '7 days', if_not_exists => TRUE);
            SELECT add_compression_policy('trade_data', INTERVAL '7 days', if_not_exists => TRUE);
            
            -- Add retention policy (keep data for 1 year by default)
            SELECT add_retention_policy('price_data', INTERVAL '1 year', if_not_exists => TRUE);
            SELECT add_retention_policy('order_book_data', INTERVAL '1 year', if_not_exists => TRUE);
            SELECT add_retention_policy('trade_data', INTERVAL '1 year', if_not_exists => TRUE);
            ''')
        except Exception as e:
            logger.error(f"Failed to set up TimescaleDB hypertables and policies: {e}")
    
    # Create vector index if pgvector is enabled
    if POSTGRES_USE_PGVECTOR:
        try:
            await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_document_embeddings_embedding 
                ON document_embeddings 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            
            -- Create GIN index on JSONB metadata
            CREATE INDEX IF NOT EXISTS idx_document_embeddings_metadata 
                ON document_embeddings USING GIN (metadata);
            ''')
        except Exception as e:
            logger.error(f"Failed to set up pgvector indexes: {e}")

# Database session context manager
@asynccontextmanager
async def get_connection() -> AsyncGenerator[Connection, None]:
    """Get a database connection from the pool."""
    if _pool is None:
        await init_pool()
    
    async with _pool.acquire() as conn:
        try:
            yield conn
        except Exception as e:
            await conn.execute('ROLLBACK')
            logger.error(f"Database error: {e}")
            raise

# Transaction context manager
@asynccontextmanager
async def transaction() -> AsyncGenerator[Connection, None]:
    """Transaction context manager for database operations."""
    if _pool is None:
        await init_pool()
    
    async with _pool.acquire() as conn:
        async with conn.transaction():
            try:
                yield conn
            except Exception as e:
                logger.error(f"Transaction failed: {e}")
                raise

# Generic CRUD operations
class Database:
    """Database access layer with common CRUD operations."""
    
    def __init__(self, table: str, id_field: str = 'id'):
        self.table = table
        self.id_field = id_field
    
    async def insert(self, data: Dict[str, Any], conn: Optional[Connection] = None) -> str:
        """Insert a new record into the database."""
        if not data:
            raise ValueError("No data provided for insertion")
            
        columns = ', '.join(data.keys())
        placeholders = ', '.join(f'${i+1}' for i in range(len(data)))
        values = list(data.values())
        
        query = f'''
        INSERT INTO {self.table} ({columns})
        VALUES ({placeholders})
        RETURNING {self.id_field};
        '''
        
        if conn:
            return await conn.fetchval(query, *values)
        
        async with get_connection() as conn:
            return await conn.fetchval(query, *values)
    
    async def get(self, id: str, conn: Optional[Connection] = None) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        query = f'SELECT * FROM {self.table} WHERE {self.id_field} = $1;'
        
        if conn:
            row = await conn.fetchrow(query, id)
        else:
            async with get_connection() as conn:
                row = await conn.fetchrow(query, id)
        
        return dict(row) if row else None
    
    async def update(
        self, 
        id: str, 
        data: Dict[str, Any], 
        conn: Optional[Connection] = None
    ) -> bool:
        """Update a record by ID."""
        if not data:
            return False
            
        set_clause = ', '.join(f"{k} = ${i+2}" for i, k in enumerate(data.keys()))
        values = list(data.values()) + [id]
        
        query = f'''
        UPDATE {self.table}
        SET {set_clause}
        WHERE {self.id_field} = ${len(values)};
        '''
        
        if conn:
            result = await conn.execute(query, *values)
        else:
            async with get_connection() as conn:
                result = await conn.execute(query, *values)
        
        return 'UPDATE 1' in result
    
    async def delete(self, id: str, conn: Optional[Connection] = None) -> bool:
        """Delete a record by ID."""
        query = f'DELETE FROM {self.table} WHERE {self.id_field} = $1;'
        
        if conn:
            result = await conn.execute(query, id)
        else:
            async with get_connection() as conn:
                result = await conn.execute(query, id)
        
        return 'DELETE 1' in result
    
    async def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        conn: Optional[Connection] = None
    ) -> List[Dict[str, Any]]:
        """Query records with optional filtering and pagination."""
        where_clause = ''
        values = []
        
        if filters:
            conditions = []
            for i, (key, value) in enumerate(filters.items(), 1):
                if isinstance(value, (list, tuple)):
                    placeholders = ', '.join([f'${i+j+1}' for j in range(len(value))])
                    conditions.append(f"{key} IN ({placeholders})")
                    values.extend(value)
                    i += len(value) - 1
                else:
                    conditions.append(f"{key} = ${i}")
                    values.append(value)
            where_clause = 'WHERE ' + ' AND '.join(conditions)
        
        query = f'SELECT * FROM {self.table} {where_clause}'
        
        if order_by:
            query += f' ORDER BY {order_by}'
            if desc:
                query += ' DESC'
        
        if limit is not None:
            query += f' LIMIT {limit}'
            if offset is not None:
                query += f' OFFSET {offset}'
        
        if conn:
            rows = await conn.fetch(query, *values)
        else:
            async with get_connection() as conn:
                rows = await conn.fetch(query, *values)
        
        return [dict(row) for row in rows]

# Specialized repositories
class PriceDataRepository(Database):
    """Repository for price data operations."""
    
    def __init__(self):
        super().__init__('price_data', 'time')
    
    async def get_latest_prices(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        limit: int = 100,
        conn: Optional[Connection] = None
    ) -> List[Dict[str, Any]]:
        """Get the latest price data for a symbol and optional exchange."""
        query = '''
        SELECT * FROM price_data
        WHERE symbol = $1
        '''
        
        params = [symbol]
        
        if exchange:
            query += ' AND exchange = $2'
            params.append(exchange)
        
        query += ' ORDER BY time DESC LIMIT $' + str(len(params) + 1)
        params.append(limit)
        
        if conn:
            rows = await conn.fetch(query, *params)
        else:
            async with get_connection() as conn:
                rows = await conn.fetch(query, *params)
        
        return [dict(row) for row in rows]
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        exchange: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        conn: Optional[Connection] = None
    ) -> List[Dict[str, Any]]:
        """Get OHLCV (Open, High, Low, Close, Volume) data with time bucketing."""
        if not POSTGRES_USE_TIMESCALE:
            raise RuntimeError("TimescaleDB is required for time_bucket operations")
        
        query = '''
        SELECT 
            time_bucket($1::interval, time) AS bucket,
            first(open, time) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, time) AS close,
            sum(volume) AS volume,
            symbol,
            exchange
        FROM price_data
        WHERE symbol = $2
        '''
        
        params = [timeframe, symbol]
        param_count = 2
        
        if exchange:
            param_count += 1
            query += f' AND exchange = ${param_count}'
            params.append(exchange)
        
        if start_time:
            param_count += 1
            query += f' AND time >= ${param_count}'
            params.append(start_time)
        
        if end_time:
            param_count += 1
            query += f' AND time <= ${param_count}'
            params.append(end_time)
        
        query += f'''
        GROUP BY bucket, symbol, exchange
        ORDER BY bucket DESC
        LIMIT ${param_count + 1}
        '''
        params.append(limit)
        
        if conn:
            rows = await conn.fetch(query, *params)
        else:
            async with get_connection() as conn:
                rows = await conn.fetch(query, *params)
        
        return [dict(row) for row in rows]

class OrderBookRepository(Database):
    """Repository for order book data operations."""
    
    def __init__(self):
        super().__init__('order_book_data', 'time')
    
    async def get_order_book_snapshot(
        self,
        symbol: str,
        exchange: str,
        time: Optional[datetime] = None,
        depth: int = 10,
        conn: Optional[Connection] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get order book snapshot for a specific time or the latest."""
        if time is None:
            time = datetime.now(timezone.utc)
        
        # Get bids
        bids_query = '''
        SELECT price, amount
        FROM order_book_data
        WHERE symbol = $1 
          AND exchange = $2
          AND is_bid = TRUE
          AND time <= $3
        ORDER BY price DESC, time DESC
        LIMIT $4;
        '''
        
        # Get asks
        asks_query = '''
        SELECT price, amount
        FROM order_book_data
        WHERE symbol = $1 
          AND exchange = $2
          AND is_bid = FALSE
          AND time <= $3
        ORDER BY price ASC, time DESC
        LIMIT $4;
        '''
        
        if conn:
            bids = await conn.fetch(bids_query, symbol, exchange, time, depth)
            asks = await conn.fetch(asks_query, symbol, exchange, time, depth)
        else:
            async with get_connection() as conn:
                bids = await conn.fetch(bids_query, symbol, exchange, time, depth)
                asks = await conn.fetch(asks_query, symbol, exchange, time, depth)
        
        return {
            'bids': [{'price': float(bid['price']), 'amount': float(bid['amount'])} for bid in bids],
            'asks': [{'price': float(ask['price']), 'amount': float(ask['amount'])} for ask in asks]
        }

class DocumentEmbeddingRepository(Database):
    """Repository for document embeddings with pgvector support."""
    
    def __init__(self):
        if not POSTGRES_USE_PGVECTOR:
            raise RuntimeError("pgvector is not enabled")
        super().__init__('document_embeddings', 'id')
    
    async def find_similar(
        self,
        embedding: List[float],
        limit: int = 5,
        min_similarity: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        conn: Optional[Connection] = None
    ) -> List[Dict[str, Any]]:
        """Find documents similar to the given embedding."""
        if not embedding:
            return []
        
        where_clause = ''
        params = [embedding, min_similarity, limit]
        
        if filters:
            conditions = []
            for i, (key, value) in enumerate(filters.items(), 4):  # Start from $4
                if isinstance(value, (list, tuple)):
                    placeholders = ', '.join([f'${i+j}' for j in range(len(value))])
                    conditions.append(f"{key} IN ({placeholders})")
                    params.extend(value)
                    i += len(value)
                else:
                    conditions.append(f"{key} = ${i}")
                    params.append(value)
            where_clause = 'AND ' + ' AND '.join(conditions)
        
        query = f'''
        SELECT 
            id,
            content,
            metadata,
            1 - (embedding <=> $1) AS similarity
        FROM document_embeddings
        WHERE 1 - (embedding <=> $1) > $2
        {where_clause}
        ORDER BY embedding <=> $1
        LIMIT $3;
        '''
        
        if conn:
            rows = await conn.fetch(query, *params)
        else:
            async with get_connection() as conn:
                rows = await conn.fetch(query, *params)
        
        return [dict(row) for row in rows]

# Initialize database connection on module import
async def init_db():
    """Initialize the database connection pool."""
    try:
        await init_pool()
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database connection pool: {e}")
        raise

# Clean up database connections
async def close_db():
    """Close all database connections in the pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")

# Create repository instances
price_repo = PriceDataRepository()
order_book_repo = OrderBookRepository()
document_embedding_repo = DocumentEmbeddingRepository() if POSTGRES_USE_PGVECTOR else None

# Initialize database on module import
if __name__ != "__main__":
    import asyncio
    
    async def _init():
        try:
            await init_db()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(_init())
    else:
        loop.run_until_complete(_init())