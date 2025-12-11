"""
PostgreSQL + pgvector database layer for trade setup embeddings.

Provides efficient similarity search for finding historical trade setups
that match the current market pattern.
"""

import asyncio
import asyncpg
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class StoredTradeSetup:
    """A trade setup stored in the database"""
    id: Optional[int]
    embedding: np.ndarray
    direction: int  # 1=long, -1=short
    profit_pct: float
    leverage: float
    hold_duration: int
    entry_timestamp: float
    entry_price: float
    exit_price: float
    symbol: str
    timeframe: str  # e.g., '1s', '1m', '1h'
    window_size: int
    # Raw price window for visualization
    price_window: Optional[np.ndarray] = None
    created_at: Optional[datetime] = None


@dataclass
class SimilarSetup:
    """Result from similarity search"""
    setup: StoredTradeSetup
    similarity: float  # Cosine similarity (0-1)
    distance: float    # L2 distance


class TradeEmbeddingDB:
    """
    Database interface for trade setup embeddings using pgvector.
    
    Schema:
    - trade_setups: Main table with embeddings and metadata
    - Uses HNSW index for fast approximate nearest neighbor search
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'trade_embeddings',
        user: str = 'postgres',
        password: str = 'postgres',
        embedding_dim: int = 128
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.embedding_dim = embedding_dim
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Establish database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=10
            )
            logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from PostgreSQL")
    
    async def initialize_schema(self):
        """Create tables and indexes"""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create main table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS trade_setups (
                    id SERIAL PRIMARY KEY,
                    embedding vector({self.embedding_dim}),
                    direction SMALLINT NOT NULL,
                    profit_pct REAL NOT NULL,
                    leverage REAL NOT NULL,
                    hold_duration INTEGER NOT NULL,
                    entry_timestamp DOUBLE PRECISION NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    exit_price DOUBLE PRECISION NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    window_size INTEGER NOT NULL,
                    price_window BYTEA,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    
                    -- Indexes for filtering
                    CONSTRAINT valid_direction CHECK (direction IN (-1, 1))
                )
            """)
            
            # Create HNSW index for fast similarity search
            # HNSW is better than IVFFlat for this use case (fewer vectors, need high recall)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS trade_setups_embedding_idx 
                ON trade_setups 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
            
            # Additional indexes for filtering
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS trade_setups_symbol_idx ON trade_setups(symbol)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS trade_setups_direction_idx ON trade_setups(direction)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS trade_setups_profit_idx ON trade_setups(profit_pct)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS trade_setups_timestamp_idx ON trade_setups(entry_timestamp)
            """)
            
            logger.info("Database schema initialized")
    
    async def insert_setup(self, setup: StoredTradeSetup) -> int:
        """Insert a single trade setup"""
        async with self.pool.acquire() as conn:
            # Convert embedding to pgvector format
            embedding_str = '[' + ','.join(map(str, setup.embedding.tolist())) + ']'
            
            # Serialize price window if present
            price_window_bytes = None
            if setup.price_window is not None:
                price_window_bytes = setup.price_window.astype(np.float32).tobytes()
            
            row = await conn.fetchrow("""
                INSERT INTO trade_setups (
                    embedding, direction, profit_pct, leverage, hold_duration,
                    entry_timestamp, entry_price, exit_price, symbol, timeframe,
                    window_size, price_window
                ) VALUES ($1::vector, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
            """,
                embedding_str,
                setup.direction,
                setup.profit_pct,
                setup.leverage,
                setup.hold_duration,
                setup.entry_timestamp,
                setup.entry_price,
                setup.exit_price,
                setup.symbol,
                setup.timeframe,
                setup.window_size,
                price_window_bytes
            )
            
            return row['id']
    
    async def insert_batch(self, setups: List[StoredTradeSetup], batch_size: int = 100) -> List[int]:
        """Insert multiple trade setups efficiently"""
        ids = []
        
        async with self.pool.acquire() as conn:
            # Prepare statement for batch insert
            stmt = await conn.prepare("""
                INSERT INTO trade_setups (
                    embedding, direction, profit_pct, leverage, hold_duration,
                    entry_timestamp, entry_price, exit_price, symbol, timeframe,
                    window_size, price_window
                ) VALUES ($1::vector, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
            """)
            
            for i in range(0, len(setups), batch_size):
                batch = setups[i:i + batch_size]
                
                async with conn.transaction():
                    for setup in batch:
                        embedding_str = '[' + ','.join(map(str, setup.embedding.tolist())) + ']'
                        
                        price_window_bytes = None
                        if setup.price_window is not None:
                            price_window_bytes = setup.price_window.astype(np.float32).tobytes()
                        
                        row = await stmt.fetchrow(
                            embedding_str,
                            setup.direction,
                            setup.profit_pct,
                            setup.leverage,
                            setup.hold_duration,
                            setup.entry_timestamp,
                            setup.entry_price,
                            setup.exit_price,
                            setup.symbol,
                            setup.timeframe,
                            setup.window_size,
                            price_window_bytes
                        )
                        ids.append(row['id'])
                
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(setups) + batch_size - 1)//batch_size}")
        
        return ids
    
    async def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        symbol: Optional[str] = None,
        direction: Optional[int] = None,
        min_profit: Optional[float] = None,
        max_profit: Optional[float] = None,
        min_timestamp: Optional[float] = None,
        max_timestamp: Optional[float] = None
    ) -> List[SimilarSetup]:
        """
        Find k most similar trade setups to query embedding.
        
        Args:
            query_embedding: Query vector (must be L2 normalized)
            k: Number of results to return
            symbol: Filter by symbol (optional)
            direction: Filter by direction (optional)
            min_profit: Minimum profit percentage (optional)
            max_profit: Maximum profit percentage (optional)
            min_timestamp: Minimum entry timestamp (optional)
            max_timestamp: Maximum entry timestamp (optional)
            
        Returns:
            List of SimilarSetup objects, sorted by similarity (descending)
        """
        async with self.pool.acquire() as conn:
            # Build query with optional filters
            embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
            
            filters = []
            params = [embedding_str, k]
            param_idx = 3
            
            if symbol:
                filters.append(f"symbol = ${param_idx}")
                params.append(symbol)
                param_idx += 1
            
            if direction is not None:
                filters.append(f"direction = ${param_idx}")
                params.append(direction)
                param_idx += 1
            
            if min_profit is not None:
                filters.append(f"profit_pct >= ${param_idx}")
                params.append(min_profit)
                param_idx += 1
            
            if max_profit is not None:
                filters.append(f"profit_pct <= ${param_idx}")
                params.append(max_profit)
                param_idx += 1
            
            if min_timestamp is not None:
                filters.append(f"entry_timestamp >= ${param_idx}")
                params.append(min_timestamp)
                param_idx += 1
            
            if max_timestamp is not None:
                filters.append(f"entry_timestamp <= ${param_idx}")
                params.append(max_timestamp)
                param_idx += 1
            
            where_clause = ""
            if filters:
                where_clause = "WHERE " + " AND ".join(filters)
            
            # Use cosine distance (1 - cosine_similarity)
            query = f"""
                SELECT 
                    id, embedding, direction, profit_pct, leverage, hold_duration,
                    entry_timestamp, entry_price, exit_price, symbol, timeframe,
                    window_size, price_window, created_at,
                    1 - (embedding <=> $1::vector) as similarity,
                    embedding <-> $1::vector as l2_distance
                FROM trade_setups
                {where_clause}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """
            
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                # Parse embedding back to numpy
                emb_str = row['embedding']
                if isinstance(emb_str, str):
                    emb = np.array([float(x) for x in emb_str.strip('[]').split(',')])
                else:
                    emb = np.array(emb_str)
                
                # Deserialize price window
                price_window = None
                if row['price_window']:
                    price_window = np.frombuffer(row['price_window'], dtype=np.float32)
                
                setup = StoredTradeSetup(
                    id=row['id'],
                    embedding=emb,
                    direction=row['direction'],
                    profit_pct=row['profit_pct'],
                    leverage=row['leverage'],
                    hold_duration=row['hold_duration'],
                    entry_timestamp=row['entry_timestamp'],
                    entry_price=row['entry_price'],
                    exit_price=row['exit_price'],
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    window_size=row['window_size'],
                    price_window=price_window,
                    created_at=row['created_at']
                )
                
                results.append(SimilarSetup(
                    setup=setup,
                    similarity=row['similarity'],
                    distance=row['l2_distance']
                ))
            
            return results
    
    async def get_setup_by_id(self, setup_id: int) -> Optional[StoredTradeSetup]:
        """Get a specific setup by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM trade_setups WHERE id = $1
            """, setup_id)
            
            if not row:
                return None
            
            emb_str = row['embedding']
            if isinstance(emb_str, str):
                emb = np.array([float(x) for x in emb_str.strip('[]').split(',')])
            else:
                emb = np.array(emb_str)
            
            price_window = None
            if row['price_window']:
                price_window = np.frombuffer(row['price_window'], dtype=np.float32)
            
            return StoredTradeSetup(
                id=row['id'],
                embedding=emb,
                direction=row['direction'],
                profit_pct=row['profit_pct'],
                leverage=row['leverage'],
                hold_duration=row['hold_duration'],
                entry_timestamp=row['entry_timestamp'],
                entry_price=row['entry_price'],
                exit_price=row['exit_price'],
                symbol=row['symbol'],
                timeframe=row['timeframe'],
                window_size=row['window_size'],
                price_window=price_window,
                created_at=row['created_at']
            )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM trade_setups")
            
            symbol_counts = await conn.fetch("""
                SELECT symbol, COUNT(*) as count 
                FROM trade_setups 
                GROUP BY symbol
            """)
            
            direction_counts = await conn.fetch("""
                SELECT direction, COUNT(*) as count 
                FROM trade_setups 
                GROUP BY direction
            """)
            
            profit_stats = await conn.fetchrow("""
                SELECT 
                    AVG(profit_pct) as avg_profit,
                    STDDEV(profit_pct) as std_profit,
                    MIN(profit_pct) as min_profit,
                    MAX(profit_pct) as max_profit,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY profit_pct) as median_profit
                FROM trade_setups
            """)
            
            return {
                'total_setups': count,
                'by_symbol': {row['symbol']: row['count'] for row in symbol_counts},
                'by_direction': {
                    'long': next((r['count'] for r in direction_counts if r['direction'] == 1), 0),
                    'short': next((r['count'] for r in direction_counts if r['direction'] == -1), 0)
                },
                'profit_stats': {
                    'avg': profit_stats['avg_profit'],
                    'std': profit_stats['std_profit'],
                    'min': profit_stats['min_profit'],
                    'max': profit_stats['max_profit'],
                    'median': profit_stats['median_profit']
                } if profit_stats else None
            }
    
    async def delete_all(self):
        """Delete all trade setups (use with caution!)"""
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE trade_setups RESTART IDENTITY")
            logger.warning("Deleted all trade setups from database")


# Synchronous wrapper for non-async contexts
class TradeEmbeddingDBSync:
    """Synchronous wrapper around the async database interface"""
    
    def __init__(self, **kwargs):
        self._db = TradeEmbeddingDB(**kwargs)
        self._loop = asyncio.new_event_loop()
    
    def _run(self, coro):
        return self._loop.run_until_complete(coro)
    
    def connect(self):
        self._run(self._db.connect())
    
    def disconnect(self):
        self._run(self._db.disconnect())
        self._loop.close()
    
    def initialize_schema(self):
        self._run(self._db.initialize_schema())
    
    def insert_setup(self, setup: StoredTradeSetup) -> int:
        return self._run(self._db.insert_setup(setup))
    
    def insert_batch(self, setups: List[StoredTradeSetup], batch_size: int = 100) -> List[int]:
        return self._run(self._db.insert_batch(setups, batch_size))
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10, **kwargs) -> List[SimilarSetup]:
        return self._run(self._db.search_similar(query_embedding, k, **kwargs))
    
    def get_setup_by_id(self, setup_id: int) -> Optional[StoredTradeSetup]:
        return self._run(self._db.get_setup_by_id(setup_id))
    
    def get_stats(self) -> Dict[str, Any]:
        return self._run(self._db.get_stats())
    
    def delete_all(self):
        self._run(self._db.delete_all())
