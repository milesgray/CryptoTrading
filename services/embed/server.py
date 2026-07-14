"""
FastAPI backend for real-time trade setup embedding and similarity search.

Uses NumPy + FAISS for storage - no database required.

Endpoints:
- POST /embed: Embed a price window
- POST /search: Find similar historical setups
- GET /setup/{id}: Get a specific setup
- GET /stats: Store statistics
- WS /ws/live: WebSocket for live price streaming and real-time matching
"""
import os
import json
import numpy as np
import asyncio
import asyncpg
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from collections import deque

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import logging

from models.encoder import PriceWindowEncoder, normalize_price_window, extract_trade_setups
from pipeline import TradePipeline
from database.numpy_store import NumpyVectorStore
from database.pgvector_store import TradeEmbeddingDB, StoredTradeSetup
from cryptotrading.trade.oracle import LeveragedDPOracle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class EmbedRequest(BaseModel):
    prices: List[float] = Field(..., description="Raw prices (will be normalized)")

class EmbedResponse(BaseModel):
    embedding: List[float]
    normalized_prices: List[float]

class BatchEmbedRequest(BaseModel):
    prices_list: List[List[float]] = Field(..., description="List of raw prices windows to embed")

class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]]

class AddSetupRequest(BaseModel):
    symbol: str
    timeframe: str
    prices: List[float]
    direction: int
    profit_pct: float
    leverage: float
    hold_duration: int
    entry_timestamp: float
    entry_price: float
    exit_price: float

class AddSetupResponse(BaseModel):
    success: bool
    id: int

class SearchRequest(BaseModel):
    prices: List[float] = Field(..., description="Price window to match")
    k: int = Field(default=10, ge=1, le=100)
    symbol: Optional[str] = None
    direction: Optional[int] = Field(default=None, ge=-1, le=1)
    min_profit: Optional[float] = None
    max_profit: Optional[float] = None

class SetupResponse(BaseModel):
    id: int
    direction: int
    profit_pct: float
    leverage: float
    hold_duration: int
    entry_timestamp: float
    entry_price: float
    exit_price: float
    symbol: str
    timeframe: str
    window_size: int
    price_window: Optional[List[float]] = None

class SearchResultResponse(BaseModel):
    setup: SetupResponse
    similarity: float
    distance: float

class SearchResponse(BaseModel):
    query_embedding: List[float]
    results: List[SearchResultResponse]
    avg_similarity: float
    avg_profit: float
    direction_consensus: Dict[str, float]

# ============================================================================
# Application State
# ============================================================================

class AppState:
    def __init__(self):
        self.pipeline: Optional[TradePipeline] = None
        self.encoder: Optional[PriceWindowEncoder] = None
        self.store: Optional[NumpyVectorStore] = None
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.window_size: int = 100
        self.embedding_dim: int = 128
        self.config: Dict[str, Any] = {}
        self.active_connections: List[WebSocket] = []
        self.price_buffers: Dict[str, deque] = {}

    def generate_embedding(self, x: np.ndarray) -> np.ndarray:
        if self.pipeline is not None:
            return self.pipeline.generate_embedding(x)
        
        if self.encoder is None:
            raise ValueError("Neither pipeline nor encoder is initialized")
        
        is_single = x.ndim == 1
        if is_single:
            x_batch = np.expand_dims(x, axis=0)
        else:
            x_batch = x
            
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_batch).float().to(self.device)
            emb = self.encoder(x_tensor).cpu().numpy()
            
        if is_single:
            return emb[0]
        return emb


app = FastAPI(
    title="Trade Setup Embedding API",
    description="Real-time trade pattern matching using contrastive learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = AppState()


def _process_symbol_sync(sym, prices, timestamps, window_size, encoder, device):
    """Synchronous CPU-bound processing to compute oracle actions and extract setups."""
    oracle = LeveragedDPOracle(max_leverage=20.0, transaction_cost=0.001)
    actions, leverages = oracle.compute_oracle_actions(prices)
    setups = extract_trade_setups(
        prices=prices,
        timestamps=timestamps,
        oracle_actions=actions,
        oracle_leverages=leverages,
        window_size=window_size
    )
    if not setups:
        return [], None

    # Generate embeddings
    encoder.eval()
    embeddings = []
    batch_size = 256
    for i in range(0, len(setups), batch_size):
        batch = setups[i:i + batch_size]
        windows = np.stack([s.price_window for s in batch])
        with torch.no_grad():
            x = torch.from_numpy(windows).float().to(device)
            emb = encoder(x).cpu().numpy()
        embeddings.append(emb)

    if embeddings:
        embeddings = np.vstack(embeddings)

    return setups, embeddings


async def auto_populate_db():
    """Background task to wait for bootstrapped price data and automatically populate trade setups vector store."""
    logger.info("Starting auto-population background task...")
    
    # Wait for price_data to be populated by the retrieval service
    retries = 0
    max_retries = 30
    prices_found = False
    
    # Give the connection/schema initialization a moment
    await asyncio.sleep(2)
    
    from cryptotrading.data.postgres import init_pool
    
    model_path = Path(state.config.get('model_path', 'models/trained/encoder.pt'))
    
    # If the trained model is missing, we clear any existing database records to force retraining and regeneration
    if not model_path.exists():
        logger.warning(f"No trained model found at {model_path}. Forcing clearing of vector store for retraining...")
        if isinstance(state.store, TradeEmbeddingDB):
            try:
                await state.store.delete_all()
                logger.info("Cleared existing setups to allow retraining and repopulation.")
            except Exception as e:
                logger.error(f"Error clearing setups: {e}")
    
    while retries < max_retries:
        if state.store is None:
            await asyncio.sleep(2)
            continue
            
        if isinstance(state.store, TradeEmbeddingDB):
            try:
                stats = await state.store.get_stats()
                total_setups = stats['total_setups']
            except Exception as e:
                logger.warning(f"Error getting stats from store: {e}")
                total_setups = 0
        else:
            total_setups = len(state.store.metadata)
            
        if total_setups > 0:
            logger.info(f"Vector store already contains {total_setups} setups. Skipping auto-population.")
            return
            
        # Check if price_data has entries using connection pool
        try:
            pool = state.store.pool if (isinstance(state.store, TradeEmbeddingDB) and state.store.pool is not None) else await init_pool()
            async with pool.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM price_data")
                
            if count and count >= 120:  # Need at least some window of data
                logger.info(f"Found {count} price records in database. Starting embedding extraction...")
                prices_found = True
                break
        except Exception as e:
            logger.warning(f"Error checking price_data count (retry {retries}/{max_retries}): {e}")
            
        retries += 1
        await asyncio.sleep(5)
        
    if not prices_found:
        logger.warning("No price data found in database after waiting. Auto-population aborted.")
        return
        
    # Query price data
    try:
        pool = state.store.pool if (isinstance(state.store, TradeEmbeddingDB) and state.store.pool is not None) else await init_pool()
        
        # Get unique symbols
        async with pool.acquire() as conn:
            symbols_rows = await conn.fetch("SELECT DISTINCT symbol FROM price_data")
        
        symbols = [r['symbol'] for r in symbols_rows]
        if not symbols:
            logger.warning("No symbols found in price data.")
            return
            
        # 1. First, extract setups for all symbols
        all_setups_by_symbol = {}
        for sym in symbols:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT time, close FROM (SELECT time, close FROM price_data WHERE symbol = $1 ORDER BY time DESC LIMIT 100000) AS sub ORDER BY time ASC",
                    sym
                )
                
            if not rows:
                continue
                
            prices_list = []
            timestamps_list = []
            for r in rows:
                prices_list.append(float(r['close']))
                timestamps_list.append(r['time'].timestamp())
                
            prices = np.array(prices_list, dtype=np.float64)
            timestamps = np.array(timestamps_list, dtype=np.float64)
            
            if len(prices) < state.window_size:
                logger.warning(f"Insufficient price data for {sym} to extract setups (need {state.window_size}, got {len(prices)})")
                continue
                
            logger.info(f"Extracting setups for {sym} ({len(prices)} prices)...")
            
            # Use DP oracle to get optimal setups
            oracle = LeveragedDPOracle(max_leverage=20.0, transaction_cost=0.001)
            actions, leverages = await asyncio.to_thread(oracle.compute_oracle_actions, prices)
            setups = extract_trade_setups(
                prices=prices,
                timestamps=timestamps,
                oracle_actions=actions,
                oracle_leverages=leverages,
                window_size=state.window_size
            )
            
            if setups:
                all_setups_by_symbol[sym] = (setups, prices, timestamps)
                logger.info(f"Extracted {len(setups)} setups for {sym}")
                
        # Combine all setups for training
        train_setups = []
        for sym, (setups, _, _) in all_setups_by_symbol.items():
            train_setups.extend(setups)
            
        if not train_setups:
            logger.warning("No setups found across any symbols. Auto-population aborted.")
            return
            
        # 2. Train the encoder if weights file is missing
        if not model_path.exists():
            logger.info(f"Training contrastive encoder on {len(train_setups)} setups...")
            from models.trainer import EncoderTrainer
            
            trainer = EncoderTrainer(
                window_size=state.window_size,
                embedding_dim=state.embedding_dim,
                device=state.device
            )
            
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Split train/val
            np.random.shuffle(train_setups)
            split_idx = int(len(train_setups) * 0.9)
            t_setups = train_setups[:split_idx]
            v_setups = train_setups[split_idx:]
            
            # Ensure we have at least some validation data
            if not v_setups:
                v_setups = t_setups
                
            # Train for 15 epochs on CPU/GPU
            await asyncio.to_thread(
                trainer.train,
                train_setups=t_setups,
                val_setups=v_setups,
                num_epochs=15,
                save_path=model_path.parent
            )
            
            # Load trained weights
            state.encoder = trainer.encoder
            state.encoder.eval()
            state.pipeline.encoder = state.encoder
            logger.info(f"Saved trained encoder weights to {model_path}")
            
        # 3. Generate embeddings and insert into vector store
        total_inserted = 0
        device = next(state.encoder.parameters()).device
        
        for sym, (setups, prices, timestamps) in all_setups_by_symbol.items():
            logger.info(f"Generating embeddings and inserting setups for {sym}...")
            
            # Generate embeddings in batches
            embeddings = []
            batch_size = 256
            for i in range(0, len(setups), batch_size):
                batch = setups[i:i + batch_size]
                windows = np.stack([s.price_window for s in batch])
                emb = state.generate_embedding(windows)
                embeddings.append(emb)
                
            if embeddings:
                embeddings = np.vstack(embeddings)
            else:
                continue
                
            # Convert to StoredTradeSetup objects
            stored_setups = []
            for idx, setup in enumerate(setups):
                start_idx = max(0, setup.entry_idx - state.window_size)
                end_idx = setup.entry_idx
                raw_prices = prices[start_idx:end_idx]
                entry_price = float(prices[setup.entry_idx])
                exit_idx = min(setup.entry_idx + setup.hold_duration, len(prices) - 1)
                exit_price = float(prices[exit_idx])
                
                target_len = state.window_size
                if len(raw_prices) < target_len:
                    raw_prices = np.pad(raw_prices, (target_len - len(raw_prices), 0), mode='edge')
                elif len(raw_prices) > target_len:
                    raw_prices = raw_prices[-target_len:]
                    
                stored = StoredTradeSetup(
                    id=None,
                    embedding=embeddings[idx],
                    direction=setup.direction,
                    profit_pct=setup.profit_pct,
                    leverage=setup.leverage,
                    hold_duration=setup.hold_duration,
                    entry_timestamp=float(timestamps[setup.entry_idx]),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    symbol=sym,
                    timeframe='1m',
                    window_size=state.window_size,
                    price_window=raw_prices
                )
                stored_setups.append(stored)
                
            # Insert into store
            if isinstance(state.store, TradeEmbeddingDB):
                ids = await state.store.insert_batch(stored_setups)
                total_inserted += len(ids)
            else:
                price_windows_padded = np.array([s.price_window for s in stored_setups])
                from database.numpy_store import StoredTradeSetup as NumpyStoredTradeSetup
                numpy_stored_setups = []
                for s in stored_setups:
                    n_s = NumpyStoredTradeSetup(
                        id=0,
                        direction=s.direction,
                        profit_pct=s.profit_pct,
                        leverage=s.leverage,
                        hold_duration=s.hold_duration,
                        entry_timestamp=s.entry_timestamp,
                        entry_price=s.entry_price,
                        exit_price=s.exit_price,
                        symbol=s.symbol,
                        timeframe=s.timeframe,
                        window_size=s.window_size
                    )
                    numpy_stored_setups.append(n_s)
                ids = state.store.add_batch(embeddings, numpy_stored_setups, price_windows_padded)
                state.store.save()
                total_inserted += len(ids)
                
            await asyncio.sleep(0.05)
            
        logger.info(f"Auto-population complete. Extracted: {len(train_setups)}, Inserted/Saved: {total_inserted}")
    except Exception as e:
        logger.error(f"Error during auto-population background task: {e}", exc_info=True)


# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup():
    logger.info("Starting up Trade Embedding API...")
    
    # Load config
    config_path = Path("config.json")
    if not config_path.exists():
        with open(config_path, 'w') as f:
            json.dump({
                'model_path': 'models/trained/encoder.pt',
                'store_path': 'vector_store',
                'window_size': 100,
                'embedding_dim': 128,
                'use_chronos': True,
            }, f)
    
    with open(config_path) as f:
        state.config = json.load(f)

    state.window_size = state.config.get('window_size', 100)
    cnn_dim = state.config.get('embedding_dim', 128)
    state.embedding_dim = cnn_dim
    
    # Initialize encoder
    state.pipeline = TradePipeline(
        window_size=state.window_size,
        embedding_dim=cnn_dim,
        device=state.device,
        chronos_model_id=state.config.get('chronos_model_id', 'amazon/chronos-t5-base'),
        chronos_torch_dtype=state.config.get('chronos_torch_dtype', 'bfloat16')
    )
    
    # Load encoder weights
    model_path = Path(state.config.get('model_path', 'models/trained/encoder.pt'))
    state.pipeline.initialize_encoder(model_path)
    state.encoder = state.pipeline.encoder
    
    # Initialize Chronos if requested in config
    if state.config.get('use_chronos', False):
        logger.info("Initializing Chronos model...")
        state.pipeline.initialize_chronos()
        
        # Resolve Chronos dimension dynamically
        chronos_model = state.pipeline.chronos.model.model
        if hasattr(chronos_model.config, "d_model"):
            chronos_dim = chronos_model.config.d_model
        elif hasattr(chronos_model.config, "hidden_size"):
            chronos_dim = chronos_model.config.hidden_size
        else:
            chronos_dim = 768
            
        state.embedding_dim = cnn_dim + chronos_dim
        logger.info(f"Chronos enabled. Adjusted embedding dimension: {state.embedding_dim} ({cnn_dim} CNN + {chronos_dim} Chronos)")

    # Initialize vector store
    db_backend = os.getenv("DB_BACKEND", "numpy").lower()
    if db_backend == "postgres":
        postgres_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/crypto_trading")
        logger.info(f"Initializing PostgreSQL/pgvector store with URI: {postgres_uri}")
        state.store = TradeEmbeddingDB(
            dsn=postgres_uri,
            embedding_dim=state.embedding_dim
        )
        await state.store.connect()
        await state.store.initialize_schema()
        stats = await state.store.get_stats()
    else:
        store_path = state.config.get('store_path', 'vector_store')
        state.store = NumpyVectorStore(
            store_path=store_path,
            embedding_dim=state.embedding_dim
        )
        stats = state.store.get_stats()
    
    logger.info(f"Vector store loaded: {stats}")
    
    # Start auto-population task in the background
    asyncio.create_task(auto_populate_db())
    
    logger.info("Startup complete!")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down...")
    
    if state.store:
        if isinstance(state.store, TradeEmbeddingDB):
            await state.store.disconnect()
        else:
            state.store.save()
    
    for ws in state.active_connections:
        await ws.close()


# ============================================================================
# REST Endpoints
# ============================================================================

@app.post("/embed", response_model=EmbedResponse)
async def embed_prices(request: EmbedRequest):
    """Embed a price window into the learned representation space."""
    if len(request.prices) < 2:
        raise HTTPException(400, "Need at least 2 prices")
    
    prices = np.array(request.prices, dtype=np.float32)
    normalized = normalize_price_window(prices)
    
    target_len = state.window_size - 1
    if len(normalized) < target_len:
        normalized = np.pad(normalized, (target_len - len(normalized), 0), mode='edge')
    elif len(normalized) > target_len:
        normalized = normalized[-target_len:]
    
    embedding = state.generate_embedding(normalized)
    
    return EmbedResponse(
        embedding=embedding.tolist(),
        normalized_prices=normalized.tolist()
    )


@app.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_prices_batch(request: BatchEmbedRequest):
    """Embed multiple price windows into representation space."""
    if not request.prices_list:
        return BatchEmbedResponse(embeddings=[])

    target_len = state.window_size - 1
    normalized_list = []
    
    for prices in request.prices_list:
        if len(prices) < 2:
            raise HTTPException(400, "Each price window must have at least 2 prices")
        
        p_arr = np.array(prices, dtype=np.float32)
        norm = normalize_price_window(p_arr)
        
        if len(norm) < target_len:
            norm = np.pad(norm, (target_len - len(norm), 0), mode='edge')
        elif len(norm) > target_len:
            norm = norm[-target_len:]
        
        normalized_list.append(norm)

    x_batch = np.stack(normalized_list)
    
    embeddings = []
    batch_size = 256
    device = state.device
    
    for i in range(0, len(x_batch), batch_size):
        sub_batch = x_batch[i:i + batch_size]
        embedding = state.generate_embedding(sub_batch)
        embeddings.append(embedding)

    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.empty((0, state.embedding_dim))

    return BatchEmbedResponse(embeddings=embeddings.tolist())


@app.post("/setup/add", response_model=AddSetupResponse)
async def add_setup(request: AddSetupRequest):
    """Insert a new trade setup into the database dynamically."""
    if state.store is None:
        raise HTTPException(503, "Store not available")
    
    if len(request.prices) < 2:
        raise HTTPException(400, "Need at least 2 prices to generate embedding")
        
    # 1. Generate embedding for the historical price window
    prices = np.array(request.prices, dtype=np.float32)
    normalized = normalize_price_window(prices)
    
    target_len = state.window_size - 1
    if len(normalized) < target_len:
        normalized = np.pad(normalized, (target_len - len(normalized), 0), mode='edge')
    elif len(normalized) > target_len:
        normalized = normalized[-target_len:]
        
    embedding = state.generate_embedding(normalized)
        
    # 2. Re-pad/crop original prices to window_size for storage
    raw_prices = prices
    target_store_len = state.window_size
    if len(raw_prices) < target_store_len:
        raw_prices = np.pad(raw_prices, (target_store_len - len(raw_prices), 0), mode='edge')
    elif len(raw_prices) > target_store_len:
        raw_prices = raw_prices[-target_store_len:]
        
    # 3. Create StoredTradeSetup
    stored = StoredTradeSetup(
        id=None,
        embedding=embedding,
        direction=request.direction,
        profit_pct=request.profit_pct,
        leverage=request.leverage,
        hold_duration=request.hold_duration,
        entry_timestamp=request.entry_timestamp,
        entry_price=request.entry_price,
        exit_price=request.exit_price,
        symbol=request.symbol,
        timeframe=request.timeframe,
        window_size=state.window_size,
        price_window=raw_prices
    )
    
    # 4. Insert into store
    if isinstance(state.store, TradeEmbeddingDB):
        setup_id = await state.store.insert_setup(stored)
    else:
        # NumpyVectorStore implementation: StoredTradeSetup from database.numpy_store
        from database.numpy_store import StoredTradeSetup as NumpyStoredTradeSetup
        numpy_stored = NumpyStoredTradeSetup(
            id=0,
            direction=stored.direction,
            profit_pct=stored.profit_pct,
            leverage=stored.leverage,
            hold_duration=stored.hold_duration,
            entry_timestamp=stored.entry_timestamp,
            entry_price=stored.entry_price,
            exit_price=stored.exit_price,
            symbol=stored.symbol,
            timeframe=stored.timeframe,
            window_size=stored.window_size
        )
        setup_id = state.store.add(embedding, numpy_stored, raw_prices)
        state.store.save()
        
    return AddSetupResponse(success=True, id=setup_id)


@app.post("/search", response_model=SearchResponse)
async def search_similar(request: SearchRequest):
    """Find similar historical trade setups."""
    if state.store is None:
        raise HTTPException(503, "Store not available")
    
    if len(request.prices) < 2:
        raise HTTPException(400, "Need at least 2 prices")
    
    prices = np.array(request.prices, dtype=np.float32)
    normalized = normalize_price_window(prices)
    
    target_len = state.window_size - 1
    if len(normalized) < target_len:
        normalized = np.pad(normalized, (target_len - len(normalized), 0), mode='edge')
    elif len(normalized) > target_len:
        normalized = normalized[-target_len:]
    
    query_embedding = state.generate_embedding(normalized)
    
    if isinstance(state.store, TradeEmbeddingDB):
        results = await state.store.search_similar(
            query_embedding=query_embedding,
            k=request.k,
            symbol=request.symbol,
            direction=request.direction,
            min_profit=request.min_profit,
            max_profit=request.max_profit
        )
    else:
        results = state.store.search(
            query_embedding=query_embedding,
            k=request.k,
            symbol=request.symbol,
            direction=request.direction,
            min_profit=request.min_profit,
            max_profit=request.max_profit
        )
    
    result_responses = []
    for r in results:
        if isinstance(state.store, TradeEmbeddingDB):
            price_window = r.setup.price_window
        else:
            price_window = state.store.get_price_window(r.setup.id)
        
        setup_response = SetupResponse(
            id=r.setup.id,
            direction=r.setup.direction,
            profit_pct=r.setup.profit_pct,
            leverage=r.setup.leverage,
            hold_duration=r.setup.hold_duration,
            entry_timestamp=r.setup.entry_timestamp,
            entry_price=r.setup.entry_price,
            exit_price=r.setup.exit_price,
            symbol=r.setup.symbol,
            timeframe=r.setup.timeframe,
            window_size=r.setup.window_size,
            price_window=price_window.tolist() if price_window is not None else None
        )
        result_responses.append(SearchResultResponse(
            setup=setup_response,
            similarity=r.similarity,
            distance=r.distance
        ))
    
    if results:
        avg_similarity = np.mean([r.similarity for r in results])
        avg_profit = np.mean([r.setup.profit_pct for r in results])
        
        long_count = sum(1 for r in results if r.setup.direction == 1)
        short_count = sum(1 for r in results if r.setup.direction == -1)
        total = long_count + short_count
        
        direction_consensus = {
            'long': long_count / total if total > 0 else 0,
            'short': short_count / total if total > 0 else 0
        }
    else:
        avg_similarity = 0.0
        avg_profit = 0.0
        direction_consensus = {'long': 0, 'short': 0}
    
    return SearchResponse(
        query_embedding=query_embedding.tolist(),
        results=result_responses,
        avg_similarity=avg_similarity,
        avg_profit=avg_profit,
        direction_consensus=direction_consensus
    )


@app.get("/setup/{setup_id}", response_model=SetupResponse)
async def get_setup(setup_id: int):
    """Get a specific trade setup by ID"""
    if state.store is None:
        raise HTTPException(503, "Store not available")
    
    if isinstance(state.store, TradeEmbeddingDB):
        setup = await state.store.get_setup_by_id(setup_id)
        if setup is None:
            raise HTTPException(404, f"Setup {setup_id} not found")
        price_window = setup.price_window
    else:
        result = state.store.get_by_id(setup_id)
        if result is None:
            raise HTTPException(404, f"Setup {setup_id} not found")
        setup, embedding, price_window = result
    
    return SetupResponse(
        id=setup.id,
        direction=setup.direction,
        profit_pct=setup.profit_pct,
        leverage=setup.leverage,
        hold_duration=setup.hold_duration,
        entry_timestamp=setup.entry_timestamp,
        entry_price=setup.entry_price,
        exit_price=setup.exit_price,
        symbol=setup.symbol,
        timeframe=setup.timeframe,
        window_size=setup.window_size,
        price_window=price_window.tolist() if price_window is not None else None
    )


@app.get("/stats")
async def get_stats():
    """Get store statistics"""
    if state.store is None:
        raise HTTPException(503, "Store not available")
    
    if isinstance(state.store, TradeEmbeddingDB):
        return await state.store.get_stats()
    return state.store.get_stats()


@app.get("/health")
async def health_check():
    if state.store:
        if isinstance(state.store, TradeEmbeddingDB):
            stats = await state.store.get_stats()
            num_setups = stats['total_setups']
        else:
            num_setups = len(state.store.metadata)
    else:
        num_setups = 0

    return {
        'status': 'healthy',
        'encoder_loaded': state.encoder is not None,
        'store_loaded': state.store is not None,
        'num_setups': num_setups,
        'device': state.device,
        'window_size': state.window_size,
        'embedding_dim': state.embedding_dim
    }


# ============================================================================
# WebSocket for Live Streaming
# ============================================================================

@app.websocket("/ws/live/{symbol}")
async def websocket_live(websocket: WebSocket, symbol: str):
    """
    WebSocket for live price streaming and real-time pattern matching.
    
    Client sends: {"type": "price", "price": 50000.0, "timestamp": 1234567890.123}
    Server sends: {"type": "match", "similarity": 0.85, "direction_signal": "long", ...}
    """
    await websocket.accept()
    state.active_connections.append(websocket)
    
    if symbol not in state.price_buffers:
        state.price_buffers[symbol] = deque(maxlen=state.window_size)
    
    buffer = state.price_buffers[symbol]
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'price':
                price = float(data['price'])
                timestamp = float(data.get('timestamp', datetime.now().timestamp()))
                
                buffer.append((price, timestamp))
                
                if len(buffer) >= state.window_size:
                    prices = np.array([p[0] for p in buffer])
                    normalized = normalize_price_window(prices)
                    
                    query_embedding = state.generate_embedding(normalized)
                    
                    if state.store:
                        if isinstance(state.store, TradeEmbeddingDB):
                            results = await state.store.search_similar(
                                query_embedding=query_embedding,
                                k=5,
                                symbol=symbol
                            )
                        else:
                            if len(state.store.metadata) > 0:
                                results = state.store.search(
                                    query_embedding=query_embedding,
                                    k=5,
                                    symbol=symbol
                                )
                            else:
                                results = []
                        
                        if results and results[0].similarity > 0.7:
                            long_weight = sum(
                                r.similarity * (1 if r.setup.direction == 1 else 0)
                                for r in results
                            )
                            short_weight = sum(
                                r.similarity * (1 if r.setup.direction == -1 else 0)
                                for r in results
                            )
                            
                            total_weight = long_weight + short_weight
                            
                            if total_weight > 0:
                                direction_signal = 'long' if long_weight > short_weight else 'short'
                                confidence = max(long_weight, short_weight) / total_weight
                                
                                await websocket.send_json({
                                    'type': 'match',
                                    'timestamp': timestamp,
                                    'similarity': float(results[0].similarity),
                                    'direction_signal': direction_signal,
                                    'confidence': float(confidence),
                                    'avg_profit': float(np.mean([r.setup.profit_pct for r in results])),
                                    'similar_setups': [
                                        {
                                            'id': r.setup.id,
                                            'direction': r.setup.direction,
                                            'profit_pct': r.setup.profit_pct,
                                            'similarity': r.similarity,
                                            'entry_price': r.setup.entry_price,
                                            'exit_price': r.setup.exit_price
                                        }
                                        for r in results[:3]
                                    ]
                                })
            
            elif data.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        state.active_connections.remove(websocket)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8301))
    uvicorn.run(app, host="0.0.0.0", port=port)
