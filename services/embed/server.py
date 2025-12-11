"""
FastAPI backend for real-time trade setup embedding and similarity search.

Endpoints:
- POST /embed: Embed a price window
- POST /search: Find similar historical setups
- GET /setup/{id}: Get a specific setup
- GET /stats: Database statistics
- WS /ws/live: WebSocket for live price streaming and real-time matching
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from collections import deque

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import logging

from models.encoder import PriceWindowEncoder, normalize_price_window
from models.dp_oracle import DPOracle
from database.pgvector_store import TradeEmbeddingDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class EmbedRequest(BaseModel):
    """Request to embed a price window"""
    prices: List[float] = Field(..., description="Raw prices (will be normalized)")
    
class EmbedResponse(BaseModel):
    """Embedding response"""
    embedding: List[float]
    normalized_prices: List[float]


class SearchRequest(BaseModel):
    """Request to search for similar setups"""
    prices: List[float] = Field(..., description="Price window to match")
    k: int = Field(default=10, ge=1, le=100)
    symbol: Optional[str] = None
    direction: Optional[int] = Field(default=None, ge=-1, le=1)
    min_profit: Optional[float] = None
    max_profit: Optional[float] = None


class SetupResponse(BaseModel):
    """Trade setup response"""
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
    created_at: Optional[str] = None


class SearchResultResponse(BaseModel):
    """Single search result"""
    setup: SetupResponse
    similarity: float
    distance: float


class SearchResponse(BaseModel):
    """Search response"""
    query_embedding: List[float]
    results: List[SearchResultResponse]
    avg_similarity: float
    avg_profit: float
    direction_consensus: Dict[str, float]  # {'long': 0.7, 'short': 0.3}


class OracleSignal(BaseModel):
    """Real-time oracle signal"""
    timestamp: float
    action: str  # 'long', 'short', 'close', 'hold'
    leverage: float
    confidence: float
    similar_setups: List[SearchResultResponse]


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Application state container"""
    
    def __init__(self):
        self.encoder: Optional[PriceWindowEncoder] = None
        self.db: Optional[TradeEmbeddingDB] = None
        self.oracle: Optional[DPOracle] = None
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.window_size: int = 100
        self.embedding_dim: int = 128
        self.config: Dict[str, Any] = {}
        
        # WebSocket connections for live updates
        self.active_connections: List[WebSocket] = []
        
        # Price buffer for live matching
        self.price_buffers: Dict[str, deque] = {}  # symbol -> price buffer


app = FastAPI(
    title="Trade Setup Embedding API",
    description="Real-time trade pattern matching using contrastive learning",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = AppState()


# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize encoder and database on startup"""
    logger.info("Starting up Trade Embedding API...")
    
    # Load config
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            state.config = json.load(f)
    else:
        state.config = {
            'model_path': 'models/trained/encoder.pt',
            'window_size': 100,
            'embedding_dim': 128,
            'db_host': 'localhost',
            'db_port': 5432,
            'db_name': 'trade_embeddings',
            'db_user': 'postgres',
            'db_password': 'postgres'
        }
    
    state.window_size = state.config.get('window_size', 100)
    state.embedding_dim = state.config.get('embedding_dim', 128)
    
    # Initialize encoder
    state.encoder = PriceWindowEncoder(
        window_size=state.window_size - 1,  # -1 for returns
        embedding_dim=state.embedding_dim
    ).to(state.device)
    
    # Load trained weights if available
    model_path = Path(state.config.get('model_path', 'models/trained/encoder.pt'))
    if model_path.exists():
        state.encoder.load_state_dict(
            torch.load(model_path, map_location=state.device)
        )
        logger.info(f"Loaded encoder from {model_path}")
    else:
        logger.warning(f"No trained model at {model_path}, using random weights")
    
    state.encoder.eval()
    
    # Initialize database
    state.db = TradeEmbeddingDB(
        host=state.config.get('db_host', 'localhost'),
        port=state.config.get('db_port', 5432),
        database=state.config.get('db_name', 'trade_embeddings'),
        user=state.config.get('db_user', 'postgres'),
        password=state.config.get('db_password', 'postgres'),
        embedding_dim=state.embedding_dim
    )
    
    try:
        await state.db.connect()
        await state.db.initialize_schema()
        logger.info("Database connected and initialized")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        # Continue without database - some endpoints will fail
    
    # Initialize oracle
    state.oracle = DPOracle(
        max_leverage=20.0,
        transaction_cost=0.001
    )
    
    logger.info("Startup complete!")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    
    if state.db:
        await state.db.disconnect()
    
    # Close all WebSocket connections
    for ws in state.active_connections:
        await ws.close()


# ============================================================================
# REST Endpoints
# ============================================================================

@app.post("/embed", response_model=EmbedResponse)
async def embed_prices(request: EmbedRequest):
    """
    Embed a price window into the learned representation space.
    
    The prices are normalized to log returns before encoding.
    """
    if len(request.prices) < 2:
        raise HTTPException(400, "Need at least 2 prices")
    
    prices = np.array(request.prices, dtype=np.float32)
    
    # Normalize to log returns
    normalized = normalize_price_window(prices)
    
    # Pad/truncate to window size
    target_len = state.window_size - 1
    if len(normalized) < target_len:
        normalized = np.pad(normalized, (target_len - len(normalized), 0), mode='edge')
    elif len(normalized) > target_len:
        normalized = normalized[-target_len:]
    
    # Encode
    with torch.no_grad():
        x = torch.from_numpy(normalized).float().unsqueeze(0).to(state.device)
        embedding = state.encoder(x).cpu().numpy()[0]
    
    return EmbedResponse(
        embedding=embedding.tolist(),
        normalized_prices=normalized.tolist()
    )


@app.post("/search", response_model=SearchResponse)
async def search_similar(request: SearchRequest):
    """
    Find similar historical trade setups.
    
    Returns the k most similar setups along with aggregated statistics.
    """
    if state.db is None or state.db.pool is None:
        raise HTTPException(503, "Database not available")
    
    if len(request.prices) < 2:
        raise HTTPException(400, "Need at least 2 prices")
    
    prices = np.array(request.prices, dtype=np.float32)
    normalized = normalize_price_window(prices)
    
    # Pad/truncate
    target_len = state.window_size - 1
    if len(normalized) < target_len:
        normalized = np.pad(normalized, (target_len - len(normalized), 0), mode='edge')
    elif len(normalized) > target_len:
        normalized = normalized[-target_len:]
    
    # Encode
    with torch.no_grad():
        x = torch.from_numpy(normalized).float().unsqueeze(0).to(state.device)
        query_embedding = state.encoder(x).cpu().numpy()[0]
    
    # Search
    results = await state.db.search_similar(
        query_embedding=query_embedding,
        k=request.k,
        symbol=request.symbol,
        direction=request.direction,
        min_profit=request.min_profit,
        max_profit=request.max_profit
    )
    
    # Convert to response format
    result_responses = []
    for r in results:
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
            price_window=r.setup.price_window.tolist() if r.setup.price_window is not None else None,
            created_at=r.setup.created_at.isoformat() if r.setup.created_at else None
        )
        result_responses.append(SearchResultResponse(
            setup=setup_response,
            similarity=r.similarity,
            distance=r.distance
        ))
    
    # Aggregate statistics
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
    if state.db is None or state.db.pool is None:
        raise HTTPException(503, "Database not available")
    
    setup = await state.db.get_setup_by_id(setup_id)
    
    if setup is None:
        raise HTTPException(404, f"Setup {setup_id} not found")
    
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
        price_window=setup.price_window.tolist() if setup.price_window is not None else None,
        created_at=setup.created_at.isoformat() if setup.created_at else None
    )


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    if state.db is None or state.db.pool is None:
        raise HTTPException(503, "Database not available")
    
    return await state.db.get_stats()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'encoder_loaded': state.encoder is not None,
        'db_connected': state.db is not None and state.db.pool is not None,
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
    WebSocket endpoint for live price streaming and real-time pattern matching.
    
    Client sends price updates, server responds with similar setups when patterns match.
    
    Message format (client -> server):
    {
        "type": "price",
        "price": 50000.0,
        "timestamp": 1234567890.123
    }
    
    Message format (server -> client):
    {
        "type": "match",
        "similarity": 0.85,
        "direction_signal": "long",
        "confidence": 0.75,
        "similar_setups": [...]
    }
    """
    await websocket.accept()
    state.active_connections.append(websocket)
    
    # Initialize price buffer for this symbol
    if symbol not in state.price_buffers:
        state.price_buffers[symbol] = deque(maxlen=state.window_size)
    
    buffer = state.price_buffers[symbol]
    
    try:
        while True:
            # Receive price update
            data = await websocket.receive_json()
            
            if data.get('type') == 'price':
                price = float(data['price'])
                timestamp = float(data.get('timestamp', datetime.now().timestamp()))
                
                buffer.append((price, timestamp))
                
                # Check if we have enough data for matching
                if len(buffer) >= state.window_size:
                    prices = np.array([p[0] for p in buffer])
                    
                    # Normalize and encode
                    normalized = normalize_price_window(prices)
                    
                    with torch.no_grad():
                        x = torch.from_numpy(normalized).float().unsqueeze(0).to(state.device)
                        query_embedding = state.encoder(x).cpu().numpy()[0]
                    
                    # Search for similar setups
                    if state.db and state.db.pool:
                        results = await state.db.search_similar(
                            query_embedding=query_embedding,
                            k=5,
                            symbol=symbol
                        )
                        
                        if results and results[0].similarity > 0.7:  # Threshold for signaling
                            # Calculate consensus
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
                                
                                # Send match signal
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
