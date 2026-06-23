# services/retrieval/main.py

from fastapi import FastAPI
from encoder import RetrievalServiceEncoder
from forecaster import RetrievalForecaster
import numpy as np
from typing import Dict, Any
import uvicorn

app = FastAPI()

# Initialize encoder and forecaster
encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=56)
forecaster = RetrievalForecaster(encoder_service)

# Load historical data and build index
# TODO: Replace with real data loading
for i in range(100):
    prices = np.random.rand(60) + i * 0.1
    order_book = {"bids": [[100 + i, 10]], "asks": [[101 + i, 10]]}
    encoder_service.add_segment(prices, order_book, {
        "id": i,
        "prices": prices.tolist(),
        "order_book": order_book
    })

encoder_service.build_index(n_trees=10)

@app.get("/forecast")
async def forecast(symbol: str = "BTC", k: int = 5) -> Dict[str, Any]:
    """Forecast endpoint."""
    # TODO: Fetch real data from Rollbit
    current_prices = np.random.rand(60) + 50 * 0.1
    current_order_book = {"bids": [[150, 10]], "asks": [[151, 10]]}
    
    return forecaster.forecast(current_prices, current_order_book, k=k)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)