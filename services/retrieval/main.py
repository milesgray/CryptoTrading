from fastapi import FastAPI
from encoder import RetrievalServiceEncoder
from forecaster import RetrievalForecaster
from data_loader import RollbitDataLoader
import numpy as np
from typing import Dict, Any
import uvicorn

app = FastAPI()

# Initialize encoder and forecaster
encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=56)
forecaster = RetrievalForecaster(encoder_service)

# Load historical data and build index
data_loader = RollbitDataLoader(symbol="BTC")
historical_data = data_loader.fetch_historical_data(limit=1000)

for segment in historical_data:
    encoder_service.add_segment(
        prices=np.array(segment["prices"]),
        order_book=segment["order_book"],
        metadata={"id": segment.get("id", "unknown")}
    )

encoder_service.build_index(n_trees=10)

@app.get("/forecast")
async def forecast(symbol: str = "BTC", k: int = 5) -> Dict[str, Any]:
    """Forecast endpoint."""
    # Fetch real data from Rollbit
    latest_data = data_loader.fetch_latest_data()
    
    return forecaster.forecast(
        prices=np.array(latest_data["prices"]),
        order_book=latest_data["order_book"],
        k=k
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)