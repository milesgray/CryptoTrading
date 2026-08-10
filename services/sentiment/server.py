import os
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import analyzer
try:
    from cryptotrading.sentiment.analyzer import CryptoSentimentAnalyzer
except ImportError:
    CryptoSentimentAnalyzer = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sentiment_service")

from cryptotrading.service import ServiceServer

service = ServiceServer(title="Sentiment Service", description="Crypto Sentiment Analysis FastAPI Microservice")
app = service.app

analyzer = None

class SentimentTextRequest(BaseModel):
    text: str

@service.on_startup
async def startup_event():
    global analyzer
    if CryptoSentimentAnalyzer is not None:
        try:
            analyzer = CryptoSentimentAnalyzer()
            logger.info("CryptoSentimentAnalyzer initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize CryptoSentimentAnalyzer: {e}")

@app.post("/analyze")
async def analyze_sentiment(req: SentimentTextRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    if analyzer is not None:
        try:
            scores = analyzer.analyze_text(req.text)
            return {"text": req.text, "scores": scores}
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            
    # Basic fallback VADER heuristic
    positive_words = {"bull", "bullish", "moon", "pump", "up", "buy", "gain"}
    negative_words = {"bear", "bearish", "dump", "crash", "down", "sell", "loss", "fud"}
    
    tokens = req.text.lower().split()
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)
    compound = (pos - neg) / (max(pos + neg, 1))
    
    return {
        "text": req.text,
        "scores": {
            "compound": round(compound, 4),
            "pos": pos,
            "neg": neg
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
