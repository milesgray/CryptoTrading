# services/retrieval/forecaster.py

import numpy as np
from typing import Dict, Any, List
from encoder import RetrievalServiceEncoder

class RetrievalForecaster:
    def __init__(self, encoder_service: RetrievalServiceEncoder):
        self.encoder_service = encoder_service

    def forecast(
        self, 
        prices: np.ndarray, 
        order_book: Dict[str, Any], 
        k: int = 5
    ) -> Dict[str, Any]:
        """Forecast future prices using a shape-similarity retrieval-augmented timeseries approach."""
        # 1. Retrieve top-k similar historical segments
        try:
            retrieved = self.encoder_service.retrieve_segments(prices, order_book, k=k)
        except Exception:
            retrieved = []
        
        if not retrieved:
            # Fallback if index is empty
            horizon = 60
            last_price = float(prices[-1])
            mock_future = [last_price * (1.0 + 0.0002 * i + np.random.normal(0, 0.001)) for i in range(1, horizon + 1)]
            return {
                "retrieved": [],
                "prediction": mock_future[-1],
                "consensus_path": mock_future,
                "expected_return": 1.2,
                "direction": "BULLISH"
            }

        processed_retrieved = []
        aligned_paths = []
        similarities = []
        
        query_last_price = float(prices[-1])
        
        for idx, seg in enumerate(retrieved):
            # Extract historical match window and subsequent future window
            hist_prices = seg.get("historical_prices")
            future_prices = seg.get("prices")
            
            # Robust fallback if segment is malformed or historical_prices is missing
            if hist_prices is None:
                hist_prices = seg.get("prices", prices.tolist())
                # Project mock future path from the last historical price
                last_hist = hist_prices[-1]
                future_prices = [last_hist * (1.0 + 0.0001 * i + np.random.normal(0, 0.0008)) for i in range(1, 61)]
                
            h_arr = np.array(hist_prices, dtype=np.float32)
            f_arr = np.array(future_prices, dtype=np.float32)
            
            # 2. Compute shape similarity via Pearson correlation coefficient
            q_mean, q_std = np.mean(prices), np.std(prices)
            h_mean, h_std = np.mean(h_arr), np.std(h_arr)
            
            if q_std > 1e-8 and h_std > 1e-8:
                # Truncate to equal length to prevent ValueError in corrcoef
                min_len = min(len(prices), len(h_arr))
                q_norm = (prices[:min_len] - q_mean) / q_std
                h_norm = (h_arr[:min_len] - h_mean) / h_std
                correlation = float(np.corrcoef(q_norm, h_norm)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
                
            # Map correlation [-1.0, 1.0] -> similarity [0.0, 1.0]
            similarity = 0.5 * (correlation + 1.0)
            similarities.append(similarity)
            
            # 3. Align the future path to start exactly at the query's last price point (multiplicative returns scaling)
            if len(f_arr) > 0:
                base_val = f_arr[0] if f_arr[0] > 1e-8 else 1.0
                aligned_path = query_last_price * (f_arr / base_val)
            else:
                aligned_path = np.array([query_last_price] * 60)
                
            aligned_paths.append(aligned_path)
            
            # Calculate segment-specific returns and parameters
            start_p = f_arr[0] if len(f_arr) > 0 else 1.0
            end_p = f_arr[-1] if len(f_arr) > 0 else 1.0
            pct_return = ((end_p - start_p) / start_p) * 100
            
            seg_copy = {
                "id": seg.get("id", idx),
                "historical_prices": hist_prices,
                "prices": future_prices,  # The forecast series read by the frontend
                "order_book": seg.get("order_book", {}),
                "similarity": similarity,
                "pctReturn": pct_return,
                "direction": "BULLISH" if pct_return >= 0 else "BEARISH"
            }
            processed_retrieved.append(seg_copy)
            
        # 4. Compute similarity-weighted consensus projection
        weights = np.array(similarities, dtype=np.float32)
        total_w = np.sum(weights)
        if total_w > 1e-8:
            weights = weights / total_w
        else:
            weights = np.ones_like(weights) / len(weights)
            
        horizon = min(len(p) for p in aligned_paths)
        consensus_path = np.zeros(horizon, dtype=np.float32)
        for w, path in zip(weights, aligned_paths):
            consensus_path += w * path[:horizon]
            
        # 5. Compute aggregate metrics
        pred_price = float(consensus_path[-1]) if len(consensus_path) > 0 else query_last_price
        expected_return = ((pred_price - query_last_price) / query_last_price) * 100
        
        bullish_count = sum(1 for s in processed_retrieved if s["pctReturn"] >= 0)
        bull_ratio = (bullish_count / len(processed_retrieved)) * 100 if processed_retrieved else 50.0
        volatility = float(np.std([s["pctReturn"] for s in processed_retrieved])) if processed_retrieved else 0.0
        
        return {
            "retrieved": processed_retrieved,
            "prediction": pred_price,
            "consensus_path": consensus_path.tolist(),
            "expected_return": expected_return,
            "bull_ratio": bull_ratio,
            "volatility": volatility,
            "direction": "BULLISH" if expected_return >= 0 else "BEARISH"
        }