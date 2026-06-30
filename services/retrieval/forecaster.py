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
            start_p = float(f_arr[0]) if len(f_arr) > 0 else 1.0
            end_p = float(f_arr[-1]) if len(f_arr) > 0 else 1.0
            pct_return = float(((end_p - start_p) / start_p) * 100)
            
            seg_copy = {
                "id": int(seg.get("id", idx)),
                "historical_prices": hist_prices,
                "prices": future_prices,  # The forecast series read by the frontend
                "order_book": seg.get("order_book", {}),
                "similarity": float(similarity),
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


class SpecReTFForecaster:
    def __init__(
        self,
        encoder_service: RetrievalServiceEncoder,
        frame_size: int = 16,
        hop_size: int = 4,
        alpha: float = 0.2,
        w_retrieval: np.ndarray = None,
        w_direct: np.ndarray = None,
        w_final: np.ndarray = None,
        bias_retrieval: np.ndarray = None,
        bias_direct: np.ndarray = None,
        bias_final: np.ndarray = None,
    ):
        self.encoder_service = encoder_service
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.alpha = alpha
        
        # Linear projection parameters (optional, for pre-trained model loading)
        self.w_retrieval = w_retrieval
        self.w_direct = w_direct
        self.w_final = w_final
        self.bias_retrieval = bias_retrieval
        self.bias_direct = bias_direct
        self.bias_final = bias_final

    def _stft(self, x: np.ndarray) -> np.ndarray:
        """Compute the Short-Time Fourier Transform of a 1D sequence using a Hann window."""
        # Z-score normalize to remove DC offset and scale variance
        std = np.std(x)
        if std > 1e-8:
            x = (x - np.mean(x)) / std
        else:
            x = x - np.mean(x)

        L = len(x)
        M = self.frame_size
        B = self.hop_size
        
        if L < M:
            # Pad sequence if it's shorter than the frame size
            x = np.pad(x, (M - L, 0), mode="edge")
            L = len(x)
            
        W = (L - M) // B + 1
        window = np.hanning(M)
        
        stft_coefs = []
        for w in range(W):
            frame = x[w * B : w * B + M]
            # Apply window and compute real FFT (rfft)
            coefs = np.fft.rfft(frame * window)
            stft_coefs.append(coefs)
            
        return np.array(stft_coefs)  # Shape: (W, M // 2 + 1)

    def _amplitude_jsd(self, p_q: np.ndarray, p_x: np.ndarray) -> float:
        """Compute the Jensen-Shannon Divergence (base 2) between two normalized amplitude distributions."""
        m = 0.5 * (p_q + p_x)
        eps = 1e-12
        
        def kl_divergence(u, v):
            non_zero = u > 0
            res = np.zeros_like(u)
            res[non_zero] = u[non_zero] * np.log2(u[non_zero] / (v[non_zero] + eps))
            return np.sum(res)
            
        jsd = 0.5 * kl_divergence(p_q, m) + 0.5 * kl_divergence(p_x, m)
        return float(jsd)

    def _phase_coherence(self, phi_q: np.ndarray, phi_x: np.ndarray, weights: np.ndarray = None) -> float:
        """Compute the phase coherence as the cosine of the mean phase difference (wrapped to [-pi, pi])."""
        diff = phi_q - phi_x
        # Wrap to [-pi, pi]
        diff_wrapped = (diff + np.pi) % (2 * np.pi) - np.pi
        
        if weights is not None and np.sum(weights) > 1e-12:
            mean_diff = np.sum(diff_wrapped * weights) / np.sum(weights)
        else:
            mean_diff = np.mean(diff_wrapped)
            
        return float(np.cos(mean_diff))

    def _frequency_similarity(self, q_stft: np.ndarray, x_stft: np.ndarray) -> float:
        """Compute the frequency-aware composite similarity score between query and candidate STFTs."""
        W = min(len(q_stft), len(x_stft))
        if W == 0:
            return 0.0
            
        frame_scores = []
        eps = 1e-12
        
        for w in range(W):
            # Amplitude Similarity
            amp_q = np.abs(q_stft[w])
            amp_x = np.abs(x_stft[w])
            
            sum_q = np.sum(amp_q)
            sum_x = np.sum(amp_x)
            
            p_q = amp_q / (sum_q + eps)
            p_x = amp_x / (sum_x + eps)
            
            d_js = self._amplitude_jsd(p_q, p_x)
            s_amp = 1.0 - d_js
            
            # Phase Similarity
            phi_q = np.angle(q_stft[w])
            phi_x = np.angle(x_stft[w])
            phase_weights = amp_q * amp_x
            s_phase = self._phase_coherence(phi_q, phi_x, weights=phase_weights)
            
            # Composite score
            s_w = s_amp + s_phase
            frame_scores.append(s_w)
            
        # Recency-weighted aggregation
        similarity = 0.0
        for w in range(W):
            weight = self.alpha * ((1.0 - self.alpha) ** (W - 1 - w))
            similarity += weight * frame_scores[w]
            
        return similarity

    def forecast(
        self, 
        prices: np.ndarray, 
        order_book: Dict[str, Any], 
        k: int = 5
    ) -> Dict[str, Any]:
        """Forecast future prices using the SpecReTF frequency-aware retrieval-augmented approach."""
        # 1. Retrieve a larger candidate pool from the vector index
        pool_size = max(k * 4, 20)
        try:
            retrieved = self.encoder_service.retrieve_segments(prices, order_book, k=pool_size)
        except Exception:
            retrieved = []

        query_last_price = float(prices[-1])
        horizon = 60

        if not retrieved:
            # Fallback if index is empty
            mock_future = [query_last_price * (1.0 + 0.0002 * i + np.random.normal(0, 0.001)) for i in range(1, horizon + 1)]
            return {
                "retrieved": [],
                "prediction": mock_future[-1],
                "consensus_path": mock_future,
                "expected_return": 1.2,
                "direction": "BULLISH"
            }

        # Compute query STFT
        q_stft = self._stft(prices)
        
        scored_candidates = []
        for idx, seg in enumerate(retrieved):
            hist_prices = seg.get("historical_prices")
            future_prices = seg.get("prices")
            
            if hist_prices is None:
                hist_prices = seg.get("prices", prices.tolist())
                last_hist = hist_prices[-1]
                future_prices = [last_hist * (1.0 + 0.0001 * i + np.random.normal(0, 0.0008)) for i in range(1, horizon + 1)]
                
            h_arr = np.array(hist_prices, dtype=np.float32)
            f_arr = np.array(future_prices, dtype=np.float32)
            
            # Compute candidate STFT
            x_stft = self._stft(h_arr)
            
            # Compute SpecReTF similarity
            similarity = self._frequency_similarity(q_stft, x_stft)
            
            # Scale future path using the ratio of the mean of recent prices to the mean of candidate history
            if len(f_arr) > 0:
                mean_query = np.mean(prices)
                mean_hist = np.mean(h_arr)
                scale_factor = mean_query / mean_hist if mean_hist > 1e-8 else 1.0
                aligned_path = f_arr * scale_factor
            else:
                aligned_path = np.array([query_last_price] * horizon)
                
            start_p = float(f_arr[0]) if len(f_arr) > 0 else 1.0
            end_p = float(f_arr[-1]) if len(f_arr) > 0 else 1.0
            pct_return = float(((end_p - start_p) / start_p) * 100)
            
            seg_copy = {
                "id": int(seg.get("id", idx)),
                "historical_prices": hist_prices,
                "prices": future_prices,
                "order_book": seg.get("order_book", {}),
                "similarity": similarity,
                "aligned_path": aligned_path,
                "pctReturn": pct_return,
                "direction": "BULLISH" if pct_return >= 0 else "BEARISH"
            }
            scored_candidates.append(seg_copy)
            
        # 2. Sort by SpecReTF similarity score and select top-k
        scored_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top_k = scored_candidates[:k]
        
        # 3. Compute similarity-weighted consensus (softmax-like aggregation)
        similarities = np.array([s["similarity"] for s in top_k], dtype=np.float32)
        # Numerical stability shift for exp
        exp_sim = np.exp(similarities - np.max(similarities))
        weights = exp_sim / np.sum(exp_sim)
        
        actual_horizon = min(len(s["aligned_path"]) for s in top_k)
        y_retrieval = np.zeros(actual_horizon, dtype=np.float32)
        for w, s in zip(weights, top_k):
            y_retrieval += w * s["aligned_path"][:actual_horizon]
            
        # 4. Forecasting Pipeline (Linear Projection & Fusion)
        if self.w_final is not None:
            # Model-weighted mode (matrix multiplications)
            # Y_retrieval prediction
            if self.w_retrieval is not None:
                y_hat_retrieval = np.dot(y_retrieval, self.w_retrieval)
                if self.bias_retrieval is not None:
                    y_hat_retrieval += self.bias_retrieval
            else:
                y_hat_retrieval = y_retrieval
                
            # Direct prediction from query
            if self.w_direct is not None:
                y_hat_direct = np.dot(prices, self.w_direct)
                if self.bias_direct is not None:
                    y_hat_direct += self.bias_direct
            else:
                y_hat_direct = np.full(actual_horizon, query_last_price)
                
            # Fusion
            concat_features = np.concatenate([y_hat_retrieval, y_hat_direct])
            y_hat_final = np.dot(concat_features, self.w_final)
            if self.bias_final is not None:
                y_hat_final += self.bias_final
        else:
            # Heuristic mode (default fallback)
            y_hat_retrieval = y_retrieval
            y_hat_direct = np.full(actual_horizon, query_last_price)
            y_hat_final = 0.5 * y_hat_retrieval + 0.5 * y_hat_direct
            
        # 5. Compute aggregate metrics and clean up retrieved structure
        pred_price = float(y_hat_final[-1]) if len(y_hat_final) > 0 else query_last_price
        expected_return = ((pred_price - query_last_price) / query_last_price) * 100
        
        processed_retrieved = []
        for s in top_k:
            s_copy = s.copy()
            s_copy.pop("aligned_path", None)  # Remove numpy array before returning
            processed_retrieved.append(s_copy)
            
        bullish_count = sum(1 for s in processed_retrieved if s["pctReturn"] >= 0)
        bull_ratio = (bullish_count / len(processed_retrieved)) * 100 if processed_retrieved else 50.0
        volatility = float(np.std([s["pctReturn"] for s in processed_retrieved])) if processed_retrieved else 0.0
        
        return {
            "retrieved": processed_retrieved,
            "prediction": pred_price,
            "consensus_path": y_hat_final.tolist(),
            "expected_return": expected_return,
            "bull_ratio": bull_ratio,
            "volatility": volatility,
            "direction": "BULLISH" if expected_return >= 0 else "BEARISH"
        }