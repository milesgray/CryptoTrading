# services/retrieval/forecaster.py

"""
Retrieval Forecaster Module.

Provides shape-similarity and frequency-domain forecasting classes:
- RetrievalForecaster: Base shape-similarity projection.
- SpecReTFForecaster: Frequency-domain matching utilizing Hann-windowed STFT coefficients,
  Jensen-Shannon Divergence on amplitude distributions, phase coherence, and recency decay.
"""

import numpy as np
from typing import Dict, Any
from encoder import RetrievalServiceEncoder
import torch
from chronos import ChronosPipeline

class RetrievalForecaster:
    """
    Standard shape-similarity retrieval-augmented forecaster.
    
    Uses Pearson correlation coefficient to retrieve and align historical continuations.
    """
    
    def __init__(self, encoder_service: RetrievalServiceEncoder):
        """
        Initialize the RetrievalForecaster.

        Args:
            encoder_service (RetrievalServiceEncoder): Vector encoder service instance.
        """
        self.encoder_service = encoder_service

    def forecast(
        self, 
        prices: np.ndarray, 
        order_book: Dict[str, Any], 
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Generate future price projections based on Pearson correlation shape similarity.

        Matches historical segments, scales them via multiplicative returns relative
        to the query's terminal price point, and aggregates them using similarity weights.

        Args:
            prices (np.ndarray): 1D array of query price history.
            order_book (Dict[str, Any]): Dictionary containing order book bids/asks.
            k (int): Number of nearest neighbors to retrieve. Defaults to 5.

        Returns:
            Dict[str, Any]: Forecast statistics and consensus path.

        Raises:
            ValueError: If no segments are found or database metadata is malformed.
        """
        # 1. Retrieve top-k similar historical segments
        retrieved = self.encoder_service.retrieve_segments(prices, order_book, k=k)
        
        if not retrieved:
            raise ValueError("No matching historical segments found in index.")

        processed_retrieved = []
        aligned_paths = []
        similarities = []
        
        query_last_price = float(prices[-1])
        
        for idx, seg in enumerate(retrieved):
            # Extract historical match window and subsequent future window
            hist_prices = seg.get("historical_prices")
            future_prices = seg.get("prices")
            
            if hist_prices is None or future_prices is None:
                raise ValueError(f"Historical match segment {idx} is malformed or missing required price data.")
                
            h_arr = np.array(hist_prices, dtype=np.float32)
            f_arr = np.array(future_prices, dtype=np.float32)
            
            # 2. Compute shape similarity via Pearson correlation coefficient
            q_mean, q_std = np.mean(prices), np.std(prices)
            h_mean, h_std = np.mean(h_arr), np.std(h_arr)
            
            if q_std > 1e-8 and h_std > 1e-8:
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
                base_val = h_arr[-1] if h_arr[-1] > 1e-8 else (f_arr[0] if f_arr[0] > 1e-8 else 1.0)
                aligned_path = query_last_price * (f_arr / base_val)
            else:
                raise ValueError(f"Future prices list for segment {idx} is empty.")
                
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
    """
    Spectral Retrospective Forecaster (SpecReTF).
    
    Compares query and candidate segments in the frequency domain using Hann-windowed STFTs,
    Jensen-Shannon Divergence on normalized amplitudes, and phase coherence.
    """
    
    def __init__(
        self,
        encoder_service: RetrievalServiceEncoder,
        frame_size: int = 16,
        hop_size: int = 4,
        alpha: float = 0.2,
        horizon: int = 60,
        w_retrieval: np.ndarray = None,
        w_direct: np.ndarray = None,
        w_final: np.ndarray = None,
        bias_retrieval: np.ndarray = None,
        bias_direct: np.ndarray = None,
        bias_final: np.ndarray = None,
    ):
        """
        Initialize the SpecReTFForecaster.

        Args:
            encoder_service (RetrievalServiceEncoder): Vector encoder service instance.
            frame_size (int): Temporal frame width for Fourier windowing. Defaults to 16.
            hop_size (int): Shift width between consecutive frames. Defaults to 4.
            alpha (float): Exponential decay factor for recency weighting. Defaults to 0.2.
            horizon (int): Forecasting steps. Defaults to 60.
            w_retrieval (np.ndarray, optional): Linear projection weights for retrieval.
            w_direct (np.ndarray, optional): Linear projection weights for direct query path.
            w_final (np.ndarray, optional): Final path fusion weight matrix.
            bias_retrieval (np.ndarray, optional): Retrieval bias matrix.
            bias_direct (np.ndarray, optional): Direct pathway bias matrix.
            bias_final (np.ndarray, optional): Fusion pathway bias matrix.
        """
        self.encoder_service = encoder_service
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.alpha = alpha
        self.horizon = horizon
        
        self.w_retrieval = w_retrieval
        self.w_direct = w_direct
        self.w_final = w_final
        self.bias_retrieval = bias_retrieval
        self.bias_direct = bias_direct
        self.bias_final = bias_final

    def _stft(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Short-Time Fourier Transform of a 1D sequence using a Hann window.

        Args:
            x (np.ndarray): 1D array of input price series.

        Returns:
            np.ndarray: 2D complex array of STFT coefficients. Shape: (frames, bins).
        """
        std = np.std(x)
        if std > 1e-8:
            x = (x - np.mean(x)) / std
        else:
            x = x - np.mean(x)

        L = len(x)
        M = self.frame_size
        B = self.hop_size
        
        if L < M:
            x = np.pad(x, (M - L, 0), mode="edge")
            L = len(x)
            
        W = (L - M) // B + 1
        window = np.hanning(M)
        
        stft_coefs = []
        for w in range(W):
            frame = x[w * B : w * B + M]
            coefs = np.fft.rfft(frame * window)
            stft_coefs.append(coefs)
            
        return np.array(stft_coefs)

    def _amplitude_jsd(self, p_q: np.ndarray, p_x: np.ndarray) -> float:
        """
        Compute the Jensen-Shannon Divergence (base 2) between two normalized amplitude distributions.

        Args:
            p_q (np.ndarray): Query normalized amplitude distribution.
            p_x (np.ndarray): Candidate normalized amplitude distribution.

        Returns:
            float: Calculated Jensen-Shannon divergence bounded between 0 and 1.
        """
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
        """
        Compute the phase coherence as the cosine of the mean phase difference.

        Args:
            phi_q (np.ndarray): Query phases in radians.
            phi_x (np.ndarray): Candidate phases in radians.
            weights (np.ndarray, optional): Spectral amplitude product weights. Defaults to None.

        Returns:
            float: Phase coherence value bounded between -1 and 1.
        """
        diff = phi_q - phi_x
        diff_wrapped = (diff + np.pi) % (2 * np.pi) - np.pi
        
        if weights is not None and np.sum(weights) > 1e-12:
            mean_diff = np.sum(diff_wrapped * weights) / np.sum(weights)
        else:
            mean_diff = np.mean(diff_wrapped)
            
        return float(np.cos(mean_diff))

    def _frequency_similarity(self, q_stft: np.ndarray, x_stft: np.ndarray) -> float:
        """
        Compute the frequency-aware composite similarity score between query and candidate STFTs.

        Calculates amplitude and phase similarity across all frames and aggregates
        them using an exponential recency decay weighting.

        Args:
            q_stft (np.ndarray): Query STFT coefficients.
            x_stft (np.ndarray): Candidate STFT coefficients.

        Returns:
            float: Composite similarity score.
        """
        W = min(len(q_stft), len(x_stft))
        if W == 0:
            return 0.0
            
        frame_scores = []
        eps = 1e-12
        
        for w in range(W):
            amp_q = np.abs(q_stft[w])
            amp_x = np.abs(x_stft[w])
            
            sum_q = np.sum(amp_q)
            sum_x = np.sum(amp_x)
            
            p_q = amp_q / (sum_q + eps)
            p_x = amp_x / (sum_x + eps)
            
            d_js = self._amplitude_jsd(p_q, p_x)
            s_amp = 1.0 - d_js
            
            phi_q = np.angle(q_stft[w])
            phi_x = np.angle(x_stft[w])
            phase_weights = amp_q * amp_x
            s_phase = self._phase_coherence(phi_q, phi_x, weights=phase_weights)
            
            s_w = s_amp + s_phase
            frame_scores.append(s_w)
            
        raw_weights = [self.alpha * ((1.0 - self.alpha) ** (W - 1 - w)) for w in range(W)]
        total_weight = sum(raw_weights)
        
        similarity = 0.0
        if total_weight > 1e-12:
            for w in range(W):
                similarity += (raw_weights[w] / total_weight) * frame_scores[w]
        else:
            similarity = float(np.mean(frame_scores)) if len(frame_scores) > 0 else 0.0
            
        return similarity

    def forecast(
        self, 
        prices: np.ndarray, 
        order_book: Dict[str, Any], 
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Forecast future prices using the SpecReTF frequency-aware retrieval-augmented approach.

        Queries the index, evaluates spectral similarities (JSD + Phase coherence),
        scales retrieved future continuations using returns-based multiplicative scaling,
        and fuses projection pathways (retrieval vs direct query linear weights).

        Args:
            prices (np.ndarray): 1D array of query price history.
            order_book (Dict[str, Any]): Dictionary containing bids/asks order book.
            k (int): Number of matches to retrieve from vector pool. Defaults to 5.

        Returns:
            Dict[str, Any]: Predicted target price, consensus path, expected return.

        Raises:
            ValueError: If query prices are empty or retrieved database values are malformed.
        """
        if len(prices) == 0:
            raise ValueError("prices array must not be empty")

        pool_size = max(k * 4, 20)
        retrieved = self.encoder_service.retrieve_segments(prices, order_book, k=pool_size)

        query_last_price = float(prices[-1])

        if not retrieved:
            raise ValueError("No matching historical segments found in index.")

        # Compute query STFT
        q_stft = self._stft(prices)
        
        scored_candidates = []
        for idx, seg in enumerate(retrieved):
            hist_prices = seg.get("historical_prices")
            future_prices = seg.get("prices")
            
            if hist_prices is None or future_prices is None:
                raise ValueError(f"Historical match segment {idx} is malformed or missing required price data.")
                
            h_arr = np.array(hist_prices, dtype=np.float32)
            f_arr = np.array(future_prices, dtype=np.float32)
            
            # Compute candidate STFT
            x_stft = self._stft(h_arr)
            
            # Compute SpecReTF similarity
            similarity = self._frequency_similarity(q_stft, x_stft)
            
            if len(f_arr) > 0:
                base_val = h_arr[-1] if h_arr[-1] > 1e-8 else (f_arr[0] if f_arr[0] > 1e-8 else 1.0)
                aligned_path = query_last_price * (f_arr / base_val)
            else:
                raise ValueError(f"Future prices list for segment {idx} is empty.")
                
            start_p = float(aligned_path[0]) if len(aligned_path) > 0 else query_last_price
            end_p = float(aligned_path[-1]) if len(aligned_path) > 0 else query_last_price
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
        
        # 3. Compute similarity-weighted consensus
        similarities = np.array([s["similarity"] for s in top_k], dtype=np.float32)
        exp_sim = np.exp(similarities - np.max(similarities))
        weights = exp_sim / np.sum(exp_sim)
        
        actual_horizon = min(len(s["aligned_path"]) for s in top_k)
        y_retrieval = np.zeros(actual_horizon, dtype=np.float32)
        for w, s in zip(weights, top_k):
            y_retrieval += w * s["aligned_path"][:actual_horizon]
            
        # 4. Forecasting Pipeline (Linear Projection & Fusion)
        if self.w_final is not None:
            if self.w_retrieval is not None:
                y_hat_retrieval = np.dot(y_retrieval, self.w_retrieval)
                if self.bias_retrieval is not None:
                    y_hat_retrieval += self.bias_retrieval
            else:
                y_hat_retrieval = y_retrieval
                
            if self.w_direct is not None:
                y_hat_direct = np.dot(prices, self.w_direct)
                if self.bias_direct is not None:
                    y_hat_direct += self.bias_direct
            else:
                y_hat_direct = np.full(actual_horizon, query_last_price)
                
            concat_features = np.concatenate([y_hat_retrieval, y_hat_direct])
            y_hat_final = np.dot(concat_features, self.w_final)
            if self.bias_final is not None:
                y_hat_final += self.bias_final
        else:
            y_hat_retrieval = y_retrieval
            y_hat_direct = np.full(actual_horizon, query_last_price)
            y_hat_final = 0.5 * y_hat_retrieval + 0.5 * y_hat_direct
            
        # 5. Compute aggregate metrics
        pred_price = float(y_hat_final[-1]) if len(y_hat_final) > 0 else query_last_price
        expected_return = ((pred_price - query_last_price) / query_last_price) * 100
        
        processed_retrieved = []
        for s in top_k:
            s_copy = s.copy()
            s_copy.pop("aligned_path", None)
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


class ChronosRAFForecaster:
    """
    Chronos-based Retrieval Augmented Forecasting (RAF) framework.

    Implements the exact query augmentation and forecasting mechanisms from the paper:
    1. Retrieve the top-k best-matching historical segments.
    2. Separately apply instance normalization (zero mean, unit variance) to the original
       context and retrieved time series (which contains retrieved context + retrieved future).
    3. Enforce continuity at the join by applying an additive offset to the retrieved
       segment so that its last value matches the first value of the context.
    4. Concatenate the adjusted retrieved segment(s) and the normalized original context.
    5. Pass the augmented time series to the Chronos model to predict the future window.
    6. Denormalize the generated predictions using the original context's mean and std.
    """

    def __init__(self, encoder_service: RetrievalServiceEncoder, chronos_pipeline: ChronosPipeline):
        """
        Initialize the ChronosRAFForecaster.

        Args:
            encoder_service (RetrievalServiceEncoder): Vector encoder service instance.
            chronos_pipeline (ChronosPipeline): Pre-trained Chronos pipeline instance.
        """
        self.encoder_service = encoder_service
        self.chronos_pipeline = chronos_pipeline

    def forecast(
        self, 
        prices: np.ndarray, 
        order_book: Dict[str, Any], 
        k: int = 1
    ) -> Dict[str, Any]:
        """
        Generate similarity-augmented future price projections using the Chronos RAF framework.

        Args:
            prices (np.ndarray): 1D array of query price history.
            order_book (Dict[str, Any]): Dictionary containing order book bids/asks.
            k (int): Number of nearest neighbors to retrieve. Defaults to 1 (paper standard).

        Returns:
            Dict[str, Any]: Forecast statistics and consensus path.
        """
        if len(prices) == 0:
            raise ValueError("prices array must not be empty")

        # 1. Retrieve top-k similar historical segments
        retrieved = self.encoder_service.retrieve_segments(prices, order_book, k=k)
        
        if not retrieved:
            raise ValueError("No matching historical segments found in index.")

        # 2. Separate instance normalization & continuity offset alignment
        # Normalize original query context x_orig: z_orig = (x_orig - mean_orig) / std_orig
        query_mean = float(np.mean(prices))
        query_std = float(np.std(prices))
        if query_std < 1e-8:
            query_std = 1.0
        z_orig = (prices - query_mean) / query_std

        # Process and normalize retrieved segments
        aligned_segments = []
        processed_retrieved = []
        for idx, seg in enumerate(retrieved):
            hist_prices = seg.get("historical_prices")
            future_prices = seg.get("prices")
            if hist_prices is None or future_prices is None:
                raise ValueError(f"Historical match segment {idx} is malformed or missing required price data.")
            
            h_arr = np.array(hist_prices, dtype=np.float32)
            f_arr = np.array(future_prices, dtype=np.float32)
            
            # Combine to form the retrieved series of length C+H
            ret_series = np.concatenate([h_arr, f_arr])
            ret_mean = np.mean(ret_series)
            ret_std = np.std(ret_series)
            if ret_std < 1e-8:
                ret_std = 1.0
            
            # Instance normalization of the retrieved series
            z_ret = (ret_series - ret_mean) / ret_std
            aligned_segments.append(z_ret)

            # Compute similarity metrics for response metadata using Pearson Correlation
            q_mean, q_std = np.mean(prices), np.std(prices)
            h_mean, h_std = np.mean(h_arr), np.std(h_arr)
            if q_std > 1e-8 and h_std > 1e-8:
                min_len = min(len(prices), len(h_arr))
                q_norm = (prices[:min_len] - q_mean) / q_std
                h_norm = (h_arr[:min_len] - h_mean) / h_std
                correlation = float(np.corrcoef(q_norm, h_norm)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
            similarity = 0.5 * (correlation + 1.0)
            
            start_p = float(f_arr[0]) if len(f_arr) > 0 else 1.0
            end_p = float(f_arr[-1]) if len(f_arr) > 0 else 1.0
            pct_return = float(((end_p - start_p) / start_p) * 100)
            
            seg_copy = {
                "id": int(seg.get("id", idx)),
                "historical_prices": hist_prices,
                "prices": future_prices,
                "order_book": seg.get("order_book", {}),
                "similarity": float(similarity),
                "pctReturn": pct_return,
                "direction": "BULLISH" if pct_return >= 0 else "BEARISH"
            }
            processed_retrieved.append(seg_copy)

        # Enforce continuity starting from z_orig (right to left)
        current_context = z_orig
        adjusted_segments = []
        for z_ret in reversed(aligned_segments):
            offset = current_context[0] - z_ret[-1]
            z_ret_adj = z_ret + offset
            adjusted_segments.insert(0, z_ret_adj)
            current_context = z_ret_adj

        # Concatenate adjusted retrieved segments and normalized original context to construct the RAF query
        raf_query_parts = adjusted_segments + [z_orig]
        s_raf = np.concatenate(raf_query_parts)

        # 3. Model Inference (Chronos)
        horizon = min(len(s["prices"]) for s in processed_retrieved)
        
        # Prepare context tensor for Chronos
        s_raf_tensor = torch.tensor(s_raf, dtype=torch.float32)
        
        # Run prediction on ChronosPipeline
        with torch.no_grad():
            samples = self.chronos_pipeline.predict(
                [s_raf_tensor],
                prediction_length=horizon,
                limit_prediction_length=False
            ) # shape: (1, num_samples, horizon)
        
        samples_np = samples[0].cpu().numpy() # shape: (num_samples, horizon)
        
        # 4. Denormalize prediction using original context's mean and std
        denorm_samples = samples_np * query_std + query_mean
        
        # Consensus path is the median of predictions
        consensus_path = np.median(denorm_samples, axis=0)
        
        # 5. Compute aggregate metrics
        query_last_price = float(prices[-1])
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