"""
Final Production Trading Integration
Incorporates: deque for O(1) updates, robust cache keys
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from collections import deque
import logging

from model import (
    CryptoKoopmanJEPA,
    create_crypto_feature_tensor,
    compute_price_window_hash
)

logger = logging.getLogger(__name__)


class JEPAStateAugmentation:
    """
    REFINED: Uses content-based cache keys (hash) instead of (price, length).
    
    Original issue: (price, length) causes collisions when same price
    appears at different market states.
    
    Fix: Use hash of actual price window → no collisions, deterministic.
    """
    
    def __init__(
        self,
        jepa_model: CryptoKoopmanJEPA,
        context_window: int = 768,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        cache_size: int = 1000
    ):
        """
        Args:
            jepa_model: trained CryptoKoopmanJEPA model
            context_window: size of price history window
            device: torch device
            cache_size: maximum number of cached embeddings
        """
        self.model = jepa_model.to(device)
        self.model.eval()
        self.context_window = context_window
        self.device = device
        self.cache_size = cache_size
        
        # REFINED: Cache with content-based keys
        self._embedding_cache = {}
        self._regime_cache = {}
        
    @torch.no_grad()
    def encode_price_history(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode price history with robust caching.
        
        REFINED: Uses content hash of price window as cache key instead
        of (price[-1], len(prices)) which can collide.
        """
        # Ensure we have enough data
        if len(prices) < self.context_window:
            pad_length = self.context_window - len(prices)
            prices = np.concatenate([np.full(pad_length, prices[0]), prices])
            timestamps = np.concatenate([
                np.linspace(timestamps[0] - pad_length, timestamps[0], pad_length, endpoint=False),
                timestamps
            ])
        
        # Take last context_window points
        prices = prices[-self.context_window:]
        
        # REFINED: Compute robust cache key
        # Use hash of actual price data → no collisions
        cache_key = compute_price_window_hash(prices)
        
        # Check cache
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key], self._regime_cache[cache_key]
        
        # Convert to tensor
        x = create_crypto_feature_tensor(prices, timestamps[-self.context_window:], self.context_window)
        x = x.unsqueeze(0).to(self.device)
        
        # Extract embedding and regime
        embedding, regime_probs = self.model.extract_regime_embeddings(x)
        
        # Convert to numpy
        embedding = embedding.cpu().numpy().squeeze()
        regime_probs = regime_probs.cpu().numpy().squeeze()
        
        # Cache results
        self._embedding_cache[cache_key] = embedding
        self._regime_cache[cache_key] = regime_probs
        
        # Limit cache size (FIFO eviction)
        if len(self._embedding_cache) > self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
            del self._regime_cache[oldest_key]
        
        return embedding, regime_probs
    
    def augment_observation(
        self,
        obs: Dict,
        token: str,
        price_history: np.ndarray,
        timestamp_history: np.ndarray
    ) -> Dict:
        """
        Augment environment observation with JEPA embeddings.
        """
        # Get embedding and regime
        embedding, regime_probs = self.encode_price_history(
            price_history,
            timestamp_history
        )
        
        # Add to observation
        obs_augmented = obs.copy()
        
        # Add embedding
        obs_augmented[f'{token}_jepa_embedding'] = embedding
        
        # Add regime information
        obs_augmented[f'{token}_regime_probs'] = regime_probs
        
        dominant_regime = np.argmax(regime_probs)
        regime_confidence = regime_probs[dominant_regime]
        
        obs_augmented[f'{token}_regime_id'] = dominant_regime
        obs_augmented[f'{token}_regime_confidence'] = regime_confidence
        
        # Regime entropy (uncertainty measure)
        regime_entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-10))
        obs_augmented[f'{token}_regime_entropy'] = regime_entropy
        
        return obs_augmented


class RegimeAwareLeverageController:
    """
    Leverage controller - no changes needed from v2.
    Already correctly implemented.
    """
    
    def __init__(
        self,
        jepa_augmentation: JEPAStateAugmentation,
        base_leverage: float = 10.0,
        max_leverage: float = 100.0,
        regime_leverage_multipliers: Optional[Dict[int, float]] = None
    ):
        self.jepa = jepa_augmentation
        self.base_leverage = base_leverage
        self.max_leverage = max_leverage
        
        if regime_leverage_multipliers is None:
            self.regime_multipliers = {i: 1.0 for i in range(8)}
        else:
            self.regime_multipliers = regime_leverage_multipliers
    
    def compute_optimal_leverage(
        self,
        price_history: np.ndarray,
        timestamp_history: np.ndarray,
        confidence_threshold: float = 0.6
    ) -> float:
        """Compute optimal leverage based on market regime"""
        _, regime_probs = self.jepa.encode_price_history(
            price_history,
            timestamp_history
        )
        
        dominant_regime = np.argmax(regime_probs)
        regime_confidence = regime_probs[dominant_regime]
        
        if regime_confidence < confidence_threshold:
            return self.base_leverage
        
        multiplier = self.regime_multipliers.get(dominant_regime, 1.0)
        leverage = self.base_leverage * multiplier
        
        return np.clip(leverage, 1.0, self.max_leverage)
    
    def should_reduce_exposure(
        self,
        price_history: np.ndarray,
        timestamp_history: np.ndarray,
        regime_transition_threshold: float = 0.3
    ) -> bool:
        """Detect regime transitions"""
        _, regime_probs = self.jepa.encode_price_history(
            price_history,
            timestamp_history
        )
        
        regime_entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-10))
        normalized_entropy = regime_entropy / np.log(8)
        
        return normalized_entropy > regime_transition_threshold


class JEPAEnhancedTradingEnv:
    """
    REFINED: Uses deque for O(1) history updates instead of list.
    
    Original: Lists require O(n) for append + slice operations
    Refined: Deque with maxlen provides O(1) append and automatic trimming
    """
    
    def __init__(
        self,
        base_env,
        jepa_model: CryptoKoopmanJEPA,
        use_regime_leverage: bool = True,
        context_window: int = 768,
        max_history: int = 2000  # Keep extra for analysis
    ):
        """
        Args:
            base_env: PerpetualFuturesEnv instance
            jepa_model: trained CryptoKoopmanJEPA model
            use_regime_leverage: whether to use regime-aware leverage
            context_window: JEPA context window size
            max_history: maximum history to keep (for efficiency)
        """
        self.env = base_env
        self.tokens = base_env.tokens
        self.context_window = context_window
        
        # JEPA components
        self.jepa_aug = JEPAStateAugmentation(
            jepa_model,
            context_window=context_window
        )
        
        if use_regime_leverage:
            self.leverage_controller = RegimeAwareLeverageController(
                self.jepa_aug,
                base_leverage=10.0,
                max_leverage=base_env.max_leverage
            )
        else:
            self.leverage_controller = None
        
        # REFINED: Use deque for O(1) updates
        # Deque with maxlen automatically removes oldest elements
        self.price_histories = {
            token: deque(maxlen=max_history) 
            for token in self.tokens
        }
        self.timestamp_histories = {
            token: deque(maxlen=max_history)
            for token in self.tokens
        }
        
    def reset(self):
        """Reset environment and history"""
        obs = self.env.reset()
        
        # Clear histories (O(1) with deque)
        for token in self.tokens:
            self.price_histories[token].clear()
            self.timestamp_histories[token].clear()
        
        return self._augment_observation(obs)
    
    def _augment_observation(self, obs: Dict) -> Dict:
        """
        Augment observation with JEPA features.
        REFINED: Uses deque for efficient history management.
        """
        obs_aug = obs.copy()
        
        for token in self.tokens:
            # Get current price and timestamp
            if token in obs and 'price' in obs[token]:
                current_price = float(obs[token]['price'][0])
                current_timestamp = float(obs[token].get('timestamp', [0])[0])
                
                # REFINED: O(1) append with deque
                self.price_histories[token].append(current_price)
                self.timestamp_histories[token].append(current_timestamp)
                
                # Augment if we have enough history
                if len(self.price_histories[token]) >= self.context_window:
                    # Convert deque to numpy array for processing
                    prices = np.array(self.price_histories[token])
                    timestamps = np.array(self.timestamp_histories[token])
                    
                    obs_aug = self.jepa_aug.augment_observation(
                        obs_aug,
                        token,
                        prices,
                        timestamps
                    )
        
        return obs_aug
    
    def step(self, action):
        """
        Step with regime-aware leverage adjustment.
        """
        # Adjust leverage based on regime (if enabled)
        if self.leverage_controller is not None:
            action_adjusted = {}
            for token, (action_type, leverage) in action.items():
                if len(self.price_histories[token]) >= self.context_window:
                    prices = np.array(self.price_histories[token])
                    timestamps = np.array(self.timestamp_histories[token])
                    
                    # Get regime-optimal leverage
                    optimal_leverage = self.leverage_controller.compute_optimal_leverage(
                        prices,
                        timestamps
                    )
                    
                    # Check for regime transitions
                    if self.leverage_controller.should_reduce_exposure(prices, timestamps):
                        optimal_leverage *= 0.5  # Reduce during transitions
                    
                    action_adjusted[token] = (action_type, optimal_leverage)
                else:
                    action_adjusted[token] = (action_type, leverage)
            
            action = action_adjusted
        
        # Execute step
        obs, reward, done, info = self.env.step(action)
        
        # Augment observation
        obs_aug = self._augment_observation(obs)
        
        # Add regime info to info dict
        if self.leverage_controller is not None:
            for token in self.tokens:
                if len(self.price_histories[token]) >= self.context_window:
                    _, regime_probs = self.jepa_aug.encode_price_history(
                        np.array(self.price_histories[token]),
                        np.array(self.timestamp_histories[token])
                    )
                    dominant_regime = np.argmax(regime_probs)
                    info[f'{token}_regime'] = dominant_regime
                    info[f'{token}_regime_confidence'] = regime_probs[dominant_regime]
        
        return obs_aug, reward, done, info
    
    def render(self, mode='human'):
        """Render with regime information"""
        self.env.render(mode)
        
        if self.leverage_controller is not None:
            print("\n" + "=" * 60)
            print("Market Regime Analysis (JEPA)")
            print("=" * 60)
            
            for token in self.tokens:
                if len(self.price_histories[token]) >= self.context_window:
                    _, regime_probs = self.jepa_aug.encode_price_history(
                        np.array(self.price_histories[token]),
                        np.array(self.timestamp_histories[token])
                    )
                    
                    dominant_regime = np.argmax(regime_probs)
                    confidence = regime_probs[dominant_regime]
                    
                    entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-10))
                    normalized_entropy = entropy / np.log(8)
                    
                    print(f"\n{token}:")
                    print(f"  Regime: {dominant_regime} (confidence: {confidence:.2%})")
                    print(f"  Uncertainty: {normalized_entropy:.2%}")
                    
                    optimal_lev = self.leverage_controller.compute_optimal_leverage(
                        np.array(self.price_histories[token]),
                        np.array(self.timestamp_histories[token])
                    )
                    print(f"  Recommended leverage: {optimal_lev:.1f}x")
                    
                    if normalized_entropy > 0.3:
                        print("  ⚠ Regime transition - reduce exposure!")


if __name__ == "__main__":
    """
    Demo showing refined integration
    """
    print("="*80)
    print("Final JEPA Trading Integration Demo")
    print("="*80)
    
    # This would integrate with your actual trading environment
    print("\nKey improvements:")
    print("  1. Deque for O(1) history updates (was list with O(n))")
    print("  2. Content-based cache keys (was collision-prone (price, len))")
    print("  3. Robust regime detection and leverage control")
    print("  4. All v2 stability fixes preserved")
    
    print("\nUsage:")
    print("  env = JEPAEnhancedTradingEnv(")
    print("      base_env=your_futures_env,")
    print("      jepa_model=trained_model,")
    print("      use_regime_leverage=True")
    print("  )")
    print("  ")
    print("  obs = env.reset()  # O(1) history clear")
    print("  obs, reward, done, info = env.step(action)  # O(1) history append")
    
    print("\n✓ Integration complete!")
