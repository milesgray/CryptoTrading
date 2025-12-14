import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from .util import OracleAction, OracleTradeSegment

logger = logging.getLogger(__name__)

class GreedyOracle:
    """
    Greedy trading strategy using future price information.
    
    This generates expert demonstrations by looking ahead at future prices
    and making optimal trading decisions using a greedy approach. Perfect for 
    behavior cloning and providing a strong baseline for offline RL algorithms.
    """
    
    def __init__(self, 
                 max_holding_period: int = 50,
                 min_profit_threshold: float = 0.02,  # 2% minimum profit
                 max_leverage: float = 20.0,
                 risk_per_trade: float = 0.1,  # Risk 10% of equity per trade
                 hold_time_min: int = 5,  # Minimum holding period
                 transaction_cost: float = 0.001):
        """
        Args:
            max_holding_period: How many steps ahead to look for trading opportunities
            min_profit_threshold: Minimum expected profit to open position
            max_leverage: Maximum leverage to use
            risk_per_trade: Fraction of equity to risk per trade
            hold_time_min: Minimum steps to hold a position
            transaction_cost: Trading cost as fraction
        """
        self.max_holding_period = max_holding_period
        self.min_profit_threshold = min_profit_threshold
        self.max_leverage = max_leverage
        self.risk_per_trade = risk_per_trade
        self.hold_time_min = hold_time_min
        self.transaction_cost = transaction_cost
        
    def compute_oracle_actions(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal actions and leverage using greedy approach.
        
        Args:
            prices: Array of prices
            
        Returns:
            actions: Array of OracleAction values
            leverages: Array of leverage values
        """
        n = len(prices)
        actions = np.zeros(n, dtype=np.int64)
        leverages = np.ones(n, dtype=np.float32)
        
        current_position = None  # (direction, entry_idx, leverage, target_exit_idx)
        
        for i in range(n):
            # If we have a position, check if we should close it
            if current_position is not None:
                direction, entry_idx, leverage, target_exit_idx = current_position
                
                # Check if we've reached target exit or need emergency exit
                if i >= target_exit_idx or self._should_close_position(prices, i, entry_idx, direction):
                    actions[i] = OracleAction.CLOSE
                    current_position = None
                else:
                    actions[i] = OracleAction.HOLD
                    continue
            
            # Look for new trading opportunity
            future_prices = prices[i+1 : min(i+1+self.max_holding_period, n)]
            
            if len(future_prices) > 0:
                direction, expected_return, exit_step = self._find_best_opportunity(
                    prices[i], future_prices
                )
                
                if direction != 'hold':
                    # Calculate optimal leverage
                    leverage = self._calculate_optimal_leverage(
                        expected_return, direction, future_prices, prices[i]
                    )
                    
                    # Open position
                    action_type = OracleAction.LONG if direction == 'long' else OracleAction.SHORT
                    actions[i] = action_type
                    leverages[i] = leverage
                    current_position = (direction, i, leverage, i + exit_step)
                else:
                    actions[i] = OracleAction.HOLD
        
        return actions, leverages
    
    def _find_best_opportunity(self, 
                              current_price: float,
                              future_prices: np.ndarray) -> Tuple[str, float, int]:
        """
        Find the best trading opportunity by looking at future prices.
        
        Args:
            current_price: Current price
            future_prices: Array of future prices
            
        Returns:
            (direction, expected_return, optimal_exit_step)
            direction: 'long', 'short', or 'hold'
            expected_return: Expected profit percentage
            optimal_exit_step: Step to exit position
        """
        if len(future_prices) == 0:
            return 'hold', 0.0, 0
        
        # Look for best long opportunity
        best_long_return = 0.0
        best_long_exit = 0
        
        for exit_step, future_price in enumerate(future_prices):
            if exit_step < self.hold_time_min:
                continue
                
            # Calculate return for long position
            price_return = (future_price - current_price) / current_price
            
            # Account for transaction costs (entry + exit)
            net_return = price_return - (2 * self.transaction_cost)
            
            if net_return > best_long_return:
                best_long_return = net_return
                best_long_exit = exit_step
        
        # Look for best short opportunity
        best_short_return = 0.0
        best_short_exit = 0
        
        for exit_step, future_price in enumerate(future_prices):
            if exit_step < self.hold_time_min:
                continue
                
            # Calculate return for short position
            price_return = (current_price - future_price) / current_price
            
            # Account for transaction costs
            net_return = price_return - (2 * self.transaction_cost)
            
            if net_return > best_short_return:
                best_short_return = net_return
                best_short_exit = exit_step
        
        # Choose the best direction
        if best_long_return > best_short_return and best_long_return > self.min_profit_threshold:
            return 'long', best_long_return, best_long_exit
        elif best_short_return > self.min_profit_threshold:
            return 'short', best_short_return, best_short_exit
        else:
            return 'hold', 0.0, 0
    def _calculate_optimal_leverage(self, 
                                   expected_return: float,
                                   direction: str,
                                   future_prices: np.ndarray,
                                   current_price: float) -> float:
        """
        Calculate optimal leverage based on expected return and risk.
        
        Uses Kelly Criterion approximation for position sizing.
        
        Args:
            expected_return: Expected profit percentage
            direction: 'long' or 'short'
            future_prices: Future prices to assess risk
            current_price: Current price
            
        Returns:
            Optimal leverage value
        """
        if len(future_prices) == 0:
            return 1.0
        
        # Calculate potential drawdown during holding period
        if direction == 'long':
            # For long: max drawdown is worst price vs current
            worst_price = min(future_prices[:self.hold_time_min + 10])
            max_drawdown = (current_price - worst_price) / current_price
        else:  # short
            # For short: max drawdown is highest price vs current
            worst_price = max(future_prices[:self.hold_time_min + 10])
            max_drawdown = (worst_price - current_price) / current_price
        
        # Add safety buffer
        max_drawdown = abs(max_drawdown) * 1.5  # 50% safety margin
        
        # Kelly criterion approximation: f = edge / odds
        # Simplified: leverage = expected_return / max_drawdown
        if max_drawdown > 0:
            optimal_leverage = (expected_return * self.risk_per_trade) / max_drawdown
        else:
            optimal_leverage = self.max_leverage
        
        # Clamp to reasonable range
        optimal_leverage = np.clip(optimal_leverage, 1.0, self.max_leverage)
        
        # Round down for safety
        optimal_leverage = max(1.0, float(int(optimal_leverage)))
        
        return optimal_leverage
    
    def _should_close_position(self, 
                              prices: np.ndarray,
                              current_step: int,
                              entry_idx: int,
                              direction: str) -> bool:
        """
        Decide whether to close an existing position.
        
        Args:
            prices: Array of prices
            current_step: Current timestep
            entry_idx: Entry index
            direction: 'long' or 'short'
            
        Returns:
            True if position should be closed
        """
        if current_step <= entry_idx:
            return False
            
        current_price = prices[current_step]
        entry_price = prices[entry_idx]
        
        # Calculate current P&L
        if direction == 'long':
            current_pnl_pct = (current_price - entry_price) / entry_price
        else:
            current_pnl_pct = (entry_price - current_price) / entry_price
        
        # Emergency exit if losing too much
        if current_pnl_pct < -0.05:  # -5% stop loss
            return True
        
        # Take profit early if target almost reached and trend reversing
        if current_pnl_pct > self.min_profit_threshold * 0.8 and current_step < len(prices) - 5:
            next_few_prices = prices[current_step + 1 : current_step + 6]
            if direction == 'long' and current_price > max(next_few_prices):
                return True  # Price peaked, sell now
            elif direction == 'short' and current_price < min(next_few_prices):
                return True  # Price bottomed, cover now
        
        return False
    
    def extract_trade_segments(
        self, 
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[OracleTradeSegment]:
        """
        Extract complete trade segments from greedy oracle solution.
        
        Args:
            prices: Price array
            timestamps: Optional timestamp array
            
        Returns:
            List of OracleTradeSegment objects
        """
        actions, leverages = self.compute_oracle_actions(prices)
        
        segments = []
        n = len(prices)
        i = 0
        
        while i < n:
            if actions[i] in [OracleAction.LONG, OracleAction.SHORT]:
                # Found position start
                start_idx = i
                direction = 1 if actions[i] == OracleAction.LONG else -1
                entry_price = prices[start_idx]
                leverage = leverages[start_idx]
                
                # Find end (CLOSE action or end of array)
                end_idx = start_idx + 1
                while end_idx < n and actions[end_idx] != OracleAction.CLOSE:
                    end_idx += 1
                
                if end_idx >= n:
                    # No close found, use last price as exit
                    end_idx = n - 1
                    
                exit_price = prices[end_idx]
                
                # Calculate metrics
                segment_prices = prices[start_idx:end_idx + 1]
                
                if direction == 1:  # Long
                    profit_pct = (exit_price - entry_price) / entry_price
                    max_adverse = (entry_price - np.min(segment_prices)) / entry_price
                else:  # Short
                    profit_pct = (entry_price - exit_price) / entry_price
                    max_adverse = (np.max(segment_prices) - entry_price) / entry_price
                
                roe = (profit_pct * leverage) - (self.transaction_cost * leverage * 2)
                
                segment = OracleTradeSegment(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    profit_pct=profit_pct,
                    roe_pct=roe,
                    leverage=leverage,
                    max_adverse_excursion=max_adverse
                )
                segments.append(segment)
                
                i = end_idx + 1
            else:
                i += 1
        
        return segments
    
    def get_statistics(self, prices: np.ndarray=None, segments: List[OracleTradeSegment]=None) -> dict:
        """
        Get statistics about the oracle's performance on price data.
        
        Args:
            prices: Price array
            
        Returns:
            Dict with various statistics
        """
        if not segments:
            segments = self.extract_trade_segments(prices)
        
        if not segments:
            return {
                'num_trades': 0,
                'total_return': 0.0,
                'win_rate': 0.0
            }
        
        profits = [s.profit_pct for s in segments]
        leverages = [s.leverage for s in segments]
        durations = [s.end_idx - s.start_idx for s in segments]
        
        longs = [s for s in segments if s.direction == 1]
        shorts = [s for s in segments if s.direction == -1]
        
        return {
            'num_trades': len(segments),
            'num_longs': len(longs),
            'num_shorts': len(shorts),
            'total_return': sum(profits),
            'mean_return': np.mean(profits),
            'std_return': np.std(profits),
            'win_rate': sum(1 for p in profits if p > 0) / len(profits),
            'avg_leverage': np.mean(leverages),
            'max_leverage': max(leverages),
            'min_leverage': min(leverages),
            'avg_hold_duration': np.mean(durations),
            'max_hold_duration': max(durations),
            'min_hold_duration': min(durations),
            'max_profit': max(profits),
            'min_profit': min(profits),
            'max_loss': min([p for p in profits if p < 0]),
            'sharpe_ratio': np.mean(profits) / (np.std(profits) + 1e-8),
            'profit_factor': sum(p for p in profits if p > 0) / abs(sum(p for p in profits if p < 0) or 1e-8),
            'gain_to_pain_ratio': sum(p for p in profits if p > 0) / (sum(abs(p) for p in profits if p < 0) or 1e-8),
            'total_wins': sum(1 for p in profits if p > 0),
            'total_losses': sum(1 for p in profits if p < 0),
        }


# Legacy function kept for backward compatibility
# Use util.generate_oracle_labels for new code
def generate_oracle_dataset(env, oracle_strategy, num_steps: int = 10000) -> Dict:
    """
    Generate expert demonstrations using oracle strategy.
    
    Args:
        env: Trading environment
        oracle_strategy: OracleStrategy instance
        num_steps: Number of transitions to generate
        
    Returns:
        Dictionary with observations, actions, rewards, etc.
    """
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    
    obs = env.reset()
    total_reward = 0
    episode_count = 0
    
    logger.info(f"Generating oracle dataset with {num_steps} steps...")
    
    for step in range(num_steps):
        # Get future prices for each token
        step_actions = []
        
        for token in env.tokens:
            # Get current state
            current_price = env.get_current_price(token)
            future_prices = env.get_predicted_prices(token)
            
            # Get current position (unused in new interface but kept for compatibility)
            # position_info = env.positions.get(token)
            
            # Get oracle action using new interface
            # Convert to continuous action for backward compatibility
            prices = np.array([current_price] + future_prices)
            oracle_actions, oracle_leverages = oracle_strategy.compute_oracle_actions(prices)
            
            # Convert first action to continuous format
            action_type = oracle_actions[0]
            leverage = oracle_leverages[0]
            
            if action_type == 0 or action_type == 3:  # Do nothing or close
                action = 0.0
            elif action_type == 1:  # Long
                # Map leverage [1, max_leverage] to [0.2, 1.0]
                strength = 0.2 + 0.8 * (leverage - 1) / (oracle_strategy.max_leverage - 1)
                action = strength
            else:  # Short (action_type == 2)
                # Map leverage [1, max_leverage] to [-0.2, -1.0]
                strength = 0.2 + 0.8 * (leverage - 1) / (oracle_strategy.max_leverage - 1)
                action = -strength
            step_actions.append(action)
        
        step_actions = np.array(step_actions)
        
        # Execute action
        next_obs, reward, done, info = env.step(step_actions)
        
        # Store transition
        observations.append(obs)
        actions.append(step_actions)
        rewards.append(reward)
        next_observations.append(next_obs)
        dones.append(done)
        
        total_reward += reward
        obs = next_obs
        
        if done:
            episode_count += 1
            obs = env.reset()
            if step % 1000 == 0:
                logger.info(f"  Step {step}/{num_steps}, Episodes: {episode_count}, "
                          f"Avg Reward: {total_reward/max(episode_count, 1):.3f}")
    
    logger.info("Dataset generation complete!")
    logger.info(f"  Total steps: {num_steps}")
    logger.info(f"  Total episodes: {episode_count}")
    logger.info(f"  Average episode reward: {total_reward/max(episode_count, 1):.3f}")
    
    return {
        'observations': np.array(observations),
        'actions': np.array(actions, dtype=np.float32),
        'rewards': np.array(rewards, dtype=np.float32).reshape(-1, 1),
        'next_observations': np.array(next_observations),
        'dones': np.array(dones, dtype=np.float32).reshape(-1, 1),
        'masks': 1.0 - np.array(dones, dtype=np.float32).reshape(-1, 1),
    }


# Usage example
if __name__ == "__main__":
    """
    Example usage:
    
    # 1. Create oracle strategy
    oracle = OracleStrategy(
        lookahead_steps=50,
        min_profit_threshold=0.02,
        max_leverage=50,
        risk_per_trade=0.1
    )
    
    # 2. Generate expert dataset
    dataset = generate_oracle_dataset(env, oracle, num_steps=20000)
    
    # 3. Train ReBRAC on this data
    for update in range(10000):
        batch = sample_from_dataset(dataset, batch_size=256)
        agent.update(batch)
    
    # 4. Fine-tune online
    for episode in range(100):
        # Agent will stay close to oracle strategy but improve further
        ...
    """
    pass