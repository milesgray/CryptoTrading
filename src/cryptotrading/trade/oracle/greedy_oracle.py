import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class GreedyOracle:
    """
    Near-perfect trading strategy using future price information.
    
    This generates expert demonstrations by looking ahead at future prices
    and making optimal trading decisions. Perfect for behavior cloning
    and providing a strong baseline for offline RL algorithms like ReBRAC.
    """
    
    def __init__(self, 
                 lookahead_steps: int = 50,
                 min_profit_threshold: float = 0.02,  # 2% minimum profit
                 max_leverage: float = 100,
                 risk_per_trade: float = 0.1,  # Risk 10% of equity per trade
                 hold_time_min: int = 5,  # Minimum holding period
                 transaction_cost: float = 0.001):
        """
        Args:
            lookahead_steps: How many steps ahead to look for trading opportunities
            min_profit_threshold: Minimum expected profit to open position
            max_leverage: Maximum leverage to use
            risk_per_trade: Fraction of equity to risk per trade
            hold_time_min: Minimum steps to hold a position
            transaction_cost: Trading cost as fraction
        """
        self.lookahead_steps = lookahead_steps
        self.min_profit_threshold = min_profit_threshold
        self.max_leverage = max_leverage
        self.risk_per_trade = risk_per_trade
        self.hold_time_min = hold_time_min
        self.transaction_cost = transaction_cost
        self.position_opened_at = {}  # Track when positions were opened
        
    def find_best_opportunity(self, 
                             current_price: float,
                             future_prices: List[float]) -> Tuple[str, float, int]:
        """
        Find the best trading opportunity by looking at future prices.
        
        Args:
            current_price: Current price
            future_prices: List of future prices (lookahead_steps long)
            
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
    
    def calculate_optimal_leverage(self, 
                                   expected_return: float,
                                   direction: str,
                                   future_prices: List[float],
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
    
    def should_close_position(self,
                             token: str,
                             current_step: int,
                             position_info: Tuple,
                             current_price: float,
                             future_prices: List[float]) -> bool:
        """
        Decide whether to close an existing position.
        
        Args:
            token: Token symbol
            current_step: Current timestep
            position_info: (direction, size, entry_price, leverage, target_exit_step)
            current_price: Current price
            future_prices: Future prices
            
        Returns:
            True if position should be closed
        """
        direction, size, entry_price, leverage, target_exit_step = position_info
        
        # Check if we've reached target exit
        steps_held = current_step - self.position_opened_at.get(token, current_step)
        if steps_held >= target_exit_step:
            return True
        
        # Calculate current P&L
        if direction == 'long':
            current_pnl_pct = (current_price - entry_price) / entry_price
        else:
            current_pnl_pct = (entry_price - current_price) / entry_price
        
        # Emergency exit if losing too much
        if current_pnl_pct < -0.05:  # -5% stop loss
            return True
        
        # Take profit early if target almost reached and trend reversing
        if current_pnl_pct > self.min_profit_threshold * 0.8 and len(future_prices) > 5:
            next_few_prices = future_prices[:5]
            if direction == 'long' and current_price > max(next_few_prices):
                return True  # Price peaked, sell now
            elif direction == 'short' and current_price < min(next_few_prices):
                return True  # Price bottomed, cover now
        
        return False
    
    def get_action(self,
                   token: str,
                   current_step: int,
                   current_price: float,
                   future_prices: List[float],
                   current_position: Tuple = None) -> Tuple[int, float]:
        """
        Get optimal action for a token.
        
        Args:
            token: Token symbol
            current_step: Current timestep
            current_price: Current price
            future_prices: List of future prices (lookahead_steps)
            current_position: Current position info or None
            
        Returns:
            (action_type, leverage)
            action_type: 0=nothing, 1=long, 2=short, 3=close
        """
        # If we have a position, check if we should close it
        if current_position is not None:
            if self.should_close_position(token, current_step, current_position, 
                                         current_price, future_prices):
                if token in self.position_opened_at:
                    del self.position_opened_at[token]
                return (3, 1.0)  # Close position
            else:
                return (0, 1.0)  # Hold position
        
        # Find best trading opportunity
        direction, expected_return, exit_step = self.find_best_opportunity(
            current_price, future_prices
        )
        
        if direction == 'hold':
            return (0, 1.0)  # Do nothing
        
        # Calculate optimal leverage
        leverage = self.calculate_optimal_leverage(
            expected_return, direction, future_prices, current_price
        )
        
        # Open position
        action_type = 1 if direction == 'long' else 2
        self.position_opened_at[token] = current_step
        
        return (action_type, leverage)
    
    def get_continuous_action(self,
                            token: str,
                            current_step: int,
                            current_price: float,
                            future_prices: List[float],
                            current_position: Tuple = None) -> float:
        """
        Get continuous action in [-1, 1] range for use with continuous control algorithms.
        
        Args:
            Same as get_action
            
        Returns:
            action: float in [-1, 1]
                0 = close/no position
                positive = long (magnitude = leverage strength)
                negative = short (magnitude = leverage strength)
        """
        action_type, leverage = self.get_action(
            token, current_step, current_price, future_prices, current_position
        )
        
        if action_type == 0 or action_type == 3:  # Do nothing or close
            return 0.0
        elif action_type == 1:  # Long
            # Map leverage [1, max_leverage] to [0.2, 1.0]
            strength = 0.2 + 0.8 * (leverage - 1) / (self.max_leverage - 1)
            return strength
        else:  # Short (action_type == 2)
            # Map leverage [1, max_leverage] to [-0.2, -1.0]
            strength = 0.2 + 0.8 * (leverage - 1) / (self.max_leverage - 1)
            return -strength


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
            
            # Get current position
            position_info = env.positions.get(token)
            if position_info is not None:
                direction, size, entry_price, leverage = position_info
                current_position = (direction, size, entry_price, leverage, oracle_strategy.lookahead_steps)
            else:
                current_position = None
            
            # Get oracle action (continuous)
            action = oracle_strategy.get_continuous_action(
                token, env.current_step, current_price, future_prices, current_position
            )
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