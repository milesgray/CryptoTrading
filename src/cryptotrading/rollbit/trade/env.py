import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional

class PerpetualFuturesEnv(gym.Env):
    """
    A custom environment for perpetual futures crypto trading with multiple tokens
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 tokens: List[str],
                 initial_balance: float = 10000.0,
                 price_data: Optional[Dict[str, np.ndarray]] = None,
                 max_leverage: int = 1000,
                 transaction_cost: float = 0.0005,  # 0.05% taker fee
                 funding_rate_interval: int = 8,    # funding every 8 steps
                 ):
        super(PerpetualFuturesEnv, self).__init__()
        
        self.tokens = tokens
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        self.funding_rate_interval = funding_rate_interval
        
        # Generate price data if not provided
        if price_data is None:
            self.price_data = self._generate_price_data(1000)  # 1000 steps by default
        else:
            self.price_data = price_data
        
        # Action space: for each token, we can:
        # 0: Do nothing
        # 1: Open long position (with leverage 1-1000)
        # 2: Open short position (with leverage 1-1000)
        # 3: Close position
        self.action_space = spaces.Dict({
            token: spaces.Tuple((
                spaces.Discrete(4),  # Action type
                spaces.Box(low=1, high=max_leverage, shape=(1,), dtype=np.float32)  # Leverage
            )) for token in tokens
        })
        
        # Observation space: balance, equity, and for each token:
        # current price, position direction, position size, entry price, leverage, unrealized PnL
        self.observation_space = spaces.Dict({
            'balance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'equity': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            **{
                token: spaces.Dict({
                    'price': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                    'position': spaces.Discrete(3),  # 0: no position, 1: long, 2: short
                    'size': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                    'entry_price': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                    'leverage': spaces.Box(low=1, high=max_leverage, shape=(1,), dtype=np.float32),
                    'unrealized_pnl': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                }) for token in tokens
            }
        })
        
        self.reset()
    
    def _generate_price_data(self, length: int) -> Dict[str, np.ndarray]:
        """Generate random price data for each token"""
        price_data = {}
        for token in self.tokens:
            # Random walk with drift
            returns = np.random.normal(0.0001, 0.02, length)
            price = 100 * np.exp(np.cumsum(returns))
            price_data[token] = price
        return price_data
    
    def reset(self):
        """Reset the environment to initial state"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_step = 0
        self.positions = {token: None for token in self.tokens}
        self.funding_step_counter = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get the current observation of the environment"""
        obs = {
            'balance': np.array([self.balance], dtype=np.float32),
            'equity': np.array([self.equity], dtype=np.float32),
        }
        
        for token in self.tokens:
            price = self.price_data[token][self.current_step]
            position_info = self.positions[token]
            
            if position_info is None:
                obs[token] = {
                    'price': np.array([price], dtype=np.float32),
                    'position': 0,
                    'size': np.array([0.0], dtype=np.float32),
                    'entry_price': np.array([0.0], dtype=np.float32),
                    'leverage': np.array([1.0], dtype=np.float32),
                    'unrealized_pnl': np.array([0.0], dtype=np.float32),
                }
            else:
                direction, size, entry_price, leverage = position_info
                # Calculate unrealized PnL
                if direction == 'long':
                    pnl = size * (price - entry_price) / entry_price
                else:  # short
                    pnl = size * (entry_price - price) / entry_price
                
                obs[token] = {
                    'price': np.array([price], dtype=np.float32),
                    'position': 1 if direction == 'long' else 2,
                    'size': np.array([size], dtype=np.float32),
                    'entry_price': np.array([entry_price], dtype=np.float32),
                    'leverage': np.array([leverage], dtype=np.float32),
                    'unrealized_pnl': np.array([pnl], dtype=np.float32),
                }
        
        return obs
    
    def _calculate_bust_price(self, entry_price: float, leverage: float, direction: str) -> float:
        """Calculate the price at which the position gets liquidated"""
        if direction == 'long':
            return entry_price * (1 - 1 / leverage)
        else:  # short
            return entry_price * (1 + 1 / leverage)
    
    def _check_liquidation(self, token: str) -> bool:
        """Check if a position should be liquidated due to bust price"""
        position_info = self.positions[token]
        if position_info is None:
            return False
        
        direction, size, entry_price, leverage = position_info
        current_price = self.price_data[token][self.current_step]
        bust_price = self._calculate_bust_price(entry_price, leverage, direction)
        
        if (direction == 'long' and current_price <= bust_price) or \
           (direction == 'short' and current_price >= bust_price):
            # Position is liquidated
            self.balance -= size  # Lose the entire collateral
            self.positions[token] = None
            return True
        
        return False
    
    def _apply_funding_rates(self):
        """Apply funding rates to open positions (simplified)"""
        self.funding_step_counter += 1
        if self.funding_step_counter >= self.funding_rate_interval:
            self.funding_step_counter = 0
            
            for token in self.tokens:
                if self.positions[token] is not None:
                    direction, size, entry_price, leverage = self.positions[token]
                    # Simplified funding rate calculation
                    funding_rate = np.random.normal(0.0001, 0.0005)  # Random small rate
                    
                    if direction == 'long':
                        # Long pays short
                        funding_cost = size * funding_rate
                        self.balance -= funding_cost
                    else:
                        # Short pays long
                        funding_gain = size * funding_rate
                        self.balance += funding_gain
    
    def step(self, actions: Dict[str, Tuple[int, float]]):
        """Execute one time step within the environment"""
        self.current_step += 1
        if self.current_step >= len(self.price_data[self.tokens[0]]):
            self.current_step = 0  # Loop back to beginning if we reach the end
        
        # Apply funding rates
        self._apply_funding_rates()
        
        # Process actions for each token
        for token, (action_type, leverage) in actions.items():
            leverage = max(1, min(self.max_leverage, leverage))  # Clamp leverage to valid range
            
            # Check if current position gets liquidated
            liquidated = self._check_liquidation(token)
            if liquidated:
                continue  # Skip action if position was liquidated
            
            current_price = self.price_data[token][self.current_step]
            position_info = self.positions[token]
            
            if action_type == 0:  # Do nothing
                continue
                
            elif action_type == 1 or action_type == 2:  # Open long or short
                direction = 'long' if action_type == 1 else 'short'
                
                # Close existing position if any
                if position_info is not None:
                    old_direction, old_size, old_entry_price, old_leverage = position_info
                    # Calculate PnL for closing
                    if old_direction == 'long':
                        pnl = old_size * (current_price - old_entry_price) / old_entry_price
                    else:
                        pnl = old_size * (old_entry_price - current_price) / old_entry_price
                    
                    self.balance += old_size + pnl  # Return collateral + PnL
                    self.positions[token] = None
                
                # Calculate position size based on leverage and available balance
                position_size = (self.balance * leverage) * (1 - self.transaction_cost)
                
                # Open new position
                self.positions[token] = (direction, position_size, current_price, leverage)
                self.balance -= position_size  # Allocate collateral
                
            elif action_type == 3:  # Close position
                if position_info is not None:
                    direction, size, entry_price, leverage = position_info
                    
                    # Calculate PnL
                    if direction == 'long':
                        pnl = size * (current_price - entry_price) / entry_price
                    else:
                        pnl = size * (entry_price - current_price) / entry_price
                    
                    # Return collateral + PnL
                    self.balance += size + pnl
                    self.positions[token] = None
        
        # Check for liquidations after processing actions
        for token in self.tokens:
            self._check_liquidation(token)
        
        # Calculate total equity (balance + unrealized PnL from all positions)
        total_unrealized_pnl = 0
        for token in self.tokens:
            position_info = self.positions[token]
            if position_info is not None:
                direction, size, entry_price, leverage = position_info
                current_price = self.price_data[token][self.current_step]
                
                if direction == 'long':
                    pnl = size * (current_price - entry_price) / entry_price
                else:
                    pnl = size * (entry_price - current_price) / entry_price
                
                total_unrealized_pnl += pnl
        
        self.equity = self.balance + total_unrealized_pnl
        
        # Calculate reward (change in equity)
        reward = self.equity - self._prev_equity if hasattr(self, '_prev_equity') else 0
        self._prev_equity = self.equity
        
        # Check if done (bankrupt or reached end of data)
        done = self.equity <= 0 or self.current_step >= len(self.price_data[self.tokens[0]]) - 1
        
        return self._get_observation(), reward, done, {}
    
    def render(self, mode='human'):
        """Render the current state of the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, Equity: ${self.equity:.2f}")
            for token in self.tokens:
                price = self.price_data[token][self.current_step]
                position = self.positions[token]
                if position is None:
                    print(f"  {token}: ${price:.2f} - No position")
                else:
                    direction, size, entry_price, leverage = position
                    bust_price = self._calculate_bust_price(entry_price, leverage, direction)
                    print(f"  {token}: ${price:.2f} - {direction.upper()} {leverage}x (Entry: ${entry_price:.2f}, Bust: ${bust_price:.2f})")
            print()
    
    def close(self):
        pass

# Example usage
if __name__ == "__main__":
    # Create environment with two tokens
    tokens = ['BTC', 'ETH']
    
    # Generate sample price data
    np.random.seed(42)
    price_data = {
        'BTC': 10000 + np.cumsum(np.random.normal(0, 100, 1000)),
        'ETH': 300 + np.cumsum(np.random.normal(0, 10, 1000))
    }
    
    env = PerpetualFuturesEnv(
        tokens=tokens,
        initial_balance=10000,
        price_data=price_data,
        max_leverage=1000,
        transaction_cost=0.0005
    )
    
    # Run a simple simulation with random actions
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Random actions for each token
        actions = {
            'BTC': (np.random.randint(0, 4), np.random.uniform(1, 100)),
            'ETH': (np.random.randint(0, 4), np.random.uniform(1, 100))
        }
        
        obs, reward, done, info = env.step(actions)
        total_reward += reward
        
        if env.current_step % 100 == 0:
            env.render()
    
    print(f"Total reward: ${total_reward:.2f}")
    print(f"Final equity: ${env.equity:.2f}")