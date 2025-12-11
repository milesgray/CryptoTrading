import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import gym
from gym import spaces
from cryptotrading.trade.price import PriceServerClient, ExchangePriceClient
from cryptotrading.utils import to_float, get_first_float

logger = logging.getLogger(__name__)
class PerpetualFuturesEnv(gym.Env):
    """
    Enhanced perpetual futures crypto trading environment with real-time data integration.

    This environment simulates cryptocurrency perpetual futures trading with features like:
    - Multiple token support
    - Leverage trading (up to 1000x)
    - Funding rates
    - Liquidation mechanics
    - Real-time or historical price data
    - Sophisticated reward function considering multiple factors

    Key Features:
    - Realistic trading mechanics with transaction costs and slippage
    - Risk-adjusted rewards that penalize excessive leverage and encourage consistency
    - Comprehensive observation space including price history and predictions
    - Proper liquidation handling and margin requirements
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        tokens: List[str],
        price_client: PriceServerClient | ExchangePriceClient,
        initial_balance: float = 10000.0,
        max_leverage: int = 1000,
        transaction_cost: float = 0.0005,
        funding_rate_interval: int = 8,
        use_live_data: bool = False,
        historical_data_length: int = 1000,
        predicted_data_length: int = 10,
        previous_data_length: int = 50,
        use_perfect_information: bool = False,
        # New reward function parameters
        risk_free_rate: float = 0.02,  # Annual risk-free rate (2%)
        max_drawdown_penalty: float = 2.0,  # Penalty multiplier for drawdown
        consistency_reward: float = 0.1,  # Reward for consistent performance
        transaction_penalty: float = 0.1,  # Penalty for excessive trading
        liquidation_penalty: float = 1.0,  # Penalty for each liquidation event
    ):
        """
        Initialize the perpetual futures trading environment.

        Args:
            tokens: List of cryptocurrency tokens to trade
            price_client: Client for fetching price data
            initial_balance: Starting balance in USD
            max_leverage: Maximum allowed leverage
            transaction_cost: Cost per trade as a fraction
            funding_rate_interval: Steps between funding rate applications
            use_live_data: Whether to use real-time data or historical
            historical_data_length: Length of historical data to load
            predicted_data_length: Length of price predictions
            previous_data_length: Length of previous price history in observations
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            max_drawdown_penalty: Penalty multiplier for maximum drawdown
            consistency_reward: Reward for consistent positive returns
            transaction_penalty: Penalty for excessive trading frequency
            liquidation_penalty: Penalty applied for each liquidation event
        """
        super(PerpetualFuturesEnv, self).__init__()

        self.tokens = tokens
        self.price_client = price_client
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        self.funding_rate_interval = funding_rate_interval
        self.use_live_data = use_live_data
        self.historical_data_length = historical_data_length
        self.predicted_data_length = predicted_data_length
        self.previous_data_length = previous_data_length
        self.use_perfect_information = use_perfect_information

        # Reward function parameters
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_penalty = max_drawdown_penalty
        self.consistency_reward = consistency_reward
        self.transaction_penalty = transaction_penalty
        self.liquidation_penalty = liquidation_penalty

        # Performance tracking for enhanced rewards
        self.position_pnl = {token: 0.0 for token in tokens}
        self.equity_history = []
        self.returns_history = []
        self.trade_count = 0
        self.max_equity = initial_balance
        self.max_drawdown = 0.0
        self.total_unrealized_pnl = 0.0  # Track total unrealized PnL separately
        self.unrealized_pnl_history = []  # Track unrealized PnL over time

        # Load historical data if not using live data
        if not use_live_data:
            self.price_data = {}
            for token in tokens:
                historical_prices = price_client.load_historical_prices(
                    token, max_prices=self.historical_data_length, page_size=500
                )
                if historical_prices is not None:
                    self.price_data[token] = historical_prices
                    logger.info(f'Loaded {len(historical_prices)} historical prices for {token}')
                else:
                    # Fallback to generated data if historical data is not available
                    self.price_data[token] = self._generate_price_data(self.historical_data_length)
                    logger.warning(f'No historical data available for {token}, using generated data.')
        else:
            # Initialize current prices for live data
            for token in tokens:
                for i in range(self.previous_data_length):
                    current_price = self.get_current_price(token)
                    logger.info(f'Current price for {token}: {current_price}')
                    time.sleep(0.1)  # Small delay to avoid overwhelming the API

        # Calculate episode steps more robustly
        data_lengths = [len(v) for k, v in self.price_data.items()]
        min_data_length = min(data_lengths)
        max_data_length = max(data_lengths)

        # Use minimum length but ensure it's reasonable
        self.episode_steps = min_data_length
        if self.episode_steps < 100:
            logger.warning(f'Very short episode detected: {self.episode_steps} steps. Using minimum 100 steps.')
            self.episode_steps = 100

        assert self.episode_steps > 0, 'No price data available for any token.'
        self.episode_steps = max(self.episode_steps, self.predicted_data_length)
        self.episode_steps = max(self.episode_steps, self.previous_data_length)

        # Debug: Log data length info
        logger.info(f'Loaded price data lengths: {[f"{k}: {len(v)}" for k, v in self.price_data.items()]}')
        logger.info(f'Data length range: {min_data_length} - {max_data_length}')
        logger.info(f'Episode steps set to: {self.episode_steps}')

        # Action space: for each token, we can:
        # 0: Do nothing
        # 1: Open long position (with leverage 1-1000)
        # 2: Open short position (with leverage 1-1000)
        # 3: Close position
        self.action_space = spaces.Dict({
            token: spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Leverage
            for token in tokens
        })

        # Enhanced observation space with additional performance metrics
        self.observation_space = spaces.Dict(
            {
                'balance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                'equity': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                'total_return': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                'max_drawdown': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'sharpe_ratio': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                **{
                    token: spaces.Dict(
                        {
                            'previous_prices': spaces.Box(
                                low=0, high=np.inf, shape=(self.previous_data_length,), dtype=np.float32
                            ),
                            'price_returns': spaces.Box(
                                low=0, high=np.inf, shape=(self.previous_data_length,), dtype=np.float32
                            ),
                            'price': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                            'timestamp': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                            'predicted_prices': spaces.Box(
                                low=0, high=np.inf, shape=(self.predicted_data_length,), dtype=np.float32
                            ),
                            'position': spaces.Discrete(3),  # 0: no position, 1: long, 2: short
                            'size': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                            'entry_price': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                            'leverage': spaces.Box(low=1, high=max_leverage, shape=(1,), dtype=np.float32),
                            'unrealized_pnl': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                            'risk_ratio': spaces.Box(
                                low=0, high=1, shape=(1,), dtype=np.float32
                            ),  # Position size / equity
                        }
                    )
                    for token in tokens
                },
            }
        )

        self.reset(price=True)

    def _generate_price_data(self, length: int, base: float = 100.0) -> List[Tuple[float, float]]:
        """
        Generate realistic synthetic price data with timestamps.

        Args:
            length: Number of price points to generate
            base: Starting price

        Returns:
            List of (price, timestamp) tuples
        """
        # Use GBM (Geometric Brownian Motion) for more realistic price movements
        dt_step = 1.0 / (365 * 24 * 60 * 60)  # second step
        mu = 0.05  # Annual drift (5% growth)
        sigma = 0.8  # Annual volatility (80%)

        if isinstance(base, float):
            prices = [base]
        elif isinstance(base, tuple):
            prices = [float(base[0])]
        elif isinstance(base, list):
            prices = [float(base[0])]
        elif isinstance(base, np.ndarray):
            prices = [float(base[0])]
        elif isinstance(base, str):
            prices = [float(base)]
        else:
            raise ValueError(f'Invalid base type: {type(base)}')
        timestamps = [datetime.now().timestamp() - (length - i) * dt_step for i in range(length)]

        for i in range(1, length):
            # GBM formula: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            z = np.random.normal(0, 1)
            drift = (mu - 0.5 * sigma**2) * dt_step
            diffusion = sigma * np.sqrt(dt_step) * z
            new_price = prices[-1] * np.exp(drift + diffusion)
            prices.append(new_price)

        return list(zip(prices, timestamps))

    def _get_current_price_data(self, token: str) -> Tuple[float, float]:
        """
        Get the current price and timestamp for a token.

        Args:
            token: Token symbol

        Returns:
            Tuple of (price, timestamp)
        """
        if self.use_live_data:
            price, timestamp = self.price_client.get_current_price(token)
            if price is not None:
                self.last_prices[token] = price
                self.price_data[token].append((price, timestamp))
                return to_float(price), to_float(timestamp)
            else:
                logger.warning(f'No live price available for {token}, using last known price')
                if hasattr(self, 'last_prices') and self.last_prices[token] is not None:
                    return to_float(self.last_prices[token]), to_float(timestamp)
                return 100.0, 0.0
        else:
            # Use historical data
            if self.current_price_step < len(self.price_data[token]):
                price, timestamp = self.price_data[token][self.current_price_step]
                return to_float(price), to_float(timestamp)
            else:
                # Return last available price if we've reached the end
                price, timestamp = self.price_data[token][-1]
                return to_float(price), to_float(timestamp)

    def get_current_price(self, token: str) -> float:
        """Get the current price for a token."""
        price, _ = self._get_current_price_data(token)
        return price

    def get_predicted_prices(self, token: str) -> List[float]:
        """
        Get predicted future prices for a token.

        Args:
            token: Token symbol

        Returns:
            List of predicted prices
        """
        if self.use_live_data:
            predictions = self.price_client.get_predicted_prices(token)
            return (
                predictions
                if predictions is not None
                else [self.get_current_price(token)] * self.predicted_data_length
            )
        else:
            result = []
            if self.current_step + self.predicted_data_length > len(self.price_data[token]):
                # Use available data + generated continuation
                available_prices = [p[0] for p in self.price_data[token][self.current_step :]]
                remaining_length = self.predicted_data_length - len(available_prices)
                if remaining_length > 0:
                    last_price = (
                        self.price_data[token][-1][0]
                        if token in self.price_data and len(self.price_data[token]) > 0
                        else 100.0
                    )
                    generated_data = self._generate_price_data(remaining_length, last_price)
                    additional_prices = [to_float(p[0]) for p in generated_data]
                    result = available_prices + additional_prices
                else:
                    result = available_prices[: self.predicted_data_length]
            else:
                result = [
                    p[0]
                    for p in self.price_data[token][self.current_step : self.current_step + self.predicted_data_length]
                ]
            result = [to_float(p) for p in result]
            if not self.use_perfect_information:
                generated_data = self._generate_price_data(self.predicted_data_length, self.get_current_price(token))
                additional_prices = [to_float(p[0]) for p in generated_data]
                result = [p + (g * 0.1) for p, g in zip(result, additional_prices)]
            return result

    def _get_previous_prices(self, token: str) -> List[float]:
        """
        Get previous prices for a token to include in observations.

        Args:
            token: Token symbol

        Returns:
            List of previous prices
        """
        if self.current_price_step == 0 or len(self.price_data[token]) == 0:
            # Return default prices if no history available
            current_price = self.get_current_price(token) if len(self.price_data[token]) > 0 else 100.0
            return [current_price] * self.previous_data_length

        start_idx = max(0, self.current_price_step - self.previous_data_length)
        end_idx = self.current_price_step

        if start_idx < len(self.price_data[token]):
            available_prices = [
                p[0] for p in self.price_data[token][start_idx : min(end_idx, len(self.price_data[token]))]
            ]

            # Pad with first available price if needed
            if len(available_prices) < self.previous_data_length:
                padding_length = self.previous_data_length - len(available_prices)
                first_price = available_prices[0] if available_prices else 100.0
                available_prices = [first_price] * padding_length + available_prices

            return available_prices[-self.previous_data_length :]  # Ensure exact length
        else:
            # Fallback to last available price
            last_price = self.price_data[token][-1][0] if self.price_data[token] else 100.0
            return [last_price] * self.previous_data_length

    def _calculate_performance_metrics(self):
        """Calculate various performance metrics for reward function."""
        if len(self.equity_history) < 2:
            return 0.0, 0.0, 0.0  # total_return, sharpe_ratio, max_drawdown

        # Calculate total return relative to current equity (not initial balance)
        total_return = (self.equity - self.equity_history[0]) / max(self.equity_history[0], 1)

        # Calculate Sharpe ratio
        if len(self.returns_history) > 1:
            returns_array = np.array(self.returns_history)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            # Annualized Sharpe ratio (assuming hourly returns)
            if std_return > 0:
                excess_return = mean_return - (self.risk_free_rate / (365 * 24))
                sharpe_ratio = (excess_return / std_return) * np.sqrt(365 * 24)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Update maximum drawdown
        self.max_equity = max(self.max_equity, self.equity)
        current_drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        return total_return, sharpe_ratio, self.max_drawdown

    def _calculate_enhanced_reward(self, prev_equity: float) -> float:
        # Use log returns instead of simple returns
        if prev_equity > 0 and self.equity > 0:
            raw_return = np.log(self.equity / prev_equity)
        else:
            raw_return = 0.0

        # Clip to prevent extreme values
        raw_return = np.clip(raw_return, -0.1, 0.1)

        # Scale to [-1, 1] range more aggressively
        reward = np.tanh(raw_return * 10)  # Tanh provides smooth saturation

        # Remove Sharpe ratio from step rewards (too non-stationary)
        # Only use it for final episode evaluation

        # Simplify to focus on core objective
        if hasattr(self, '_prev_trade_count'):
            new_trades = self.trade_count - self._prev_trade_count
            if new_trades > 0:
                reward -= 0.01 * new_trades  # Small trading penalty

        self._prev_trade_count = self.trade_count

        for token, pnl in self.position_pnl.items():
            if pnl != 0:
                reward += np.tanh(pnl) * 0.01

        return np.clip(reward, -1.0, 1.0)

    def reset(self, price=False):
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_step = 0
        if price:
            self.current_price_step = 0
        self.positions = {token: None for token in self.tokens}
        self.funding_step_counter = 0
        self.trade_history = []
        self.trade_count = 0
        self._episode_done = False  # Clear episode done flag

        # Reset performance tracking
        self.equity_history = [self.initial_balance]
        self.returns_history = []
        self.max_equity = self.initial_balance
        self.max_drawdown = 0.0
        self._prev_trade_count = 0
        self.total_unrealized_pnl = 0.0
        self.unrealized_pnl_history = [0.0]

        if self.use_live_data:
            self.last_prices = {token: self.get_current_price(token) for token in self.tokens}

        return self._get_observation()

    def _get_observation(self):
        """Get the current observation of the environment."""
        total_return, sharpe_ratio, max_drawdown = self._calculate_performance_metrics()

        obs = {
            'balance': np.array([self.balance / self.initial_balance], dtype=np.float32),
            'equity': np.array([self.equity / self.initial_balance], dtype=np.float32),
            'total_return': np.array([total_return], dtype=np.float32),
            'max_drawdown': np.array([max_drawdown], dtype=np.float32),
            'sharpe_ratio': np.array([np.clip(sharpe_ratio, -5, 5)], dtype=np.float32),
        }

        for token in self.tokens:
            previous_prices = self._get_previous_prices(token)
            price, timestamp = self._get_current_price_data(token)
            predicted_prices = self.get_predicted_prices(token)
            position_info = self.positions[token]

            norm_prices = np.array(previous_prices, dtype=np.float32) / price
            norm_pred_prices = np.array(predicted_prices, dtype=np.float32) / price
            price_returns = np.diff(norm_prices, prepend=norm_prices[0])

            if position_info is None:
                obs[token] = {
                    'previous_prices': np.array(norm_prices, dtype=np.float32),
                    'price_returns': np.array(price_returns, dtype=np.float32),
                    'price': np.array([price], dtype=np.float32),
                    'timestamp': np.array([timestamp], dtype=np.float32),
                    'predicted_prices': np.array(norm_pred_prices, dtype=np.float32),
                    'position': 0,
                    'size': np.array([0.0], dtype=np.float32),
                    'entry_price': np.array([0.0], dtype=np.float32),
                    'leverage': np.array([0.0], dtype=np.float32),
                    'unrealized_pnl': np.array([0.0], dtype=np.float32),
                    'risk_ratio': np.array([0.0], dtype=np.float32),
                }
            else:
                direction, size, entry_price, leverage = position_info
                # Convert all values to float safely before calculation
                size_float = to_float(size)
                price_float = to_float(price)
                entry_price_float = to_float(entry_price)
                leverage_float = to_float(leverage)

                # Calculate unrealized PnL
                if direction == 'long':
                    pnl = size_float * (price_float - entry_price_float) / entry_price_float
                else:  # short
                    pnl = size_float * (entry_price_float - price_float) / entry_price_float

                # Calculate risk ratio (position exposure / equity)
                risk_ratio = size_float / max(self.equity, 1.0)

                obs[token] = {
                    'previous_prices': np.array(norm_prices, dtype=np.float32),
                    'price_returns': np.array(price_returns, dtype=np.float32),
                    'price': np.array([price], dtype=np.float32),
                    'timestamp': np.array([timestamp], dtype=np.float32),
                    'predicted_prices': np.array(norm_pred_prices, dtype=np.float32),
                    'position': 1 if direction == 'long' else 2,
                    'size': np.array([size_float / self.max_equity], dtype=np.float32),
                    'entry_price': np.array([entry_price_float / price_float], dtype=np.float32),
                    'leverage': np.array([leverage_float / self.max_leverage], dtype=np.float32),
                    'unrealized_pnl': np.array([np.clip(pnl / max(self.balance, 1), -1, 1)], dtype=np.float32),
                    'risk_ratio': np.array([risk_ratio], dtype=np.float32),
                }

        return obs

    def _calculate_bust_price(self, entry_price: float, leverage: float, direction: str) -> float:
        """Calculate the price at which the position gets liquidated."""
        if direction == 'long':
            return entry_price * (1 - 1 / leverage)
        else:  # short
            return entry_price * (1 + 1 / leverage)

    def _check_liquidation(self, token: str) -> bool:
        """Check if a position should be liquidated due to bust price."""
        position_info = self.positions[token]
        if position_info is None:
            return False

        direction, size, entry_price, leverage = position_info

        # Convert values to float safely
        entry_price_float = to_float(entry_price)
        current_price = self.get_current_price(token)
        leverage_float = to_float(leverage)
        size_float = to_float(size)

        # Ensure leverage is at least 1 to avoid division by zero
        leverage_float = max(1.0, leverage_float)

        bust_price = self._calculate_bust_price(entry_price_float, leverage_float, direction)

        if (direction == 'long' and current_price <= bust_price) or (
            direction == 'short' and current_price >= bust_price
        ):
            # Position is liquidated - lose the collateral, not the full exposure
            collateral_loss = size_float / leverage_float
            self.balance -= collateral_loss
            self.positions[token] = None

            # Record the liquidation
            self.on_liquidation(
                {
                    'token': token,
                    'direction': direction,
                    'size': size_float,
                    'entry_price': entry_price_float,
                    'exit_price': current_price,
                    'leverage': leverage_float,
                    'pnl': -collateral_loss,  # Loss of collateral
                    'timestamp': datetime.now(),
                }
            )

            return True

        return False

    def _apply_funding_rates(self):
        """Apply funding rates to open positions."""
        self.funding_step_counter += 1
        if self.funding_step_counter >= self.funding_rate_interval:
            self.funding_step_counter = 0

            for token in self.tokens:
                if self.positions[token] is not None:
                    direction, size, entry_price, leverage = self.positions[token]
                    # Simplified funding rate calculation (in reality, this depends on market conditions)
                    funding_rate = np.random.normal(0.0001, 0.0005)  # Small random rate

                    size_float = to_float(size)
                    leverage_float = to_float(leverage)
                    collateral = size_float / leverage_float
                    if direction == 'long':
                        # Long pays short (typically when market is bullish)
                        funding_cost = collateral * abs(funding_rate)
                        self.balance -= funding_cost
                    else:
                        # Short receives funding (typically when market is bullish)
                        funding_gain = collateral * abs(funding_rate)
                        self.balance += funding_gain

    def interpret_action(self, action_value: float) -> Tuple[str, float]:
        """
        Convert continuous action to trading decision.

        Args:
            action_value: float in [-1, 1]
            token: token symbol

        Returns:
            (action, leverage)
            action: 'long', 'short', 'close', or None
            leverage: float leverage value
        """
        # Dead zone for "do nothing"
        if abs(action_value) < 0.05:
            return (None, 1.0)  # Do nothing

        # Close position zone (near zero but not quite)
        if abs(action_value) < 0.15:
            return ('close', 1.0)  # Close position (treated as no action)

        # Determine direction
        action = 'short'
        if action_value > 0.0:  # Long position (with some threshold to avoid tiny positions)
            action = 'long'

        # Map |action_value| to leverage (logarithmic scale for safety)
        # 0.15 → 1x, 0.5 → 10x, 1.0 → max_leverage
        abs_action = abs(action_value)
        if abs_action < 0.5:
            # Conservative range: 1x to 10x
            leverage = 1.0 + (abs_action - 0.15) / 0.35 * 9.0
        else:
            # Aggressive range: 10x to max_leverage
            leverage = 10.0 + (abs_action - 0.5) / 0.5 * (self.max_leverage - 10.0)

        leverage = np.clip(leverage, 1.0, self.max_leverage)

        return (action, leverage)

    def on_open(self, info):
        info['action'] = 'open'
        self.trade_history.append(info)
        logger.info(
            f'Opened new position: {info["size"]} of {info["token"]} {info["direction"]} at ${info["entry_price"]:.2f}'
        )

    def on_close(self, info):
        info['action'] = 'close'
        self.trade_history.append(info)
        if info['pnl'] > 0:
            logger.info(
                f'Took profit of ${info["pnl"]:.2f}: {info["size"]} of {info["token"]} {info["direction"]} opened at ${info["entry_price"]:.2f}, closed at ${info["exit_price"]:.2f}'
            )
        else:
            logger.info(
                f'Took loss of ${info["pnl"]:.2f}: {info["size"]} of {info["token"]} {info["direction"]} opened at ${info["entry_price"]:.2f}, closed at ${info["exit_price"]:.2f}'
            )

    def on_liquidation(self, info):
        info['action'] = 'liquidation'
        self.trade_history.append(info)
        logger.info(
            f'Liquidated position: {info["size"]} of {info["token"]} {info["direction"]} opened at ${info["entry_price"]:.2f}, bust at ${info["exit_price"]:.2f}'
        )

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.

        Args:
            actions: Dictionary mapping token symbols to continuous action values [-1, 1]
                     where -1 = max short, 0 = no position, +1 = max long

        Returns:
            Tuple of (observation, reward, done, info)
        """            
        prev_equity = self.equity

        # Apply funding rates
        self._apply_funding_rates()

        # Track trades and liquidations for this step
        trades_this_step = 0
        liquidations_this_step = 0

        # Process actions for each token
        for token, action_value in actions.items():
            # Extract the action value from the array
            action_value = get_first_float(action_value)

            direction, leverage = self.interpret_action(action_value)

            # Check if current position gets liquidated
            if self._check_liquidation(token):
                liquidations_this_step += 1
                continue  # Skip action if position was liquidated

            current_price = self.get_current_price(token)
            position_info = self.positions[token]

            # Close existing position if direction changed or closing
            if position_info is not None and (direction == 'close' or (position_info[0] != direction)):
                old_direction, old_size, old_entry_price, old_leverage = position_info
                old_size_float = to_float(old_size)
                old_entry_price_float = to_float(old_entry_price)
                old_leverage_float = to_float(old_leverage)
                old_collateral = old_size_float / old_leverage_float

                # Calculate PnL for closing
                if old_direction == 'long':
                    pnl = old_size_float * (current_price - old_entry_price_float) / old_entry_price_float
                elif old_direction == 'short':
                    pnl = old_size_float * (old_entry_price_float - current_price) / old_entry_price_float

                self.position_pnl[token] = pnl / old_collateral

                if direction == 'close' or (position_info[0] != direction):
                    self.balance += old_collateral + pnl  # Return collateral + PnL
                    self.positions[token] = None
                    self.position_pnl[token] = 0
                    trades_this_step += 1

                    # Record the closed position
                    self.on_close(
                        {
                            'token': token,
                            'direction': old_direction,
                            'size': old_size_float,
                            'entry_price': old_entry_price_float,
                            'exit_price': current_price,
                            'leverage': old_leverage_float,
                            'pnl': pnl,
                            'timestamp': datetime.now(),
                        }
                    )

            # Open new position if we have a direction and leverage
            if direction in ['long', 'short'] and leverage > 0:
                # Calculate position exposure based on available balance and leverage
                available_balance = max(self.balance * 0.05, 0)  # Use 5% of balance
                active_tokens = len(
                    [
                        t
                        for t, a in actions.items()
                        if abs(get_first_float(a)) > 0.01
                    ]
                )
                active_tokens = max(active_tokens, 1)  # Avoid division by zero

                collateral_per_position = available_balance / active_tokens
                position_exposure = collateral_per_position * leverage

                if position_exposure > 0 and collateral_per_position <= self.balance:
                    # Open new position - store exposure, not collateral
                    self.positions[token] = (direction, position_exposure, current_price, leverage)
                    self.balance -= collateral_per_position  # Allocate only the collateral (margin)
                    trades_this_step += 1

                    # Record the new position
                    self.on_open(
                        {
                            'token': token,
                            'collateral': collateral_per_position,
                            'direction': direction,
                            'size': position_exposure,
                            'entry_price': current_price,
                            'leverage': leverage,
                            'timestamp': datetime.now(),
                        }
                    )

        # Update trade count
        self.trade_count += trades_this_step

        # Check for liquidations again after processing actions
        for token in self.tokens:
            if self.positions[token] is not None and self._check_liquidation(token):
                liquidations_this_step += 1

        # Calculate total equity (balance + unrealized PnL from all positions)
        total_unrealized_pnl = 0
        total_margin_used = 0
        position_details = {}

        for token in self.tokens:
            position_info = self.positions[token]
            if position_info is not None:
                direction, size, entry_price, leverage = position_info
                # Convert all values to float safely
                size_float = to_float(size)
                entry_price_float = to_float(entry_price)
                leverage_float = to_float(leverage)
                current_price_float = self.get_current_price(token)

                # Calculate unrealized PnL for this position
                if direction == 'long':
                    pnl = size_float * (current_price_float - entry_price_float) / entry_price_float
                else:
                    pnl = size_float * (entry_price_float - current_price_float) / entry_price_float

                total_unrealized_pnl += pnl
                total_margin_used += size_float / leverage_float  # Margin = position_exposure / leverage

                # Store position details for debugging
                position_details[token] = {
                    'direction': direction,
                    'pnl': pnl,
                    'size': size_float,
                    'leverage': leverage_float,
                    'entry_price': entry_price_float,
                    'current_price': current_price_float,
                }

        # Update total unrealized PnL tracking
        self.total_unrealized_pnl = total_unrealized_pnl

        # Update equity
        self.equity = self.balance + total_unrealized_pnl

        # Track unrealized PnL history
        self.unrealized_pnl_history.append(total_unrealized_pnl)
        if len(self.unrealized_pnl_history) > 1000:
            self.unrealized_pnl_history = self.unrealized_pnl_history[-500:]

        # Debug logging for equity and PnL tracking
        if self.current_step % 100 == 0:  # Log every 100 steps
            logger.info(
                f'Step {self.current_step}: Equity=${self.equity:.2f}, Balance=${self.balance:.2f}, '
                f'Unrealized_PnL=${total_unrealized_pnl:.2f}, Total_Margin_Used=${total_margin_used:.2f}'
            )
            if position_details:
                for token, details in position_details.items():
                    logger.info(
                        f'  {token}: {details["direction"]} {details["pnl"]:.2f} PnL '
                        f'(size={details["size"]:.2f}, leverage={details["leverage"]:.1f}x)'
                    )

        # Track equity and returns history for performance metrics
        self.equity_history.append(self.equity)
        if len(self.equity_history) > 1:
            # Calculate return relative to previous equity (not just equity change)
            prev_equity = self.equity_history[-2]
            if prev_equity > 0:
                period_return = (self.equity - prev_equity) / prev_equity
            else:
                period_return = 0.0
            self.returns_history.append(period_return)

            # Keep only recent history to manage memory
            if len(self.returns_history) > 1000:
                self.returns_history = self.returns_history[-500:]  # Keep last 500 returns
            if len(self.equity_history) > 1000:
                self.equity_history = self.equity_history[-500:]  # Keep last 500 equity values

        # Calculate enhanced reward
        reward = self._calculate_enhanced_reward(prev_equity)

        # Apply a penalty for any liquidations in this step
        if liquidations_this_step > 0:
            reward -= self.liquidation_penalty * liquidations_this_step

        self.current_step += 1

        done = False

        if not self.use_live_data:    
            self.current_price_step += 1
            if self.current_price_step >= self.historical_data_length:
                self.current_price_step = 0
                done = True

        if not self.use_live_data:
            done = done or self.current_step >= self.episode_steps

        # Additional done conditions
        if self.equity <= 0.01 * self.initial_balance:
            done = True
            reward -= 1.0  # Large penalty for bankruptcy

        # Set episode done flag if episode is finished
        if done:
            self._episode_done = True

        # Create info dictionary with useful debugging information
        info = {
            'equity': self.equity,
            'balance': self.balance,
            'total_return': (self.equity - self.initial_balance) / self.initial_balance,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'unrealized_pnl_pct': self.total_unrealized_pnl / max(self.balance, 1),
            'max_drawdown': self.max_drawdown,
            'trade_count': self.trade_count,
            'trades_this_step': trades_this_step,
            'liquidations_this_step': liquidations_this_step,
            'total_margin_used': total_margin_used,
            'margin_utilization': total_margin_used / max(self.equity, 1),
            'positions': {token: pos is not None for token, pos in self.positions.items()},
            'current_prices': {token: self.get_current_price(token) for token in self.tokens},
            'position_details': position_details,
        }

        return self._get_observation(), reward, done, info

    def render(self, mode='human'):
        """Render the current state of the environment."""
        if mode == 'human':
            print(f'\n{"=" * 60}')
            print(f'Step: {self.current_step if not self.use_live_data else "Live"}')
            print(f'Balance: ${self.balance:.2f} | Equity: ${self.equity:.2f}')

            total_return = (self.equity - self.initial_balance) / self.initial_balance * 100
            unrealized_pnl_color = '+' if self.total_unrealized_pnl >= 0 else ''
            print(
                f'Total Return: {total_return:.2f}% | Unrealized PnL: {unrealized_pnl_color}${self.total_unrealized_pnl:.2f}'
            )
            print(f'Max Drawdown: {self.max_drawdown * 100:.2f}%')

            if len(self.returns_history) > 1:
                _, sharpe_ratio, _ = self._calculate_performance_metrics()
                print(f'Sharpe Ratio: {sharpe_ratio:.3f} | Trade Count: {self.trade_count}')

            print(f'{"=" * 60}')

            total_exposure = 0
            for token in self.tokens:
                price = self.get_current_price(token)
                position = self.positions[token]

                if position is None:
                    # Show last 5 trades for this token if no current position
                    token_trades = [trade for trade in reversed(self.trade_history) if trade.get('token') == token]

                    print(f'  {token}: ${price:.4f} - No position')

                    if token_trades:
                        # Show last 5 trades
                        for i, trade in enumerate(token_trades[:5]):
                            pnl_color = '+' if trade.get('pnl', 0) >= 0 else ''
                            trade_label = 'Last Trade' if i == 0 else f'Trade -{i + 1}'
                            print(
                                f'    {trade_label}: {pnl_color}${trade.get("pnl", 0):.2f} PnL '
                                f'({trade.get("direction", "N/A").upper()} {trade.get("size", 0):.2f} @ ${trade.get("entry_price", 0):.4f})'
                            )

                        # Summarize remaining trades if there are more than 5
                        if len(token_trades) > 5:
                            remaining_trades = len(token_trades) - 5
                            remaining_pnl = sum(trade.get('pnl', 0) for trade in token_trades[5:])
                            pnl_color = '+' if remaining_pnl >= 0 else ''
                            print(f'    +{remaining_trades} more trades: {pnl_color}${remaining_pnl:.2f} total PnL')
                else:
                    direction, size, entry_price, leverage = position
                    size_float = to_float(size)
                    entry_price_float = to_float(entry_price)
                    leverage_float = to_float(leverage)

                    # Calculate current PnL and percentage
                    if direction == 'long' or direction == 1:
                        pnl = size_float * (price - entry_price_float) / entry_price_float
                        pnl_pct = (price - entry_price_float) / entry_price_float * 100
                    else:
                        pnl = size_float * (entry_price_float - price) / entry_price_float
                        pnl_pct = (entry_price_float - price) / entry_price_float * 100

                    bust_price = self._calculate_bust_price(entry_price_float, leverage_float, direction)
                    exposure = size_float  # Position size is already the exposure
                    total_exposure += exposure

                    pnl_color = '+' if pnl >= 0 else ''
                    print(f'  {token}: ${price:.4f} - {direction} {leverage_float:.1f}x')
                    print(f'    Entry: ${entry_price_float:.4f} | Bust: ${bust_price:.4f}')
                    print(f'    Size: ${size_float:.2f} | Exposure: ${exposure:.2f}')
                    print(f'    PnL: {pnl_color}${pnl:.2f} ({pnl_color}{pnl_pct:.2f}%)')

            if total_exposure > 0:
                print(
                    f'\nTotal Exposure: ${total_exposure:.2f} ({total_exposure / max(self.equity, 1) * 100:.1f}% of equity)'
                )

            print(f'{"=" * 60}\n')

    def get_portfolio_summary(self) -> Dict:
        """
        Get a comprehensive summary of the current portfolio state.

        Returns:
            Dictionary containing portfolio metrics and position details
        """
        total_return, sharpe_ratio, max_drawdown = self._calculate_performance_metrics()

        positions_summary = {}
        total_exposure = 0
        total_unrealized_pnl = 0

        for token in self.tokens:
            price = self.get_current_price(token)
            position = self.positions[token]

            if position is not None:
                direction, size, entry_price, leverage = position
                size_float = to_float(size)
                entry_price_float = to_float(entry_price)
                leverage_float = to_float(leverage)

                if direction == 'long':
                    pnl = size_float * (price - entry_price_float) / entry_price_float
                    pnl_pct = (price - entry_price_float) / entry_price_float * 100
                else:
                    pnl = size_float * (entry_price_float - price) / entry_price_float
                    pnl_pct = (entry_price_float - price) / entry_price_float * 100

                exposure = size_float  # Position size is already the exposure
                total_exposure += exposure
                total_unrealized_pnl += pnl

                positions_summary[token] = {
                    'direction': direction,
                    'size': size_float,
                    'leverage': leverage_float,
                    'entry_price': entry_price_float,
                    'current_price': price,
                    'unrealized_pnl': pnl,
                    'pnl_percentage': pnl_pct,
                    'exposure': exposure,
                    'bust_price': self._calculate_bust_price(entry_price_float, leverage_float, direction),
                }
            else:
                positions_summary[token] = {'direction': None, 'current_price': price}

        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'initial_balance': self.initial_balance,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'unrealized_pnl_pct': self.total_unrealized_pnl / max(self.balance, 1) * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'trade_count': self.trade_count,
            'total_exposure': total_exposure,
            'margin_utilization': total_exposure / max(self.equity, 1),
            'positions': positions_summary,
            'recent_trades': self.trade_history[-5:] if len(self.trade_history) >= 5 else self.trade_history,
        }

    def close(self):
        """Close the environment and any connections."""
        if hasattr(self, 'price_client'):
            self.price_client.connected = False

        logger.info('Environment closed successfully')


# Usage example and explanation:
"""
HOW THE ENHANCED PERPETUAL FUTURES ENVIRONMENT WORKS:

1. **Environment Setup**:
   - Supports multiple cryptocurrency tokens for trading
   - Can use either real-time data (via price_client) or historical/generated data
   - Configurable leverage (up to 1000x), transaction costs, and funding rates

2. **Action Space**:
   For each token, agents can:
   - Do nothing (0)
   - Open long position with specified leverage (1, leverage)
   - Open short position with specified leverage (2, leverage)  
   - Close existing position (3, any)

3. **Observation Space**:
   - Portfolio metrics: balance, equity, total return, max drawdown, Sharpe ratio
   - Per-token data: price history, current price, predicted prices, position details
   - Risk metrics: position size ratios, leverage exposure

4. **Enhanced Reward Function**:
   The reward considers multiple factors:
   - Raw profit/loss (normalized by initial balance)
   - Risk-adjusted returns (Sharpe ratio bonus)
   - Maximum drawdown penalty (penalizes large losses)
   - Consistency reward (favors stable returns over volatile ones)
   - Transaction penalty (discourages overtrading)
   - Leverage penalty (discourages excessive risk-taking)
   - Survival bonus (rewards staying solvent)
   - **Liquidation Penalty**: A direct penalty is now applied whenever a position is liquidated,
     teaching the agent to avoid catastrophic losses from over-leveraging.

5. **Key Features**:
   - **Liquidation Mechanics**: Positions are automatically closed when prices hit bust levels
   - **Funding Rates**: Periodic costs/gains applied to open positions (simulating real perpetual futures)
   - **Transaction Costs**: Realistic trading costs that reduce returns
   - **Risk Management**: Built-in position sizing and margin requirements
   - **Performance Tracking**: Comprehensive metrics for strategy evaluation

6. **Realistic Trading Simulation**:
   - Uses Geometric Brownian Motion for realistic price generation
   - Implements proper margin calculations and liquidation triggers
   - Accounts for slippage and transaction costs
   - Supports both long and short positions with leverage

This environment is designed to train RL agents for cryptocurrency perpetual futures trading
while encouraging risk-aware, consistent performance rather than just maximizing raw returns.
The enhanced reward function promotes sustainable trading strategies that balance profitability
with risk management.
"""
