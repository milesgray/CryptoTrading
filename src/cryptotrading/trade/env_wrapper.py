import gym
from gym import spaces
import numpy as np

from .env import PerpetualFuturesEnv # Make sure the env file is accessible

class FlattenedActionWrapper(gym.ActionWrapper):
    """
    A wrapper to convert the agent's flat, continuous action vector into the
    environment's complex, structured action dictionary.
    """
    def __init__(self, env: PerpetualFuturesEnv):
        super().__init__(env)
        self.env = env
        self.tokens = self.env.tokens
        
        # The agent will output a single flat vector. 
        # For each token, it needs to decide on:
        # 1. Action Type (continuous value we will discretize)
        # 2. Leverage (continuous value we will discretize)        
        num_actions_per_token = 1
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(len(self.tokens) * num_actions_per_token,), 
            dtype=np.float32
        )

    def action(self, action: np.ndarray) -> dict:
        """
        Translates the agent's flat action vector into the environment's dictionary format.
        
        Args:
            action (np.ndarray): A flat array of continuous values between -1 and 1.
        
        Returns:
            dict: The structured action dictionary expected by PerpetualFuturesEnv.
        """
        structured_action = {}
        action_idx = 0
        
        for token in self.tokens:
            action_type_continuous = action[action_idx]
            action_type, leverage = self.env.interpret_action(action_type_continuous)
            action_int = 0
            if action_type == "long":
                action_int = 1
            elif action_type == "short":
                action_int = 2
            elif action_type == "close":
                action_int = 3
            structured_action[token] = (int(action_int), float(leverage))
            action_idx += 1
            
        return structured_action

class FlattenedObservationWrapper(gym.ObservationWrapper):
    """
    A wrapper to flatten the environment's dictionary-based observation into a single vector.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        
        # Flatten the complex observation space into a single Box space.
        # We define a canonical order for the keys to ensure consistency.
        self.obs_key_order = ['balance', 'equity']
        for token in self.env.tokens:
            self.obs_key_order.extend([
                f'{token}_price', f'{token}_position', f'{token}_size',
                f'{token}_entry_price', f'{token}_leverage', f'{token}_unrealized_pnl'
            ])

        # Calculate the total size of the flattened observation space
        low = []
        high = []
        
        original_obs_space = self.env.observation_space
        
        low.extend(original_obs_space['balance'].low)
        high.extend(original_obs_space['balance'].high)
        low.extend(original_obs_space['equity'].low)
        high.extend(original_obs_space['equity'].high)
        
        for token in self.env.tokens:
            token_space = original_obs_space[token]
            low.extend(token_space['previous_prices'].low)
            high.extend(token_space['previous_prices'].high)
            low.extend(token_space['price'].low)
            high.extend(token_space['price'].high)
            low.extend(token_space['predicted_prices'].low)
            high.extend(token_space['predicted_prices'].high)
            low.append(0) # position is discrete
            high.append(token_space['position'].n - 1)
            low.extend(token_space['size'].low)
            high.extend(token_space['size'].high)
            low.extend(token_space['entry_price'].low)
            high.extend(token_space['entry_price'].high)
            low.extend(np.array([1]))
            high.extend(np.array([self.env.max_leverage]))
            low.extend(np.array([-1]))
            high.extend(token_space['unrealized_pnl'].high)
            low.extend(token_space['risk_ratio'].low)
            high.extend(token_space['risk_ratio'].high)
            
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32), 
            high=np.array(high, dtype=np.float32), 
            dtype=np.float32
        )

    def observation(self, observation: dict) -> np.ndarray:
        """
        Flattens the observation dictionary into a single numpy array.
        
        Args:
            observation (dict): The original observation from the environment.
        
        Returns:
            np.ndarray: The flattened observation vector.
        """
        flat_obs = []
        flat_obs.extend(observation['balance'])
        flat_obs.extend(observation['equity'])
        
        for token in self.env.tokens:
            token_obs = observation[token]
            flat_obs.extend(token_obs['previous_prices'])
            flat_obs.extend(token_obs['price'])
            flat_obs.extend(token_obs['predicted_prices'])
            flat_obs.append(token_obs['position']) # Append discrete value directly
            flat_obs.extend(token_obs['size'])
            flat_obs.extend(token_obs['entry_price'])
            flat_obs.extend(token_obs['leverage'])
            flat_obs.extend(token_obs['unrealized_pnl'])
            
        return np.array(flat_obs, dtype=np.float32)

def make_wrapped_futures_env(**kwargs):
    """Factory function to create and wrap the environment."""
    env = PerpetualFuturesEnv(**kwargs)
    env = FlattenedObservationWrapper(env)
    env = FlattenedActionWrapper(env)
    return env