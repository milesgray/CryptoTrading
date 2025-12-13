"""
Dynamic Programming Oracle for optimal trading path computation.

This module solves for the GLOBALLY optimal trading path using Dynamic Programming,
providing mathematically perfect labels for contrastive learning.
"""

import numpy as np
from typing import Tuple, List, Optional
from .util import OracleAction, OracleTradeSegment

import logging

logger = logging.getLogger(__name__)



class DPOracleStrategy:
    """
    Solves for the GLOBALLY optimal trading path using Dynamic Programming.
    This replaces the greedy heuristic with a mathematically perfect teacher.
    """
    
    def __init__(
        self,
        lookahead: int = 100,  # Used for leverage calculation safety
        max_leverage: float = 20.0,
        transaction_cost: float = 0.001
    ):
        self.lookahead = lookahead
        self.max_leverage = max_leverage
        self.fee = transaction_cost
    
    def compute_oracle_actions(
        self, 
        prices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal actions and leverage.
        
        1. Solve Viterbi path for State (Long/Short/Flat).
        2. Calculate max safe leverage for the identified intervals.
        
        Args:
            prices: Array of prices
            
        Returns:
            actions: Array of OracleAction values
            leverages: Array of leverage values
        """
        # 1. Compute Optimal States (Flat, Long, Short)
        states = self._solve_dp(prices)
        
        # 2. Convert States to Actions
        actions = self._states_to_actions(states)
        
        # 3. Compute Safe Leverage for the chosen path
        leverages = self._compute_safe_leverage(prices, states, actions)
        
        return actions, leverages

    def _solve_dp(self, prices: np.ndarray) -> np.ndarray:
        """
        DP State definition: 0=Flat, 1=Long, 2=Short
        Objective: Maximize Log Wealth.
        
        Args:
            prices: Price array
            
        Returns:
            Array of optimal states (-1=short, 0=flat, 1=long)
        """
        n = len(prices)
        # dp[t][state] = max log_wealth ending in this state at time t
        dp = np.full((n, 3), -np.inf)
        path = np.zeros((n, 3), dtype=int)
        
        # Initialize: can only start flat
        dp[0][0] = 0.0
        # Starting in a position means we entered at t=0
        dp[0][1] = -self.fee  # Pay fee to enter long
        dp[0][2] = -self.fee  # Pay fee to enter short
        
        log_prices = np.log(prices + 1e-8)
        
        for t in range(1, n):
            # Log return from t-1 to t
            r = log_prices[t] - log_prices[t-1]
            
            # --- To State 0 (Flat) at time t ---
            # From Flat at t-1: stay flat (no cost, no return)
            v_from_flat = dp[t-1][0]
            # From Long at t-1: close at t (get the return from t-1 to t, then pay fee)
            v_from_long = dp[t-1][1] + r - self.fee
            # From Short at t-1: close at t (get -return, then pay fee)
            v_from_short = dp[t-1][2] - r - self.fee
            
            vals = [v_from_flat, v_from_long, v_from_short]
            dp[t][0] = max(vals)
            path[t][0] = int(np.argmax(vals))
            
            # --- To State 1 (Long) at time t ---
            # From Flat at t-1: enter long at t-1, get return from t-1 to t
            v_from_flat = dp[t-1][0] - self.fee + r
            # From Long at t-1: hold, get return
            v_from_long = dp[t-1][1] + r
            # From Short at t-1: close short and enter long (2 fees), get return
            v_from_short = dp[t-1][2] - r - 2*self.fee + r  # -r to close short, +r from new long
            
            vals = [v_from_flat, v_from_long, v_from_short]
            dp[t][1] = max(vals)
            path[t][1] = int(np.argmax(vals))
            
            # --- To State 2 (Short) at time t ---
            # From Flat at t-1: enter short at t-1, get -return
            v_from_flat = dp[t-1][0] - self.fee - r
            # From Long at t-1: close long and enter short (2 fees)
            v_from_long = dp[t-1][1] + r - 2*self.fee - r  # +r to close long, -r from new short
            # From Short at t-1: hold, get -return
            v_from_short = dp[t-1][2] - r
            
            vals = [v_from_flat, v_from_long, v_from_short]
            dp[t][2] = max(vals)
            path[t][2] = int(np.argmax(vals))
        
        # Backtrack from the best final state
        optimal_states = np.zeros(n, dtype=int)
        
        # Find best final state (we want to end flat ideally, but pick best)
        final_vals = dp[n-1].copy()
        curr_state = int(np.argmax(final_vals))
        
        # Backtrack
        for t in range(n-1, -1, -1):
            # Map internal state [0,1,2] to output [-1,0,1]
            if curr_state == 0:
                optimal_states[t] = 0  # Flat
            elif curr_state == 1:
                optimal_states[t] = 1  # Long
            else:  # curr_state == 2
                optimal_states[t] = -1  # Short
            
            # Move to previous state
            if t > 0:
                curr_state = path[t][curr_state]
        
        return optimal_states

    def _states_to_actions(self, states: np.ndarray) -> np.ndarray:
        """
        Convert state sequence to action sequence.
        
        Args:
            states: Array of states (-1=short, 0=flat, 1=long)
            
        Returns:
            Array of OracleAction values
        """
        actions = np.zeros(len(states), dtype=np.int64)
        
        for t in range(1, len(states)):
            prev = states[t-1]
            curr = states[t]
            
            if prev == 0 and curr == 1:
                actions[t] = OracleAction.LONG
            elif prev == 0 and curr == -1:
                actions[t] = OracleAction.SHORT
            elif prev != 0 and curr == 0:
                actions[t] = OracleAction.CLOSE
            elif prev == 1 and curr == -1:
                actions[t] = OracleAction.SHORT  # Flip
            elif prev == -1 and curr == 1:
                actions[t] = OracleAction.LONG  # Flip
            else:
                actions[t] = OracleAction.HOLD
                
        return actions

    def _compute_safe_leverage(
        self, 
        prices: np.ndarray, 
        states: np.ndarray, 
        actions: np.ndarray
    ) -> np.ndarray:
        """
        Calculate leverage based on maximum adverse excursion (drawdown) 
        during the holding period defined by the DP.
        
        Args:
            prices: Price array
            states: State array
            actions: Action array
            
        Returns:
            Array of leverage values
        """
        n = len(prices)
        leverages = np.ones(n, dtype=np.float32)
        
        t = 0
        while t < n:
            if states[t] == 0:
                t += 1
                continue
                
            # Found a position, find exit
            start_t = t
            pos_type = states[t]  # 1 or -1
            end_t = t + 1
            while end_t < n and states[end_t] == pos_type:
                end_t += 1
            
            # Analyze this segment
            segment_prices = prices[start_t : min(end_t + 1, n)]
            entry_p = prices[start_t]
            
            if pos_type == 1:  # Long
                # Max drawdown from entry
                min_p = np.min(segment_prices)
                max_adverse = (entry_p - min_p) / entry_p
            else:  # Short
                # Max rise from entry
                max_p = np.max(segment_prices)
                max_adverse = (max_p - entry_p) / entry_p
                
            max_adverse = max(max_adverse, 0.001)
            
            # Safe leverage formula: buffer / drawdown
            # We want 20% equity buffer (liquidate at 80% loss)
            safe_lev = 0.8 / max_adverse
            safe_lev = min(max(safe_lev, 1.0), self.max_leverage)
            
            # Assign to the entry action
            if actions[start_t] in [OracleAction.LONG, OracleAction.SHORT]:
                leverages[start_t] = safe_lev
                
            # Advance
            t = end_t
            
        return leverages
    
    def extract_trade_segments(
        self, 
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[OracleTradeSegment]:
        """
        Extract complete trade segments from oracle solution.
        
        Args:
            prices: Price array
            timestamps: Optional timestamp array
            
        Returns:
            List of OracleTradeSegment objects
        """
        actions, leverages = self.compute_oracle_actions(prices)
        states = self._solve_dp(prices)
        
        segments = []
        n = len(prices)
        t = 0
        
        while t < n:
            if states[t] == 0:
                t += 1
                continue
            
            # Found position start
            start_t = t
            direction = states[t]
            entry_price = prices[t]
            
            # Find end
            end_t = t + 1
            while end_t < n and states[end_t] == direction:
                end_t += 1
            
            end_t = min(end_t, n - 1)
            exit_price = prices[end_t]
            
            # Calculate metrics
            segment_prices = prices[start_t:end_t + 1]
            
            if direction == 1:  # Long
                profit_pct = (exit_price - entry_price) / entry_price
                max_adverse = (entry_price - np.min(segment_prices)) / entry_price
            else:  # Short
                profit_pct = (entry_price - exit_price) / entry_price
                max_adverse = (np.max(segment_prices) - entry_price) / entry_price
            
            segment = OracleTradeSegment(
                start_idx=start_t,
                end_idx=end_t,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                profit_pct=profit_pct,
                leverage=leverages[start_t],
                max_adverse_excursion=max_adverse
            )
            segments.append(segment)
            
            t = end_t + 1
        
        return segments
    
    def get_statistics(self, prices: np.ndarray) -> dict:
        """
        Get statistics about the oracle's performance on price data.
        
        Args:
            prices: Price array
            
        Returns:
            Dict with various statistics
        """
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

