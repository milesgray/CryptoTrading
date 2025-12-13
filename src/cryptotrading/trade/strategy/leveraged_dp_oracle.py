"""
Leveraged Dynamic Programming Oracle.

This module solves for the GLOBALLY optimal trading path by analyzing 
discrete trade segments and optimizing for Safe Leverage.

Unlike standard DP which optimizes raw price delta, this oracle optimizes:
    maximize: Log(Final_Equity)
    subject to: Liquidation_Risk < Threshold
"""

import numpy as np
from typing import Tuple, List, Optional
from .util import OracleAction, OracleTradeSegment
import logging

logger = logging.getLogger(__name__)


class LeveragedDPOracleStrategy:
    def __init__(
        self,
        max_holding_period: int = 1000,
        max_leverage: float = 20.0,
        transaction_cost: float = 0.0005, # 0.05% per side
        risk_buffer: float = 0.20         # Liquidate at 80% loss (leave 20% buffer)
    ):
        self.max_window = max_holding_period
        self.max_lev = max_leverage
        self.fee = transaction_cost
        self.risk_buffer = risk_buffer

    def compute_oracle_actions(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal actions and leverage.
        """
        # 1. Solve the Segment-Based DP
        segments = self._solve_segment_dp(prices)
        
        # 2. Convert Segments to dense arrays (Actions, Leverage)
        actions = np.zeros(len(prices), dtype=np.int64)
        leverages = np.ones(len(prices), dtype=np.float32)
        
        for seg in segments:
            # Set Entry Action
            if seg.direction == 1:
                actions[seg.start_idx] = OracleAction.LONG
            else:
                actions[seg.start_idx] = OracleAction.SHORT
            
            # Set Close Action
            if seg.end_idx < len(prices):
                actions[seg.end_idx] = OracleAction.CLOSE
            
            # Fill Holds
            # Note: We don't overwrite the entry/exit, just the middle
            if seg.end_idx > seg.start_idx + 1:
                actions[seg.start_idx + 1 : seg.end_idx] = OracleAction.HOLD
            
            # Set Leverage (applied at entry, effectively constant for the trade)
            leverages[seg.start_idx] = seg.leverage

        return actions, leverages

    def extract_trade_segments(
        self, 
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[OracleTradeSegment]:
        """
        Wrapper to return the segments directly.
        """
        return self._solve_segment_dp(prices)

    def _solve_segment_dp(self, prices: np.ndarray) -> List[OracleTradeSegment]:
        """
        Solves optimal path using Segment-Based Dynamic Programming.
        
        dp[t] = Max Log Wealth achievable by time t.
        We iterate forward. At each 'start_time' i, we look ahead up to 'max_window' steps
        to find potential 'end_time' j.
        
        We update: dp[j] = max(dp[j], dp[i] + log_return_of_segment(i, j))
        """
        n = len(prices)
        
        # dp[t] stores max log wealth at time t
        # parent[t] stores the index 'i' that we came from to get to 't'
        # meta[t] stores tuple (direction, leverage) of the trade ending at t
        dp = np.zeros(n)
        parent = np.full(n, -1, dtype=int)
        meta = np.zeros((n, 2)) # [direction, leverage]
        
        # We process 'i' as the START of a potential trade (or the end of previous)
        # We propagate values forward.
        
        # Optimization: Pre-calculate logs? No, prices change too much with leverage.
        
        for i in range(n - 1):
            # 1. Option: Stay Flat from i to i+1
            # If we are richer at i than whatever path led to i+1 so far, update i+1
            if dp[i] > dp[i+1] or parent[i+1] == -1:
                dp[i+1] = dp[i]
                parent[i+1] = i
                meta[i+1] = [0, 0] # 0 direction (flat)

            # 2. Option: Open a trade at i, close at j
            # We scan j from i+1 to i+max_window
            
            # Init trackers for min/max price in window to calculate leverage efficiently
            window_min = prices[i]
            window_max = prices[i]
            entry_price = prices[i]
            
            max_scan = min(n, i + self.max_window)
            
            for j in range(i + 1, max_scan):
                curr_price = prices[j]
                
                # Update Min/Max for Drawdown calc
                if curr_price < window_min:
                    window_min = curr_price
                if curr_price > window_max:
                    window_max = curr_price
                
                # --- Evaluate LONG (i -> j) ---
                # Drawdown is based on lowest price seen during hold
                dd_long = (entry_price - window_min) / entry_price
                max_adv_long = max(dd_long, 0.0001)
                
                # Safe Leverage Calculation
                # We want: leverage * max_adverse <= (1 - risk_buffer)
                # e.g. lev * 0.05 <= 0.8  -> lev <= 16
                lev_long = (1.0 - self.risk_buffer) / max_adv_long
                lev_long = min(max(lev_long, 1.0), self.max_lev)
                
                # Calculate Final ROE
                raw_ret_long = (curr_price - entry_price) / entry_price
                # Cost: Open Fee + Close Fee (approximate close fee on projected capital)
                # Accurate: wealth * (1 + (raw*lev) - fee*lev - fee*lev)
                roe_long = (raw_ret_long * lev_long) - (self.fee * lev_long * 2)
                
                if roe_long > -0.99: # Avoid log domain error
                    wealth_long = dp[i] + np.log1p(roe_long)
                    if wealth_long > dp[j]:
                        dp[j] = wealth_long
                        parent[j] = i
                        meta[j] = [1, lev_long]

                # --- Evaluate SHORT (i -> j) ---
                # Drawdown is based on highest price seen during hold
                dd_short = (window_max - entry_price) / entry_price
                max_adv_short = max(dd_short, 0.0001)
                
                lev_short = (1.0 - self.risk_buffer) / max_adv_short
                lev_short = min(max(lev_short, 1.0), self.max_lev)
                
                raw_ret_short = (entry_price - curr_price) / entry_price
                roe_short = (raw_ret_short * lev_short) - (self.fee * lev_short * 2)
                
                if roe_short > -0.99:
                    wealth_short = dp[i] + np.log1p(roe_short)
                    if wealth_short > dp[j]:
                        dp[j] = wealth_short
                        parent[j] = i
                        meta[j] = [-1, lev_short]

        # 3. Backtrack to find optimal path
        segments = []
        curr = n - 1
        
        # If end of array wasn't reached (unlikely), backtrack from max
        if parent[curr] == -1:
            curr = np.argmax(dp)
            
        while curr > 0:
            prev = parent[curr]
            if prev == -1: # Should not happen if initialized correctly
                break
                
            direction = int(meta[curr][0])
            lev = meta[curr][1]
            
            if direction != 0:
                # Calculate stats for the segment object
                start_p = prices[prev]
                end_p = prices[curr]
                
                # Re-calculate segment metrics for the record
                # (We could cache these, but re-calc is cheap)
                seg_prices = prices[prev : curr + 1]
                
                if direction == 1:
                    raw_p = (end_p - start_p) / start_p
                    mae = (start_p - np.min(seg_prices)) / start_p
                else:
                    raw_p = (start_p - end_p) / start_p
                    mae = (np.max(seg_prices) - start_p) / start_p
                
                roe = (raw_p * lev) - (self.fee * lev * 2)
                
                segments.append(OracleTradeSegment(
                    start_idx=prev,
                    end_idx=curr,
                    direction=direction,
                    entry_price=start_p,
                    exit_price=end_p,
                    profit_pct=raw_p,
                    roe_pct=roe,
                    leverage=lev,
                    max_adverse_excursion=max(mae, 0.0)
                ))
            
            curr = prev
            
        segments.reverse() # Backtrack gives reverse order
        return segments

    def get_statistics(self, prices: np.ndarray) -> dict:
        segments = self.extract_trade_segments(prices)
        
        if not segments:
            return {'num_trades': 0, 'total_return': 0.0}
        
        roes = [s.roe_pct for s in segments]
        raw_rets = [s.profit_pct for s in segments]
        levs = [s.leverage for s in segments]
        
        # Simulating compounding wealth
        wealth = 1.0
        for r in roes:
            wealth *= (1 + r)
            
        return {
            'num_trades': len(segments),
            'final_wealth_multiplier': wealth,
            'avg_leverage': np.mean(levs),
            'avg_roe': np.mean(roes),
            'avg_raw_return': np.mean(raw_rets),
            'win_rate': sum(1 for r in roes if r > 0) / len(roes),
            'best_trade_roe': max(roes),
            'worst_trade_roe': min(roes)
        }

