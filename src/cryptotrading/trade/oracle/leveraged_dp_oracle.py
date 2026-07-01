"""
Leveraged Dynamic Programming Oracle (v2).

Solves for the globally optimal trading path (maximize Log(Final_Equity)
subject to a liquidation-risk buffer), exactly like v1 -- but:

  1. The O(n * max_window) inner loop is numba-JIT'd, so it's feasible on
     multi-million-tick histories (v1's pure-Python nested loop is not).
  2. It also emits a dense "opportunity map": for every timestep, the best
     forward long/short ROE, leverage, and exit distance found within the
     lookahead window. The single DP backbone is only one thread through
     that space -- for training a policy (esp. for contrastive / pattern-
     matching setups) the opportunity map is a much richer label than the
     one globally-optimal, mutually-exclusive path.
  3. Guards against the ndarray-truthiness bug (`if not segments`), zero /
     NaN prices, and log-domain blowups. Optional funding drag on long
     holds.

Drop-in replacement: same public surface as v1 (`compute_oracle_actions`,
`extract_trade_segments`, `get_statistics`), plus new
`compute_opportunity_map` / `extract_significant_opportunities`.
"""

import numpy as np
from typing import Tuple, List, Optional
from .util import OracleAction, OracleTradeSegment
import logging

logger = logging.getLogger(__name__)

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    logger.warning(
        "numba not installed -- falling back to pure Python DP core. "
        "This will be extremely slow (potentially hours) on multi-million "
        "tick histories. `pip install numba` is strongly recommended."
    )


# ---------------------------------------------------------------------------
# Core numeric kernel (numba-compiled when available)
# ---------------------------------------------------------------------------

def _make_dp_core():
    def _dp_core(prices, max_window, max_lev, fee, risk_buffer, funding_rate_per_step):
        n = prices.shape[0]

        # --- global DP state (single best non-overlapping partition) ---
        dp = np.zeros(n, dtype=np.float64)
        parent = np.full(n, -1, dtype=np.int64)
        direction_arr = np.zeros(n, dtype=np.int64)   # 0 flat, 1 long, -1 short
        leverage_arr = np.zeros(n, dtype=np.float64)

        # --- dense opportunity map (best forward trade FROM each i) ---
        best_long_roe = np.zeros(n, dtype=np.float64)
        best_long_lev = np.ones(n, dtype=np.float64)
        best_long_exit = np.zeros(n, dtype=np.int64)
        best_short_roe = np.zeros(n, dtype=np.float64)
        best_short_lev = np.ones(n, dtype=np.float64)
        best_short_exit = np.zeros(n, dtype=np.int64)

        two_fee = 2.0 * fee
        one_minus_risk = 1.0 - risk_buffer

        for i in range(n - 1):
            entry_price = prices[i]
            if np.isnan(entry_price) or entry_price <= 0.0:
                # bad tick -- can't safely size a position off it
                if dp[i] > dp[i + 1] or parent[i + 1] == -1:
                    dp[i + 1] = dp[i]
                    parent[i + 1] = i
                    direction_arr[i + 1] = 0
                    leverage_arr[i + 1] = 0.0
                continue

            # Option: stay flat i -> i+1
            if dp[i] > dp[i + 1] or parent[i + 1] == -1:
                dp[i + 1] = dp[i]
                parent[i + 1] = i
                direction_arr[i + 1] = 0
                leverage_arr[i + 1] = 0.0

            window_min = entry_price
            window_max = entry_price
            max_scan = min(n, i + max_window)

            cur_best_long_roe = -1.0
            cur_best_long_lev = 1.0
            cur_best_long_exit = 0
            cur_best_short_roe = -1.0
            cur_best_short_lev = 1.0
            cur_best_short_exit = 0

            inv_entry = 1.0 / entry_price
            funding_drag = 0.0

            for j in range(i + 1, max_scan):
                funding_drag += funding_rate_per_step
                curr_price = prices[j]
                if np.isnan(curr_price) or curr_price <= 0.0:
                    continue
                if curr_price < window_min:
                    window_min = curr_price
                if curr_price > window_max:
                    window_max = curr_price

                hold_steps = j - i

                # ---- LONG (i -> j) ----
                dd_long = (entry_price - window_min) * inv_entry
                max_adv_long = max(dd_long, 1e-4)
                lev_long = one_minus_risk / max_adv_long
                if lev_long > max_lev:
                    lev_long = max_lev
                if lev_long < 1.0:
                    lev_long = 1.0

                raw_ret_long = (curr_price - entry_price) * inv_entry
                roe_long = lev_long * (raw_ret_long - two_fee - funding_drag)

                if roe_long > -0.99:
                    wealth_long = dp[i] + np.log1p(roe_long)
                    if wealth_long > dp[j]:
                        dp[j] = wealth_long
                        parent[j] = i
                        direction_arr[j] = 1
                        leverage_arr[j] = lev_long
                    if roe_long > cur_best_long_roe:
                        cur_best_long_roe = roe_long
                        cur_best_long_lev = lev_long
                        cur_best_long_exit = hold_steps

                # ---- SHORT (i -> j) ----
                dd_short = (window_max - entry_price) * inv_entry
                max_adv_short = max(dd_short, 1e-4)
                lev_short = one_minus_risk / max_adv_short
                if lev_short > max_lev:
                    lev_short = max_lev
                if lev_short < 1.0:
                    lev_short = 1.0

                raw_ret_short = -raw_ret_long
                roe_short = lev_short * (raw_ret_short - two_fee - funding_drag)

                if roe_short > -0.99:
                    wealth_short = dp[i] + np.log1p(roe_short)
                    if wealth_short > dp[j]:
                        dp[j] = wealth_short
                        parent[j] = i
                        direction_arr[j] = -1
                        leverage_arr[j] = lev_short
                    if roe_short > cur_best_short_roe:
                        cur_best_short_roe = roe_short
                        cur_best_short_lev = lev_short
                        cur_best_short_exit = hold_steps

            best_long_roe[i] = cur_best_long_roe
            best_long_lev[i] = cur_best_long_lev
            best_long_exit[i] = cur_best_long_exit
            best_short_roe[i] = cur_best_short_roe
            best_short_lev[i] = cur_best_short_lev
            best_short_exit[i] = cur_best_short_exit

        return (dp, parent, direction_arr, leverage_arr,
                best_long_roe, best_long_lev, best_long_exit,
                best_short_roe, best_short_lev, best_short_exit)

    if _NUMBA_AVAILABLE:
        return njit(cache=True, fastmath=True)(_dp_core)
    return _dp_core


_dp_core = _make_dp_core()


class LeveragedDPOracle:
    def __init__(
        self,
        max_holding_period: int = 1000,
        max_leverage: float = 20.0,
        transaction_cost: float = 0.0005,   # 0.05% per side
        risk_buffer: float = 0.20,          # liquidate at 80% loss (20% buffer)
        funding_rate_per_step: float = 0.0, # e.g. avg |funding| per bar, applied * leverage * hold_steps
    ):
        self.max_window = max_holding_period
        self.max_lev = max_leverage
        self.fee = transaction_cost
        self.risk_buffer = risk_buffer
        self.funding_rate_per_step = funding_rate_per_step
        self._last_result = None
        self._last_prices_id = None
        self._last_input_prices = None
        self._last_contig_prices = None

    # ------------------------------------------------------------------
    # Internal: run (and cache) the core scan
    # ------------------------------------------------------------------
    def _run_core(self, prices: np.ndarray):
        if self._last_input_prices is prices and self._last_result is not None:
            return self._last_result

        prices_contig = np.ascontiguousarray(prices, dtype=np.float64)
        if prices_contig.ndim != 1:
            raise ValueError(f"prices must be 1-D, got shape {prices_contig.shape}")
        if len(prices_contig) < 2:
            raise ValueError("prices must have at least 2 points")

        key = (id(prices_contig), prices_contig.shape[0], float(prices_contig[0]), float(prices_contig[-1]))
        if self._last_prices_id == key and self._last_result is not None:
            self._last_input_prices = prices
            return self._last_result

        result = _dp_core(
            prices_contig, self.max_window, self.max_lev, self.fee,
            self.risk_buffer, self.funding_rate_per_step,
        )
        self._last_input_prices = prices
        self._last_contig_prices = prices_contig
        self._last_prices_id = key
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Public API (compatible with v1)
    # ------------------------------------------------------------------
    def compute_oracle_actions(
        self,
        prices: np.ndarray,
        segments: Optional[List[OracleTradeSegment]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dense per-tick (action, leverage) arrays for the DP backbone."""
        if segments is None:
            segments = self._solve_segment_dp(prices)

        n = len(prices)
        actions = np.zeros(n, dtype=np.int64)
        leverages = np.ones(n, dtype=np.float32)

        for seg in segments:
            actions[seg.start_idx] = OracleAction.LONG if seg.direction == 1 else OracleAction.SHORT
            if seg.end_idx < n:
                actions[seg.end_idx] = OracleAction.CLOSE
            if seg.end_idx > seg.start_idx + 1:
                actions[seg.start_idx + 1: seg.end_idx] = OracleAction.HOLD
            leverages[seg.start_idx] = seg.leverage

        return actions, leverages

    def extract_trade_segments(
        self,
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> List[OracleTradeSegment]:
        return self._solve_segment_dp(prices)

    def _solve_segment_dp(self, prices: np.ndarray) -> List[OracleTradeSegment]:
        prices = np.ascontiguousarray(prices, dtype=np.float64)
        n = len(prices)
        (dp, parent, direction_arr, leverage_arr, *_rest) = self._run_core(prices)

        segments = []
        curr = n - 1
        if parent[curr] == -1:
            curr = int(np.argmax(dp))

        while curr > 0:
            prev = parent[curr]
            if prev == -1:
                break

            direction = int(direction_arr[curr])
            lev = float(leverage_arr[curr])

            if direction != 0:
                start_p = float(prices[prev])
                end_p = float(prices[curr])
                seg_prices = prices[prev: curr + 1]

                if direction == 1:
                    raw_p = (end_p - start_p) / start_p
                    mae = (start_p - np.min(seg_prices)) / start_p
                else:
                    raw_p = (start_p - end_p) / start_p
                    mae = (np.max(seg_prices) - start_p) / start_p

                roe = raw_p * lev - self.fee * lev * 2.0

                segments.append(OracleTradeSegment(
                    start_idx=int(prev),
                    end_idx=int(curr),
                    direction=direction,
                    entry_price=start_p,
                    exit_price=end_p,
                    profit_pct=raw_p,
                    roe_pct=roe,
                    leverage=lev,
                    max_adverse_excursion=max(mae, 0.0),
                ))

            curr = prev

        segments.reverse()
        return segments

    def get_statistics(
        self,
        prices: Optional[np.ndarray] = None,
        segments: Optional[List[OracleTradeSegment]] = None,
    ) -> dict:
        if segments is None:
            segments = self.extract_trade_segments(prices)

        if not segments:
            return {'num_trades': 0, 'total_return': 0.0}

        roes = [s.roe_pct for s in segments]
        raw_rets = [s.profit_pct for s in segments]
        levs = [s.leverage for s in segments]

        wealth = 1.0
        for r in roes:
            wealth *= (1.0 + r)

        return {
            'num_trades': len(segments),
            'final_wealth_multiplier': wealth,
            'avg_leverage': float(np.mean(levs)),
            'avg_roe': float(np.mean(roes)),
            'avg_raw_return': float(np.mean(raw_rets)),
            'win_rate': sum(1 for r in roes if r > 0) / len(roes),
            'best_trade_roe': max(roes),
            'worst_trade_roe': min(roes),
        }

    # ------------------------------------------------------------------
    # New: dense opportunity map + significant-opportunity extraction
    # ------------------------------------------------------------------
    def compute_opportunity_map(self, prices: np.ndarray) -> dict:
        """
        For every index i, the best forward long and short trade found
        within [i, i+max_window), independent of whether the global DP
        partition actually took it. O(n) memory, computed as a byproduct
        of the same O(n*max_window) scan used for the DP backbone.

        This is the right signal for "all possible good trades" -- e.g.
        contrastive pattern-matching (similar setups -> similar
        opportunity vectors) or as a dense auxiliary regression target,
        as opposed to the single mutually-exclusive DP path.
        """
        prices = np.ascontiguousarray(prices, dtype=np.float64)
        (*_dp_stuff,
         best_long_roe, best_long_lev, best_long_exit,
         best_short_roe, best_short_lev, best_short_exit) = self._run_core(prices)

        return {
            'long_roe': best_long_roe,
            'long_leverage': best_long_lev,
            'long_exit_offset': best_long_exit,
            'short_roe': best_short_roe,
            'short_leverage': best_short_lev,
            'short_exit_offset': best_short_exit,
        }

    def extract_significant_opportunities(
        self,
        prices: np.ndarray,
        min_roe: float = 0.02,
        max_overlap_fraction: float = 1.0,
    ) -> List[OracleTradeSegment]:
        """
        All (not just the single optimal partition) local best-opportunity
        segments whose ROE clears `min_roe`, one per start index, sorted
        by start_idx. These *do* overlap in time with each other (that's
        the point -- it's the full candidate set, not a partition), which
        is what you want for contrastive positive/negative mining rather
        than for direct behavior-cloning labels.

        `max_overlap_fraction` < 1.0 does simple greedy non-max suppression
        (keep the highest-ROE segment, drop others overlapping it by more
        than that fraction) if you want a de-duplicated, still-diverse set.
        """
        opp = self.compute_opportunity_map(prices)
        n = len(prices)
        candidates: List[OracleTradeSegment] = []

        for i in range(n - 1):
            for direction, roe_arr, lev_arr, exit_arr in (
                (1, opp['long_roe'], opp['long_leverage'], opp['long_exit_offset']),
                (-1, opp['short_roe'], opp['short_leverage'], opp['short_exit_offset']),
            ):
                roe = roe_arr[i]
                if roe < min_roe:
                    continue
                exit_offset = int(exit_arr[i])
                if exit_offset <= 0:
                    continue
                j = i + exit_offset
                if j >= n:
                    continue

                start_p, end_p = float(prices[i]), float(prices[j])
                raw_p = (end_p - start_p) / start_p if direction == 1 else (start_p - end_p) / start_p
                seg_prices = prices[i: j + 1]
                mae = (
                    (start_p - np.min(seg_prices)) / start_p if direction == 1
                    else (np.max(seg_prices) - start_p) / start_p
                )

                candidates.append(OracleTradeSegment(
                    start_idx=i,
                    end_idx=j,
                    direction=direction,
                    entry_price=start_p,
                    exit_price=end_p,
                    profit_pct=raw_p,
                    roe_pct=float(roe),
                    leverage=float(lev_arr[i]),
                    max_adverse_excursion=max(float(mae), 0.0),
                ))

        candidates.sort(key=lambda s: s.roe_pct, reverse=True)

        if max_overlap_fraction >= 1.0:
            candidates.sort(key=lambda s: s.start_idx)
            return candidates

        # Greedy NMS on ROE, keeping temporal diversity.
        kept: List[OracleTradeSegment] = []
        for cand in candidates:
            overlaps_too_much = False
            for k in kept:
                lo = max(cand.start_idx, k.start_idx)
                hi = min(cand.end_idx, k.end_idx)
                overlap = max(0, hi - lo)
                span = min(cand.end_idx - cand.start_idx, k.end_idx - k.start_idx)
                if span > 0 and overlap / span > max_overlap_fraction:
                    overlaps_too_much = True
                    break
            if not overlaps_too_much:
                kept.append(cand)

        kept.sort(key=lambda s: s.start_idx)
        return kept