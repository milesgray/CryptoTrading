# Plan: Order Book Analytics & Multi-Exchange Metrics

## 1. Goal Description
Check order book analytics, correct validation logic errors, document index pricing formulation, write a comprehensive unit test suite, and add multi-exchange meta-statistics.

## 2. Proposed Changes
- Correct swapped bid/ask filter assignments in `book.py`.
- Enhance mathematical docstrings in `formula.py`.
- Implement price dispersion, cross-exchange arbitrage, index deviation, and HHI concentration metrics in `metrics.py`.
- Add unit tests in `tests/test_order_book_analytics.py`.

---

## 7. Critique & Revisions

### Critique Process
- **Search 1**: `order book aggregation crypto python best practices 2024`
  - Findings: Best practices emphasize using WebSocket streams instead of REST polling to avoid latency, maintaining local states via snapshots and deltas, avoiding pandas in live loops, and using integer/fixed-point arithmetic to avoid floating-point drift.
  - Alignment: Our core index calculations use dictionary-based structures rather than pandas, which aligns with performance recommendations. However, the current manager uses CCXT REST polling (`fetch_order_book`) which is prone to rate-limiting and introduces high latency (not suitable for 500ms real-time updates).
- **Search 2**: `multi-exchange order book arbitrage price dispersion pitfalls`
  - Findings: Arbitrage profits are razor-thin and execution fails due to latency, transaction fees, slippage, and capital requirements. Gaps must be modeled net of fees.
  - Alignment: Our new global arbitrage metric calculates raw spread: `best_bid_A - best_ask_B`. It does not factor in exchange maker/taker fees, transfer costs, or slippage, which can result in false arbitrage signals.
- **Search 3**: `order book HHI concentration calculation`
  - Findings: HHI computed on total depth can be easily manipulated or skewed by deep out-of-the-money resting orders. Depth concentration should be calculated within specific percentage thresholds (e.g. 0.5% or 1% from mid-price).
  - Alignment: Our current HHI calculates concentration using total depth. This is a gap since far-off spoofing orders can skew the HHI.

### Score Before Revision
- **Plan Score**: 8/10
- **Reasoning**: The plan successfully targets and fixes logical bugs, adds high-value metrics, and tests them thoroughly. However, it lacks realistic cost modeling for arbitrage, uses total depth instead of near-mid depth for HHI, and relies on high-latency REST calls for 500ms updates.

### Identified Gaps
- **Gap 1: REST Polling Latency**: Using CCXT REST calls to fetch order books at 500ms intervals is slow and will trigger exchange rate limits quickly. Production requires WebSocket incremental updates.
- **Gap 2: Raw vs Net Arbitrage Spread**: Global arbitrage calculation does not deduct trading fees (maker/taker fees are usually 0.1% to 0.2% combined), which makes many flagged opportunities unprofitable.
- **Gap 3: Total vs Near-Mid Depth Concentration**: Computing HHI on total depth is susceptible to spoofing (large orders placed far from the mid price to skew depth metrics).

### Architectural Pattern Deviation (if any)
- **None**: We did not deviate from the existing microservices and repository design patterns.

### Risk Mitigation & Revisions
1. **Arbitrage Fee Buffer**: We should modify `calculate_multi_exchange_metrics` to optionally support a `fee_rate` parameter (e.g. `0.002` for 0.2%) to deduct transaction costs from the calculated spread: `net_spread = spread - (buy_price * fee_rate + sell_price * fee_rate)`.
2. **HHI Thresholds**: Update the HHI calculation to compute concentration using depth within a tight threshold (e.g. 1%) of the mid-price instead of total depth.
3. **Execution Warning**: Document that the REST polling implementation in `OrderBookManager` is only suitable for simulation/recording under low frequency, and that production execution requires subscribing to WebSocket feeds.
