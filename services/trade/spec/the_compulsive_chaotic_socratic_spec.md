# The Compulsive (Chaotic Socratic) - Polymarket Trading System Specification

> **Personality**: Impulsive, high-energy, risk-seeking, contrarian.
> **Motto**: *"What if the opposite of conventional wisdom is true?"*
> **Risk Profile**: High risk, high conviction, anti-consensus.
> **Trading Style**: Breakout trading, tail-risk hedging, short-term scalping.

---

## **1. Strategy Overview**

### **1.1 Philosophy**
The Compulsive (Chaotic Socratic) is designed to **thrive in chaos** by challenging market orthodoxy. It embodies the Socratic method of relentless questioning, flipping conventional wisdom to uncover **asymmetric opportunities** in Polymarket's prediction markets. The system assumes that:
- **Crowds are often wrong** at inflection points.
- **Extreme moves are unsustainable** and prone to violent reversals.
- **Liquidity traps** are deliberate and exploitable.

### **1.2 Core Strategies**
| Strategy | Objective | Timeframe | Risk Level |
|----------|-----------|-----------|------------|
| **Contrarian Fading** | Fade extreme moves (e.g., 90%+ probability events) | 1min - 1hr | Extreme |
| **Breakout Hunting** | Trade false breakouts with low volume | 5min - 30min | High |
| **Liquidity Trap Scalping** | Exploit stop-loss clusters | 1min - 10min | High |
| **Tail-Risk Hedging** | Hedge long positions with inverse bets | 1hr - 1day | Medium |
| **Volatility Arbitrage** | Capitalize on mispriced implied volatility | 1min - 1hr | High |

### **1.3 Key Principles**
1. **Anti-Consensus Betting**: If the market is 90% confident in an outcome, the system **fades it aggressively** (e.g., betting on the 10% underdog).
2. **Volatility as Opportunity**: High volatility = high edge. The system **scales into positions** as volatility spikes.
3. **Liquidity as a Weapon**: Thin order books and stop-loss clusters are **hunting grounds** for scalping.
4. **Socratic Provocation**: Every trade is preceded by a **challenge to the status quo** (see [Section 6](#6-socratic-questions)).

---

## **2. Data Requirements**

### **2.1 High-Frequency Market Data**
| Data Type | Source | Frequency | Use Case |
|-----------|--------|-----------|----------|
| **Order Book Snapshots** | Polymarket API | 100ms - 500ms | Liquidity analysis, stop-loss detection |
| **Trade History** | Polymarket API | Real-time | Volume analysis, momentum signals |
| **Price Feeds** | Polymarket API | Real-time | Breakout/reversal detection |
| **Probability Feeds** | Polymarket API | Real-time | Contrarian signal generation |

### **2.2 Volatility Metrics**
| Metric | Calculation | Purpose |
|--------|-------------|---------|
| **Implied Volatility (IV)** | Derived from option-like payoffs in prediction markets | Identify mispriced events |
| **Historical Volatility (HV)** | 20-period rolling standard deviation of returns | Normalize position sizing |
| **Volume-Weighted Volatility** | HV adjusted for trading volume | Filter low-conviction moves |
| **Volatility Skew** | Difference between IV for "Yes" vs. "No" outcomes | Detect asymmetric risk |

### **2.3 Alternative Data (Optional)**
| Data Type | Source | Use Case |
|-----------|--------|----------|
| **Social Sentiment** | Twitter, Discord, Reddit | Contrarian sentiment signals |
| **News Feeds** | RSS, API (e.g., NewsAPI) | Event-driven volatility spikes |
| **On-Chain Activity** | Blockchain explorers (if applicable) | Detect insider accumulation |

### **2.4 Data Latency Requirements**
- **Order Book**: < 200ms latency for liquidity trap detection.
- **Trade Data**: < 100ms latency for scalping.
- **Volatility Metrics**: Updated every 500ms.

---

## **3. Edge Detection**

### **3.1 Liquidity Traps**
**Definition**: Areas where stop-losses or limit orders are **clustered**, creating artificial support/resistance levels that are likely to break.

**Detection Methods**:
1. **Stop-Loss Heatmaps**:
   - Aggregate stop-loss orders from order book data.
   - Flag levels with **> 3x average stop-loss volume**.
2. **Volume Imbalance**:
   - Compare bid/ask volume at key levels.
   - **Signal**: If bids at "support" are thin but asks are heavy, it’s a **bull trap** (and vice versa).
3. **Price Action Confirmation**:
   - If price **wicks into a liquidity zone** but fails to hold, it’s a trap.

**Example**:
```
If ETH > $3000 has 1000 stop-losses clustered at $2950:
- System **shorts above $3000**, targeting $2950.
- Stops out if $3000 holds for > 5 minutes.
```

### **3.2 False Breakouts**
**Definition**: Breakouts with **low volume** or **manipulative intent**, likely to reverse.

**Detection Methods**:
1. **Volume Filter**:
   - Breakout is **invalid** if volume < 50% of 20-period average.
2. **Order Book Depth**:
   - If the breakout side (e.g., bids for upside breakout) has **< 2x depth** of the opposite side, it’s suspect.
3. **Rejection Candles**:
   - If price **immediately reverses** after breaking a level, it’s a false breakout.

**Example**:
```
If BTC breaks $50k with low volume and shallow bids:
- System **fades the breakout** with a short position.
- Stop-loss: Above the breakout high + 1 ATR.
```

### **3.3 Reversals**
**Definition**: Points where the market **exhausts** its momentum and reverses.

**Detection Methods**:
1. **Overbought/Oversold Indicators**:
   - **RSI (14-period)**: > 70 = overbought, < 30 = oversold.
   - **Bollinger Bands**: Price touching upper/lower band + divergence.
2. **Volume Climax**:
   - **Signal**: If volume spikes **> 3x average** on a move, expect a reversal.
3. **Divergence**:
   - If price makes a **new high** but RSI/momentum does not, it’s a **bearish divergence**.

**Example**:
```
If SOL is at 90% probability on Polymarket with RSI > 80:
- System **bets on the 10% underdog** (contrarian fade).
- Stop-loss: If probability hits 95%.
```

---

## **4. Execution Logic**

### **4.1 Position Sizing**
**Rule**: *"The higher the conviction, the larger the bet."*

| Conviction Level | Position Size | Max Risk per Trade |
|------------------|----------------|---------------------|
| **Low** (1/5) | 0.5% of capital | 0.25% |
| **Medium** (3/5) | 2% of capital | 1% |
| **High** (5/5) | 5% of capital | 2.5% |
| **Extreme** (Socratic Override) | 10% of capital | 5% |

**Dynamic Scaling**:
- **Volatility Adjustment**: Position size **inversely proportional** to HV.
  - `Position Size = Base Size * (Target Volatility / Current HV)`
- **Liquidity Adjustment**: Reduce size if order book depth < 2x average.

### **4.2 Stop-Loss Placement**
**Rule**: *"Cut losses fast, let winners run."*

| Strategy | Stop-Loss Method | Trailing? |
|----------|-------------------|-----------|
| **Contrarian Fade** | 2x ATR from entry | Yes (after 1:1 RR) |
| **Breakout Hunt** | Below breakout low (for long) | No |
| **Liquidity Trap Scalping** | 1 ATR from entry | No |
| **Tail-Risk Hedge** | 3x ATR (wide for macro hedges) | Yes |

**Example**:
```
If fading a 90% probability event:
- Entry: Bet on 10% underdog at 90%.
- Stop-Loss: If probability hits 95% (2.5% move against).
- Target: 5% probability (1:2 risk-reward).
```

### **4.3 Trade Execution**
**Requirements**:
- **Latency**: < 100ms from signal to execution.
- **Slippage Control**: Reject trades if estimated slippage > 0.5%.
- **Order Types**:
  - **Market Orders**: For scalping (speed > precision).
  - **Limit Orders**: For contrarian fades (wait for better fill).

**Execution Flow**:
1. **Signal Generated** (e.g., liquidity trap detected).
2. **Conviction Scored** (1-5, based on edge strength).
3. **Position Sized** (using [4.1](#41-position-sizing)).
4. **Stop-Loss Set** (using [4.2](#42-stop-loss-placement)).
5. **Order Executed** (market/limit based on urgency).
6. **Monitor & Adjust** (trailing stops, take-profit).

---

## **5. Risk Management**

### **5.1 Tail-Risk Hedging**
**Objective**: Protect against **black swan events** while maintaining high-risk core positions.

**Methods**:
1. **Inverse Bets**:
   - If long on Event A, **short Event B** (negatively correlated).
   - Example: Long "Biden wins" + Short "Trump wins".
2. **Volatility Hedging**:
   - Buy **out-of-the-money (OTM) options** on extreme outcomes.
   - Example: If betting on "No" at 80%, buy a small "Yes" lottery ticket.
3. **Dynamic Hedging**:
   - Increase hedge size as **drawdown > 10%**.

### **5.2 Drawdown Limits**
| Metric | Hard Limit | Soft Limit (Alert) |
|--------|------------|-------------------|
| **Max Drawdown** | 30% | 20% |
| **Daily Loss** | 10% | 5% |
| **Single Trade Loss** | 5% | 2.5% |

**Actions on Breach**:
- **Soft Limit**: Reduce position sizes by 50%.
- **Hard Limit**: **Full stop** + manual review required to resume.

### **5.3 Leverage Rules**
- **Max Leverage**: 5x (for high-conviction trades).
- **Average Leverage**: 2-3x.
- **Leverage Reduction**: If HV > 2x average, reduce leverage by 50%.

### **5.4 Circuit Breakers**
- **Market-Wide**: Pause trading if Polymarket API latency > 1s.
- **Strategy-Specific**: Disable a strategy if it loses > 15% in a day.

---

## **6. Socratic Questions**

### **6.1 Market Assumption Challenges**
| Question | Purpose | Example Trade |
|----------|---------|---------------|
| *"What if this ‘support’ is a liquidity trap?"* | Challenge technical levels | Short at "support" if stops are clustered below. |
| *"What if the crowd is wrong about this 90% probability?"* | Contrarian fade | Bet on the 10% underdog. |
| *"What if this breakout is manipulated?"* | Detect false breakouts | Fade low-volume breakouts. |
| *"What if the news is already priced in?"* | Avoid late entries | Skip trades after major announcements. |
| *"What if the opposite trade is the real edge?"* | Reverse engineering | Flip the position if conviction is low. |

### **6.2 Strategy Self-Critique**
| Question | Purpose | Action |
|----------|---------|--------|
| *"What if our stop-loss is too tight?"* | Avoid premature exits | Widen stops for high-conviction trades. |
| *"What if the model is overfitting to past data?"* | Robustness check | Backtest on out-of-sample data. |
| *"What if latency kills the edge?"* | Infrastructure audit | Optimize API calls, use co-location. |
| *"What if the tail risk is underestimated?"* | Stress test | Simulate 2008-like crashes. |
| *"What if the opposite strategy works better?"* | Humility check | Run A/B tests against inverse logic. |

---

## **7. Technical Implementation (Optional)**

### **7.1 Polymarket API Integration**
- **Endpoints**:
  - `GET /markets` (List active markets)
  - `GET /markets/{id}/book` (Order book snapshots)
  - `POST /orders` (Place/cancel orders)
  - `WS /stream` (Real-time price/volume updates)
- **Rate Limits**: 100 requests/second (adjust for latency).
- **Authentication**: API key + HMAC signing.

### **7.2 Backtesting Framework**
- **Data**: Historical Polymarket data (via API or scraped).
- **Metrics**:
  - Sharpe Ratio (target: > 1.5)
  - Max Drawdown (< 30%)
  - Win Rate (> 40%)
  - Profit Factor (> 1.5)
- **Tools**: Python (Backtrader, Zipline), or custom.

### **7.3 Deployment Architecture**
```
┌───────────────────────────────────────────────────────┐
│                 Trading System                          │
├───────────────────┬───────────────────┬───────────────┤
│   Signal Generator │   Risk Manager    │  Execution Engine│
│   (Edge Detection) │   (Drawdown Checks)│  (API Calls)    │
└─────────┬─────────┴─────────┬─────────┴───────┬───────┘
          │                   │                 │
          ▼                   ▼                 ▼
┌───────────────────┐ ┌─────────────┐ ┌─────────────┐
│   Polymarket API   │ │  Database    │ │  Monitoring  │
│   (Real-Time Data) │ │  (Historical)│ │  (Alerts)    │
└───────────────────┘ └─────────────┘ └─────────────┘
```
- **Latency Optimization**:
  - Co-locate servers near Polymarket’s infrastructure.
  - Use WebSockets for real-time data.
  - Cache order book snapshots in-memory.

---

## **8. Example Trade Walkthrough**

### **Scenario**: *"Will Bitcoin hit $100k by EOY?"* (Polymarket Market)
- **Current Probability**: 85% "Yes" (0.85), 15% "No" (0.15).
- **Order Book**: Heavy stop-losses for "No" bets at 20%.
- **Volume**: Low (suggests weak conviction).
- **RSI**: 75 (overbought).

### **Socratic Challenge**:
> *"What if the 85% consensus is a liquidity trap, and the real edge is betting on ‘No’?"*

### **System Action**:
1. **Edge Detected**: Liquidity trap at 20% (stop-loss cluster).
2. **Conviction**: High (5/5) due to:
   - Overbought RSI.
   - Low volume.
   - Stop-loss cluster.
3. **Position Sizing**:
   - Base Size: 5% of capital.
   - Volatility Adjustment: HV is high → reduce to **3% of capital**.
4. **Trade Execution**:
   - **Buy "No" at 15% (0.15)**.
   - **Stop-Loss**: 20% (0.20) (2.5% risk).
   - **Target**: 5% (0.05) (1:3 risk-reward).
5. **Hedge**:
   - Buy a small "Yes" lot at 90% as a tail-risk hedge (1% of capital).

### **Outcome**:
- If Bitcoin **fails to hit $100k**: "No" wins → **12x return** (0.15 → 1.00).
- If Bitcoin **hits $100k**: Stop-loss triggered at 20% → **2.5% loss** (hedge offsets partially).

---

## **9. Performance Targets**
| Metric | Target | Notes |
|--------|--------|-------|
| **Annualized Return** | 100-300% | High volatility expected. |
| **Sharpe Ratio** | > 1.5 | Risk-adjusted returns. |
| **Max Drawdown** | < 30% | Aggressive but controlled. |
| **Win Rate** | 40-50% | Low win rate, high R:R. |
| **Profit Factor** | > 2.0 | Winners > 2x losers. |

---

## **10. Risks & Mitigations**
| Risk | Impact | Mitigation |
|------|--------|------------|
| **API Downtime** | Missed trades | Fallback to cached data + alerts. |
| **Slippage** | Reduced profits | Limit order size, reject high-slippage trades. |
| **Black Swan Event** | Large drawdown | Tail-risk hedging, circuit breakers. |
| **Model Overfit** | Poor live performance | Out-of-sample testing, walk-forward optimization. |
| **Latency Arbitrage** | Front-running | Co-location, optimized code. |

---

## **11. Conclusion**
The Compulsive (Chaotic Socratic) is a **high-octane, contrarian trading system** designed to exploit inefficiencies in Polymarket’s prediction markets. By combining **Socratic questioning**, **edge detection**, and **aggressive execution**, it seeks **asymmetric returns** in volatile, anti-consensus environments.

**Key Takeaways**:
- **Challenge everything**: The market is often wrong at extremes.
- **Speed matters**: Low-latency execution is critical for scalping.
- **Risk is managed, not avoided**: High risk, but with strict drawdown limits.
- **Hedge the tail**: Protect against the unknown with inverse bets.

---

## **Appendix A: Glossary**
| Term | Definition |
|------|------------|
| **ATR** | Average True Range (volatility measure). |
| **IV** | Implied Volatility (market’s expected volatility). |
| **HV** | Historical Volatility (past price volatility). |
| **RSI** | Relative Strength Index (momentum oscillator). |
| **Liquidity Trap** | A price level with clustered stop-losses, likely to break. |
| **False Breakout** | A breakout with low volume, likely to reverse. |

---

## **Appendix B: References**
- [Polymarket API Documentation](https://polymarket.com/developers)
- *The Black Swan* - Nassim Nicholas Taleb (Tail-risk philosophy)
- *The Socratic Method* - Plato (Challenging assumptions)
- *Trading in the Zone* - Mark Douglas (Psychology of trading)

---

> **Final Note**: *"The market can remain irrational longer than you can remain solvent."*
> **But what if irrationality is the edge?**
