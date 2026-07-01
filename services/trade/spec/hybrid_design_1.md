# Hybrid Design 1: The Compulsive-Phlegmatic Trading System

## 1. Strategy Overview

### Personality & Risk Profile
- **Hybrid of**: The Compulsive (Chaotic Socratic) + Phlegmatic Trading System.
- **Motto**: *"High conviction, but never reckless."*
- **Risk Profile**: High risk, high reward, but with **strict risk management** to survive drawdowns.
- **Trading Style**: Contrarian breakout trading with **patient position building** and **diversified risk**.

### Core Principles
1. **Anti-Consensus Betting**: Fade extreme moves (e.g., 90%+ probability events) with **high-conviction contrarian bets**.
2. **Edge Detection**: Exploit liquidity traps, false breakouts, and reversals using **real-time order book analysis**.
3. **Patient Execution**: Build positions **gradually** to avoid slippage and reduce market impact.
4. **Strict Risk Management**: Enforce **drawdown limits** and **diversification** to survive black swan events.

---

## 2. Data Requirements

### High-Frequency Market Data
| Data Type               | Source          | Frequency      | Use Case                          |
|------------------------|-----------------|----------------|-----------------------------------|
| Order Book Snapshots   | Polymarket API  | 100ms - 500ms  | Liquidity analysis, stop-loss detection |
| Trade History          | Polymarket API  | Real-time      | Volume analysis, momentum signals |
| Price Feeds            | Polymarket API  | Real-time      | Breakout/reversal detection       |
| Probability Feeds      | Polymarket API  | Real-time      | Contrarian signal generation      |

### Volatility Metrics
| Metric                     | Calculation                          | Purpose                          |
|---------------------------|--------------------------------------|-----------------------------------|
| Implied Volatility (IV)   | Derived from option-like payoffs     | Identify mispriced events        |
| Historical Volatility (HV)| 20-period rolling std. deviation     | Normalize position sizing        |
| Volume-Weighted Volatility| HV adjusted for trading volume       | Filter low-conviction moves      |

### Alternative Data (Optional)
| Data Type       | Source               | Use Case                          |
|----------------|----------------------|-----------------------------------|
| Social Sentiment| Twitter, Discord     | Contrarian sentiment signals      |
| News Feeds      | RSS, NewsAPI         | Event-driven volatility spikes    |

---

## 3. Edge Detection

### Liquidity Traps
- **Definition**: Areas where stop-losses or limit orders are clustered, creating artificial support/resistance.
- **Detection Methods**:
  1. **Stop-Loss Heatmaps**: Aggregate stop-loss orders from order book data.
  2. **Volume Imbalance**: Compare bid/ask volume at key levels.
  3. **Price Action Confirmation**: If price wicks into a liquidity zone but fails to hold, it’s a trap.

### False Breakouts
- **Definition**: Breakouts with low volume or manipulative intent, likely to reverse.
- **Detection Methods**:
  1. **Volume Filter**: Breakout is invalid if volume < 50% of 20-period average.
  2. **Order Book Depth**: If breakout side has < 2x depth of the opposite side, it’s suspect.

### Reversals
- **Definition**: Points where the market exhausts its momentum and reverses.
- **Detection Methods**:
  1. **Overbought/Oversold Indicators**: RSI > 70 (overbought), RSI < 30 (oversold).
  2. **Volume Climax**: If volume spikes > 3x average, expect a reversal.

---

## 4. Execution Logic

### Position Sizing
| Conviction Level | Position Size | Max Risk per Trade |
|------------------|---------------|---------------------|
| Low (1/5)        | 0.5% of capital | 0.25%              |
| Medium (3/5)     | 2% of capital  | 1%                 |
| High (5/5)       | 5% of capital  | 2.5%               |

- **Dynamic Scaling**:
  - Position size inversely proportional to historical volatility (HV).
  - Reduce size if order book depth < 2x average.

### Stop-Loss Placement
| Strategy               | Stop-Loss Method               | Trailing? |
|------------------------|---------------------------------|-----------|
| Contrarian Fade        | 2x ATR from entry               | Yes       |
| Breakout Hunt          | Below breakout low (for long)   | No        |
| Liquidity Trap Scalping| 1 ATR from entry                | No        |

### Trade Execution
- **Latency**: < 100ms from signal to execution.
- **Slippage Control**: Reject trades if estimated slippage > 0.5%.
- **Order Types**:
  - **Market Orders**: For scalping (speed > precision).
  - **Limit Orders**: For contrarian fades (wait for better fill).

---

## 5. Risk Management

### Drawdown Limits
| Metric               | Hard Limit | Soft Limit (Alert) |
|---------------------|------------|--------------------|
| Max Drawdown        | 20%        | 10%                |
| Daily Loss          | 5%         | 3%                 |
| Single Trade Loss   | 2.5%       | 1%                 |

- **Actions on Breach**:
  - **Soft Limit**: Reduce position sizes by 50%.
  - **Hard Limit**: Full stop + manual review required.

### Diversification
- **Max Exposure per Market**: 20% of capital.
- **Max Exposure per Strategy**: 30% of capital.

### Tail-Risk Hedging
- **Inverse Bets**: If long on Event A, short Event B (negatively correlated).
- **Volatility Hedging**: Buy out-of-the-money (OTM) options on extreme outcomes.

---

## 6. Socratic Questions

### Market Assumption Challenges
| Question                                                  | Purpose                          | Example Trade                     |
|-----------------------------------------------------------|-----------------------------------|-----------------------------------|
| *"What if this ‘support’ is a liquidity trap?"*          | Challenge technical levels       | Short at "support" if stops are clustered below. |
| *"What if the crowd is wrong about this 90% probability?"* | Contrarian fade               | Bet on the 10% underdog.          |

### Strategy Self-Critique
| Question                                                  | Purpose                          | Action                            |
|-----------------------------------------------------------|-----------------------------------|-----------------------------------|
| *"Is this feature necessary, or does it add fragility?"* | Simplify design                 | Remove redundant indicators.      |
| *"What is the simplest way to achieve this?"*             | Avoid over-engineering          | Use basic volume filters.         |

---

## 7. Performance Targets
| Metric               | Target       | Notes                          |
|---------------------|--------------|--------------------------------|
| Annualized Return   | 80-150%      | High volatility expected.      |
| Sharpe Ratio        | > 1.2        | Risk-adjusted returns.         |
| Max Drawdown        | < 20%        | Aggressive but controlled.     |
| Win Rate            | 40-50%       | Low win rate, high R:R.        |

---

## 8. Risks & Mitigations
| Risk                  | Impact               | Mitigation                     |
|-----------------------|----------------------|--------------------------------|
| API Downtime         | Missed trades        | Fallback to cached data.       |
| Slippage             | Reduced profits      | Limit order size.              |
| Black Swan Event     | Large drawdown       | Tail-risk hedging.             |

---

## 9. Conclusion
The **Compulsive-Phlegmatic Hybrid** combines the **high-conviction, anti-consensus edge** of The Compulsive with the **patient, risk-averse discipline** of the Phlegmatic system. By integrating strict drawdown limits and diversification, this hybrid aims to **survive black swan events** while capitalizing on **asymmetric opportunities** in Polymarket’s prediction markets.