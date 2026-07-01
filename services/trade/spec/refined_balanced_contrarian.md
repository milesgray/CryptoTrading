# Refined Balanced Contrarian

## 1. Strategy Overview

### Personality & Risk Profile
- **Hybrid of**: Compulsive (Chaotic Socratic) + Phlegmatic.
- **Motto**: *"High conviction, but never reckless."*
- **Risk Profile**: High risk, high reward, with **data-backed validation** and **sentiment-aware filters**.
- **Trading Style**: Contrarian breakout trading with **patient position building** and **dynamic risk management**.

### Core Principles
1. **Anti-Consensus Betting**: Fade extreme moves (e.g., 90%+ probability events) with **high-conviction contrarian bets**.
2. **Statistical Validation**: Contrarian signals must pass **backtesting** before execution.
3. **Sentiment Filter**: Only fade extreme probabilities if **real-time sentiment aligns**.
4. **Dynamic Position Sizing**: Adjust position sizes based on **volatility** and **recent performance**.
5. **Patient Execution**: Build positions **gradually** to avoid slippage and reduce market impact.
6. **Dynamic Leverage**: Adjust leverage based on **recent performance** (e.g., increase after wins, decrease after losses).

---

## 2. Data Requirements

### High-Frequency Market Data
| Data Type               | Source          | Frequency      | Use Case                          |
|------------------------|-----------------|----------------|-----------------------------------|
| Order Book Snapshots   | Polymarket API  | 100ms - 500ms  | Liquidity analysis                |
| Trade History          | Polymarket API  | Real-time      | Volume analysis                   |
| Price Feeds            | Polymarket API  | Real-time      | Breakout/reversal detection       |
| Probability Feeds      | Polymarket API  | Real-time      | Contrarian signal generation      |

### Volatility Metrics
| Metric                     | Calculation                          | Purpose                          |
|---------------------------|--------------------------------------|-----------------------------------|
| Implied Volatility (IV)   | Derived from option-like payoffs     | Identify mispriced events        |
| Historical Volatility (HV)| 20-period rolling std. deviation     | Normalize position sizing        |

### Alternative Data
| Data Type       | Source               | Use Case                          |
|----------------|----------------------|-----------------------------------|
| Social Sentiment| Twitter, Discord     | Sentiment filter for contrarian bets |

---

## 3. Edge Detection

### Liquidity Traps
- **Definition**: Areas where stop-losses or limit orders are clustered, creating artificial support/resistance.
- **Detection Methods**:
  1. **Stop-Loss Heatmaps**: Aggregate stop-loss orders from order book data.
  2. **Volume Imbalance**: Compare bid/ask volume at key levels.

### False Breakouts
- **Definition**: Breakouts with low volume or manipulative intent, likely to reverse.
- **Detection Methods**:
  1. **Volume Filter**: Breakout is invalid if volume < 50% of 20-period average.
  2. **Order Book Depth**: If breakout side has < 2x depth of the opposite side, it’s suspect.

### Sentiment Filter
- **Definition**: Only fade extreme probabilities if sentiment aligns.
- **Methods**:
  1. **NLP Sentiment Scores**: Parse social media for sentiment trends.
  2. **Volume Spikes**: Confirm sentiment with volume spikes.

---

## 4. Execution Logic

### Dynamic Position Sizing
| Conviction Level | Position Size | Max Risk per Trade |
|------------------|---------------|---------------------|
| Low (1/5)        | 0.5% of capital | 0.25%              |
| Medium (3/5)     | 2% of capital  | 1%                 |
| High (5/5)       | 5% of capital  | 2.5%               |

- **Dynamic Scaling**:
  - Position size inversely proportional to historical volatility (HV).
  - Reduce size if order book depth < 2x average.
  - Adjust leverage based on recent performance (e.g., increase after wins).

### Stop-Loss Placement
| Strategy               | Stop-Loss Method               | Trailing? |
|------------------------|---------------------------------|-----------|
| Contrarian Fade        | 2x ATR from entry               | Yes       |
| Breakout Hunt          | Below breakout low (for long)   | No        |

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

## 6. Performance Targets
| Metric               | Target       | Notes                          |
|---------------------|--------------|--------------------------------|
| Annualized Return   | 80-150%      | High volatility expected.      |
| Sharpe Ratio        | > 1.2        | Risk-adjusted returns.         |
| Max Drawdown        | < 20%        | Aggressive but controlled.     |
| Win Rate            | 40-50%       | Low win rate, high R:R.        |

---

## 7. Risks & Mitigations
| Risk                  | Impact               | Mitigation                     |
|-----------------------|----------------------|--------------------------------|
| False Signals         | Poor performance     | Statistical validation + sentiment filter |
| Opportunity Cost      | Missed opportunities  | Dynamic leverage adjustment    |
| Sentiment Noise       | False contrarian bets | Use multiple sentiment sources |

---

## 8. Conclusion
The **Refined Balanced Contrarian** combines the **high-conviction, anti-consensus edge** of The Compulsive with the **patient, risk-averse discipline** of the Phlegmatic system. By incorporating **statistical validation** and **sentiment filters**, this design addresses its original weaknesses while preserving its core strengths. Dynamic leverage and position sizing ensure adaptability to changing market conditions.