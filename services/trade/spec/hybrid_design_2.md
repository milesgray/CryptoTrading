# Hybrid Design 2: The Adaptive-Compulsive Trading System

## 1. Strategy Overview

### Personality & Risk Profile
- **Hybrid of**: Polymarket Trading System (Adaptive) + The Compulsive (Chaotic Socratic).
- **Motto**: *"Adapt or fade."*
- **Risk Profile**: Moderate risk, opportunistic, **narrative-driven with high-conviction execution**.
- **Trading Style**: Momentum and event-driven trading with **real-time adaptability** and **contrarian execution**.

### Core Principles
1. **Adaptability**: Adjust strategies based on **regime shifts** and **sentiment analysis**.
2. **Behavioral Signals**: Use **NLP models** and **volume spikes** to detect market intent.
3. **High-Conviction Execution**: Leverage The Compulsive’s **position sizing** and **stop-loss logic** for efficient trade execution.
4. **Narrative-Driven**: Combine **gut feel** with **data validation** to uncover hidden market narratives.

---

## 2. Data Requirements

### Real-Time Feeds
| Data Type               | Source               | Frequency      | Use Case                          |
|------------------------|----------------------|----------------|-----------------------------------|
| Order Book Dynamics    | Polymarket API       | Real-time      | Liquidity analysis, stop-loss detection |
| News Feeds             | RSS, NewsAPI         | Real-time      | Event-driven trading              |
| Social Media Sentiment | Twitter, Reddit      | Real-time      | Sentiment analysis                |
| Probability Feeds      | Polymarket API       | Real-time      | Regime detection                  |

### Behavioral Data
| Data Type               | Source               | Use Case                          |
|------------------------|----------------------|-----------------------------------|
| Volume Spikes          | Polymarket API       | Momentum detection                |
| Liquidity Patterns     | Polymarket API       | False breakout detection          |
| NLP Sentiment Scores   | Custom NLP Models    | Narrative-driven trading          |

---

## 3. Adaptability Framework

### Regime Detection
- **Definition**: Identify changes in market behavior (e.g., trending vs. ranging, high vs. low volatility).
- **Methods**:
  1. **Probability Feeds**: Monitor shifts in Polymarket’s implied probabilities.
  2. **Volatility Regimes**: Use historical volatility (HV) to detect high/low volatility periods.
  3. **Sentiment Shifts**: Track NLP sentiment scores for narrative changes.

### Dynamic Adjustments
- **Strategy Switching**:
  - **High Volatility**: Deploy contrarian fades (The Compulsive’s edge).
  - **Low Volatility**: Deploy momentum strategies (Polymarket’s adaptability).
- **Position Sizing**: Adjust based on regime (e.g., larger positions in high-conviction regimes).

### Sentiment Analysis
- **NLP Models**: Parse news and social media for sentiment trends.
- **Volume Spikes**: Identify significant volume changes as confirmation signals.
- **Liquidity Patterns**: Monitor order book depth for manipulative behavior.

---

## 4. Execution Logic

### Dynamic Position Sizing
| Conviction Level | Position Size | Max Risk per Trade |
|------------------|---------------|---------------------|
| Low (1/5)        | 1% of capital  | 0.5%                |
| Medium (3/5)     | 3% of capital  | 1.5%                |
| High (5/5)       | 5% of capital  | 2.5%                |

- **Regime Adjustment**:
  - **High Volatility**: Reduce position size by 30%.
  - **Low Volatility**: Increase position size by 20%.

### Stop-Loss Placement
| Strategy               | Stop-Loss Method               | Trailing? |
|------------------------|---------------------------------|-----------|
| Momentum Trading       | 1.5x ATR from entry             | Yes       |
| Event-Driven           | Below event-triggered level     | No        |
| Contrarian Fade        | 2x ATR from entry               | Yes       |

### Trade Execution
- **Latency**: < 100ms from signal to execution.
- **Slippage Control**: Reject trades if estimated slippage > 0.3%.
- **Order Types**:
  - **Market Orders**: For momentum trades (speed > precision).
  - **Limit Orders**: For contrarian fades (wait for better fill).

---

## 5. Risk Management

### Drawdown Limits
| Metric               | Hard Limit | Soft Limit (Alert) |
|---------------------|------------|--------------------|
| Max Drawdown        | 15%        | 8%                 |
| Daily Loss          | 5%         | 3%                 |
| Single Trade Loss   | 2.5%       | 1%                 |

- **Actions on Breach**:
  - **Soft Limit**: Reduce position sizes by 50%.
  - **Hard Limit**: Full stop + manual review required.

### Diversification
- **Max Exposure per Market**: 15% of capital.
- **Max Exposure per Strategy**: 25% of capital.

### Tail-Risk Hedging
- **Inverse Bets**: Hedge long positions with inverse bets on negatively correlated events.
- **Volatility Hedging**: Buy OTM options on extreme outcomes.

---

## 6. Socratic Questions

### Market Narrative Challenges
| Question                                                  | Purpose                          | Example Trade                     |
|-----------------------------------------------------------|-----------------------------------|-----------------------------------|
| *"What narrative is driving this price action?"*          | Uncover hidden market intent     | Fade a breakout if sentiment is overly bullish. |
| *"Why is volume spiking here?"*                          | Detect manipulative behavior     | Short if volume spike is unconfirmed. |

### Strategy Self-Critique
| Question                                                  | Purpose                          | Action                            |
|-----------------------------------------------------------|-----------------------------------|-----------------------------------|
| *"Is this regime shift real, or just noise?"*            | Avoid false signals              | Wait for confirmation from multiple indicators. |
| *"What does the order book dynamics reveal?"*            | Validate execution quality       | Avoid illiquid markets.           |

---

## 7. Performance Targets
| Metric               | Target       | Notes                          |
|---------------------|--------------|--------------------------------|
| Annualized Return   | 60-120%      | Moderate volatility expected.  |
| Sharpe Ratio        | > 1.5        | Risk-adjusted returns.         |
| Max Drawdown        | < 15%        | Controlled risk.               |
| Win Rate            | 50-60%       | Balanced win rate and R:R.     |

---

## 8. Risks & Mitigations
| Risk                  | Impact               | Mitigation                     |
|-----------------------|----------------------|--------------------------------|
| Regime Misclassification | Poor performance   | Use multiple regime detection methods. |
| Overfitting           | Poor live performance | Backtest on out-of-sample data. |
| Latency               | Missed opportunities  | Optimize API calls.            |

---

## 9. Conclusion
The **Adaptive-Compulsive Hybrid** combines the **real-time adaptability** of the Polymarket Trading System with the **high-conviction execution** of The Compulsive. By integrating regime detection, sentiment analysis, and dynamic position sizing, this hybrid aims to **capitalize on short-term momentum** while **managing drawdowns** and **adapting to changing market conditions**.