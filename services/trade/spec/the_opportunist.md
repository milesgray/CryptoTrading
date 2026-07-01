# The Opportunist

## 1. Strategy Overview

### Personality & Risk Profile
- **Hybrid of**: Rational-Analytical + Compulsive.
- **Motto**: *"Strike fast, strike smart."*
- **Risk Profile**: High risk, high reward, with **data-backed aggression**.
- **Trading Style**: High-frequency, opportunistic trading with **data validation** and **low-latency execution**.

### Core Principles
1. **Data-Backed Aggression**: The Rational-Analytical subagent validates high-conviction opportunities before execution.
2. **High-Frequency Execution**: The Compulsive subagent ensures **low-latency trade execution** (e.g., < 100ms).
3. **Dynamic Position Sizing**: Adjusts position sizes based on **data confidence** and **market volatility**.
4. **Opportunistic Fades**: Fades extreme moves (e.g., 90%+ probability events) only if validated by data.

---

## 2. Data Requirements

### Real-Time Feeds
| Data Type               | Source               | Frequency      | Use Case                          |
|------------------------|----------------------|----------------|-----------------------------------|
| Order Book Dynamics    | Polymarket API       | Real-time      | Liquidity analysis                |
| News Feeds             | RSS, NewsAPI         | Real-time      | Event-driven trading              |
| Probability Feeds      | Polymarket API       | Real-time      | Regime detection                  |
| Trade History          | Polymarket API       | Real-time      | Volume analysis                   |

### Behavioral Data
| Data Type               | Source               | Use Case                          |
|------------------------|----------------------|-----------------------------------|
| Volume Spikes          | Polymarket API       | Momentum detection                |
| NLP Sentiment Scores   | Custom NLP Models    | Narrative-driven validation       |

---

## 3. Adaptability Framework

### Data Validation
- **Definition**: Validate high-conviction opportunities with data before execution.
- **Methods**:
  1. **Backtesting**: Ensure contrarian signals have statistical edge.
  2. **Order Book Analysis**: Confirm liquidity and slippage estimates.
  3. **Sentiment Confirmation**: Use NLP models to validate market intent.

### Dynamic Position Sizing
- **Definition**: Adjust position sizes based on data confidence and volatility.
- **Methods**:
  1. **Confidence Scores**: Higher confidence = larger positions.
  2. **Volatility Adjustments**: Reduce size in high-volatility regimes.

### Opportunistic Fades
- **Definition**: Fade extreme moves (e.g., 90%+ probability events) only if validated by data.
- **Methods**:
  1. **Statistical Edge**: Require backtesting for contrarian signals.
  2. **Sentiment Filter**: Only fade if sentiment aligns.

---

## 4. Execution Logic

### Dynamic Position Sizing
| Confidence Level | Position Size | Max Risk per Trade |
|------------------|---------------|---------------------|
| Low (1/5)        | 1% of capital  | 0.5%                |
| Medium (3/5)     | 3% of capital  | 1.5%                |
| High (5/5)       | 5% of capital  | 2.5%                |

- **Volatility Adjustment**:
  - **High Volatility**: Reduce position size by 30%.
  - **Low Volatility**: Increase position size by 20%.

### Stop-Loss Placement
| Strategy               | Stop-Loss Method               | Trailing? |
|------------------------|---------------------------------|-----------|
| Momentum Trading       | 1.5x ATR from entry             | Yes       |
| Opportunistic Fade     | 2x ATR from entry               | Yes       |

### Trade Execution
- **Latency**: < 100ms from signal to execution.
- **Slippage Control**: Reject trades if estimated slippage > 0.3%.
- **Order Types**:
  - **Market Orders**: For momentum trades (speed > precision).
  - **Limit Orders**: For opportunistic fades (wait for better fill).

---

## 5. Risk Management

### Drawdown Limits
| Metric               | Hard Limit | Soft Limit (Alert) |
|---------------------|------------|--------------------|
| Max Drawdown        | 25%        | 15%                |
| Daily Loss          | 5%         | 3%                 |
| Single Trade Loss   | 3%         | 1.5%               |

- **Actions on Breach**:
  - **Soft Limit**: Reduce position sizes by 50%.
  - **Hard Limit**: Full stop + manual review required.

### Diversification
- **Max Exposure per Market**: 20% of capital.
- **Max Exposure per Strategy**: 30% of capital.

### Tail-Risk Hedging
- **Inverse Bets**: Hedge long positions with inverse bets on negatively correlated events.
- **Volatility Hedging**: Buy out-of-the-money (OTM) options on extreme outcomes.

---

## 6. Performance Targets
| Metric               | Target       | Notes                          |
|---------------------|--------------|--------------------------------|
| Annualized Return   | 100-200%     | High volatility expected.      |
| Sharpe Ratio        | > 1.0        | Risk-adjusted returns.         |
| Max Drawdown        | < 25%        | Aggressive risk.               |
| Win Rate            | 40-50%       | Low win rate, high R:R.        |

---

## 7. Risks & Mitigations
| Risk                  | Impact               | Mitigation                     |
|-----------------------|----------------------|--------------------------------|
| Data Validation Failure | Poor performance   | Backtest rigorously            |
| Latency               | Missed opportunities  | Optimize API calls             |
| Overfitting           | Poor live performance | Use out-of-sample data         |

---

## 8. Conclusion
**The Opportunist** combines the **data-driven rigor** of the Rational-Analytical subagent with the **high-frequency execution** of The Compulsive. This hybrid is designed to **capitalize on fleeting opportunities** while minimizing false signals through data validation. By integrating dynamic position sizing and low-latency execution, The Opportunist aims for **high-risk, high-reward performance** in dynamic markets.