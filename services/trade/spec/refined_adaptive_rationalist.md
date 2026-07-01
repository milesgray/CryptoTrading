# Refined Adaptive Rationalist

## 1. Strategy Overview

### Personality & Risk Profile
- **Hybrid of**: Rational-Analytical + Intuitive-Emotional.
- **Motto**: *"Data-driven, but never blind."*
- **Risk Profile**: Moderate risk, adaptive, **sentiment-aware**.
- **Trading Style**: Regime-adaptive trading with **real-time sentiment integration**.

### Core Principles
1. **Adaptability**: Adjust strategies based on **regime shifts** (trending vs. mean-reverting) and **sentiment analysis**.
2. **Sentiment as a Primary Signal**: Use **real-time sentiment spikes** to trigger trades in chaotic regimes.
3. **Dynamic Risk Management**: Adjust position sizes and stop-losses based on **confidence scores** derived from both data and sentiment.
4. **Simplified Regime Detection**: Reduce regime detection to **binary states** (trending vs. mean-reverting) to minimize latency.
5. **Static Risk Limits**: Implement **static risk limits** as a fallback for low-confidence trades.

---

## 2. Data Requirements

### Real-Time Feeds
| Data Type               | Source               | Frequency      | Use Case                          |
|------------------------|----------------------|----------------|-----------------------------------|
| Order Book Dynamics    | Polymarket API       | Real-time      | Liquidity analysis                |
| News Feeds             | RSS, NewsAPI         | Real-time      | Event-driven trading              |
| Social Media Sentiment | Twitter, Reddit      | Real-time      | Sentiment analysis                |
| Probability Feeds      | Polymarket API       | Real-time      | Regime detection                  |

### Behavioral Data
| Data Type               | Source               | Use Case                          |
|------------------------|----------------------|-----------------------------------|
| Volume Spikes          | Polymarket API       | Momentum detection                |
| NLP Sentiment Scores   | Custom NLP Models    | Narrative-driven trading          |

---

## 3. Adaptability Framework

### Regime Detection
- **Definition**: Identify changes in market behavior (e.g., trending vs. mean-reverting).
- **Methods**:
  1. **Probability Feeds**: Monitor shifts in Polymarket’s implied probabilities.
  2. **Volatility Regimes**: Use historical volatility (HV) to detect high/low volatility periods.
  3. **Sentiment Shifts**: Track NLP sentiment scores for narrative changes.

### Dynamic Adjustments
- **Strategy Switching**:
  - **High Volatility**: Deploy sentiment-driven contrarian fades.
  - **Low Volatility**: Deploy momentum strategies.
- **Position Sizing**: Adjust based on regime and sentiment confidence.

### Sentiment Analysis
- **NLP Models**: Parse news and social media for sentiment trends.
- **Sentiment Spikes**: Treat extreme sentiment (e.g., fear/greed) as a **primary signal** in chaotic regimes.

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
| Sentiment-Driven       | 2x ATR from entry               | Yes       |

### Trade Execution
- **Latency**: < 100ms from signal to execution.
- **Slippage Control**: Reject trades if estimated slippage > 0.3%.
- **Order Types**:
  - **Market Orders**: For momentum trades (speed > precision).
  - **Limit Orders**: For sentiment-driven fades (wait for better fill).

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

### Static Risk Limits
- **Fallback**: If confidence scores are low, enforce **static risk limits** (e.g., max 1% risk per trade).

### Diversification
- **Max Exposure per Market**: 15% of capital.
- **Max Exposure per Strategy**: 25% of capital.

---

## 6. Performance Targets
| Metric               | Target       | Notes                          |
|---------------------|--------------|--------------------------------|
| Annualized Return   | 50-100%      | Moderate volatility expected.  |
| Sharpe Ratio        | > 1.5        | Risk-adjusted returns.         |
| Max Drawdown        | < 15%        | Controlled risk.               |
| Win Rate            | 50-60%       | Balanced win rate and R:R.     |

---

## 7. Risks & Mitigations
| Risk                  | Impact               | Mitigation                     |
|-----------------------|----------------------|--------------------------------|
| Sentiment Noise       | False signals        | Use multiple sentiment sources. |
| Latency               | Missed opportunities  | Optimize API calls.            |
| Overfitting           | Poor live performance | Backtest on out-of-sample data. |

---

## 8. Conclusion
The **Refined Adaptive Rationalist** combines **data-driven rigor** with **sentiment-aware intuition** to create a trading system that adapts to market regimes while leveraging real-time sentiment as a primary signal. By simplifying regime detection and implementing static risk limits as a fallback, this design addresses its original weaknesses while preserving its core strengths.