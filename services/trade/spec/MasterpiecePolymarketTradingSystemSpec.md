# Masterpiece Polymarket Trading System Specification

## 1. Strategy Overview

### Personality & Risk Profile
- **Hybrid of**: Adaptive Rationalist, Balanced Contrarian, Compulsive-Phlegmatic Hybrid, and Adaptive-Compulsive Hybrid.
- **Motto**: *"Adapt, challenge, and execute with discipline."*
- **Risk Profile**: Moderate to high risk, opportunistic, narrative-driven, and adaptive.
- **Trading Style**: Momentum, event-driven, contrarian, and trend-following with real-time adaptability and strict risk management.

### Core Principles
1. **Adaptability**: Adjust strategies based on regime shifts and sentiment analysis.
2. **Anti-Consensus Betting**: Fade extreme moves (e.g., 90%+ probability events) with high-conviction contrarian bets.
3. **Edge Detection**: Exploit liquidity traps, false breakouts, and reversals using real-time order book analysis.
4. **Patient Execution**: Build positions gradually to avoid slippage and reduce market impact.
5. **Strict Risk Management**: Enforce drawdown limits and diversification to survive black swan events.
6. **Behavioral Signals**: Use NLP models and volume spikes to detect market intent.
7. **High-Conviction Execution**: Leverage dynamic position sizing and stop-loss logic for efficient trade execution.

---

## 2. Data Requirements

### High-Frequency Market Data
| Data Type               | Source          | Frequency      | Use Case                          |
|-------------------------|-----------------|----------------|-----------------------------------|
| Order Book Snapshots    | Polymarket API  | 100ms - 500ms  | Liquidity analysis, stop-loss detection |
| Trade History           | Polymarket API  | Real-time      | Volume analysis, momentum signals |
| Price Feeds             | Polymarket API  | Real-time      | Breakout/reversal detection       |
| Probability Feeds       | Polymarket API  | Real-time      | Contrarian signal generation      |

### Volatility Metrics
| Metric                     | Calculation                          | Purpose                          |
|---------------------------|--------------------------------------|-----------------------------------|
| Implied Volatility (IV)    | Derived from option-like payoffs     | Identify mispriced events        |
| Historical Volatility (HV) | 20-period rolling std. deviation     | Normalize position sizing        |
| Volume-Weighted Volatility| HV adjusted for trading volume       | Filter low-conviction moves      |

### Alternative Data
| Data Type       | Source               | Use Case                          |
|-----------------|----------------------|-----------------------------------|
| Social Sentiment| Twitter, Discord     | Contrarian sentiment signals      |
| News Feeds      | RSS, NewsAPI         | Event-driven volatility spikes    |
| On-Chain Activity| Blockchain explorers | Detect insider accumulation      |

### Behavioral Data
| Data Type               | Source               | Use Case                          |
|-------------------------|----------------------|-----------------------------------|
| Volume Spikes          | Polymarket API       | Momentum detection                |
| Liquidity Patterns     | Polymarket API       | False breakout detection          |
| NLP Sentiment Scores   | Custom NLP Models    | Narrative-driven trading          |

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

## 4. Adaptability Framework

### Regime Detection
- **Definition**: Identify changes in market behavior (e.g., trending vs. ranging, high vs. low volatility).
- **Methods**:
  1. **Probability Feeds**: Monitor shifts in Polymarket’s implied probabilities.
  2. **Volatility Regimes**: Use historical volatility (HV) to detect high/low volatility periods.
  3. **Sentiment Shifts**: Track NLP sentiment scores for narrative changes.

### Dynamic Adjustments
- **Strategy Switching**:
  - **High Volatility**: Deploy contrarian fades.
  - **Low Volatility**: Deploy momentum strategies.
- **Position Sizing**: Adjust based on regime (e.g., larger positions in high-conviction regimes).

### Sentiment Analysis
- **NLP Models**: Parse news and social media for sentiment trends.
- **Volume Spikes**: Identify significant volume changes as confirmation signals.
- **Liquidity Patterns**: Monitor order book depth for manipulative behavior.

---

## 5. Execution Logic

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
| Momentum Trading       | 1.5x ATR from entry             | Yes       |
| Event-Driven           | Below event-triggered level     | No        |

### Trade Execution
- **Latency**: < 100ms from signal to execution.
- **Slippage Control**: Reject trades if estimated slippage > 0.5%.
- **Order Types**:
  - **Market Orders**: For scalping (speed > precision).
  - **Limit Orders**: For contrarian fades (wait for better fill).

---

## 6. Risk Management

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

## 7. Socratic Questions

### Market Assumption Challenges
| Question                                                  | Purpose                          | Example Trade                     |
|-----------------------------------------------------------|-----------------------------------|-----------------------------------|
| *"What if this ‘support’ is a liquidity trap?"*          | Challenge technical levels       | Short at "support" if stops are clustered below. |
| *"What if the crowd is wrong about this 90% probability?"* | Contrarian fade               | Bet on the 10% underdog.          |
| *"What narrative is driving this price action?"*          | Uncover hidden market intent     | Fade a breakout if sentiment is overly bullish. |
| *"Why is volume spiking here?"*                          | Detect manipulative behavior     | Short if volume spike is unconfirmed. |

### Strategy Self-Critique
| Question                                                  | Purpose                          | Action                            |
|-----------------------------------------------------------|-----------------------------------|-----------------------------------|
| *"Is this feature necessary, or does it add fragility?"* | Simplify design                 | Remove redundant indicators.      |
| *"What is the simplest way to achieve this?"*             | Avoid over-engineering          | Use basic volume filters.         |
| *"Is this regime shift real, or just noise?"*            | Avoid false signals              | Wait for confirmation from multiple indicators. |
| *"What does the order book dynamics reveal?"*            | Validate execution quality       | Avoid illiquid markets.           |

---

## 8. Performance Targets

| Metric               | Target       | Notes                          |
|---------------------|--------------|--------------------------------|
| Annualized Return   | 80-150%      | High volatility expected.      |
| Sharpe Ratio        | > 1.5        | Risk-adjusted returns.         |
| Max Drawdown        | < 20%        | Aggressive but controlled.     |
| Win Rate            | 40-50%       | Low win rate, high R:R.        |

---

## 9. Risks & Mitigations

| Risk                  | Impact               | Mitigation                     |
|-----------------------|----------------------|--------------------------------|
| API Downtime         | Missed trades        | Fallback to cached data.       |
| Slippage             | Reduced profits      | Limit order size.              |
| Black Swan Event     | Large drawdown       | Tail-risk hedging.             |
| Regime Misclassification | Poor performance   | Use multiple regime detection methods. |
| Overfitting           | Poor live performance | Backtest on out-of-sample data. |
| Latency               | Missed opportunities  | Optimize API calls.            |

---

## 10. Conclusion

The **Masterpiece Polymarket Trading System** combines the **real-time adaptability** of the Adaptive Rationalist, the **high-conviction execution** of the Balanced Contrarian, the **patient discipline** of the Compulsive-Phlegmatic Hybrid, and the **narrative-driven insights** of the Adaptive-Compulsive Hybrid. By integrating regime detection, sentiment analysis, dynamic position sizing, and strict risk management, this system aims to **capitalize on short-term momentum**, **exploit contrarian opportunities**, and **manage drawdowns** effectively in the volatile Polymarket ecosystem.

**Key Takeaways**:
- **Adapt or fade**: Real-time adaptability is critical for navigating regime shifts.
- **Challenge everything**: The market is often wrong at extremes.
- **Speed matters**: Low-latency execution is essential for scalping and momentum trading.
- **Risk is managed, not avoided**: High risk, but with strict drawdown limits.
- **Hedge the tail**: Protect against the unknown with inverse bets and volatility hedging.
