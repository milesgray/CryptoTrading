# Automated Polymarket Trading System Specification

## 1. Strategy Overview

The Phlegmatic trading system is designed to exploit trends and diversification in the Polymarket ecosystem. The strategy focuses on:
- Trend-following: Identifying and capitalizing on long-term market trends.
- Carry trades: Exploiting price differentials between related assets.
- Portfolio diversification: Spreading risk across multiple markets and assets.

The system operates with a patient, process-oriented approach, avoiding over-trading and focusing on long-term edge.

## 2. Data Requirements

The system requires access to:
- Long-term historical data: Price, volume, and other market data for various assets.
- Macroeconomic indicators: Economic data that may influence market trends.

## 3. Execution Logic

The execution logic involves:
- Slow, deliberate position building: Gradually increasing positions as trends confirm.
- Unwinding positions: Slowly exiting positions as trends reverse or risk limits are approached.

## 4. Risk Management

Risk management includes:
- Strict drawdown limits: Predefined limits on acceptable losses.
- Diversification across markets: Spreading risk to avoid concentration in any single market.

## 5. Monitoring & Alerts

Monitoring and alerts cover:
- Long-term performance tracking: Regularly reviewing the performance of the strategy.
- Regime shift detection: Identifying changes in market conditions that may require strategy adjustments.

## 6. Socratic Questions

The following questions are used to simplify the design:
- Is this feature necessary, or does it add fragility?
- What is the simplest way to achieve this?

This document provides a comprehensive specification for the Phlegmatic trading system, ensuring a patient, process-oriented, and risk-averse approach to automated trading on Polymarket.