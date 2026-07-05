# Plan: Rework Order Book Snapshot Panel into a Depth Visualization

We will redesign the [OrderBookPanel.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/OrderBookPanel.jsx) into a real-time depth visualization to help users make rapid decisions on price direction.

## Proposed Changes

### [MODIFY] [OrderBookPanel.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/OrderBookPanel.jsx)

- **Import ECharts**: Use `echarts` to render a custom real-time **Order Book Depth Chart** (cumulative volume walls).
- **Imbalance / Direction Indicator**: Build a premium horizontal "Pressure Meter" showing Bid vs. Ask volume concentration.
- **Midpoint and Spread Display**: Clearly overlay the midpoint price and spread in the center of the visualization.
- **Whale Wall Callouts**: Highlight the most significant bid/ask outlier order block levels (e.g. support and resistance walls) with visual badges to identify where whales are positioning their limit orders.
- **Premium Styling**: Set up a sleek dark theme container matching the platform's overall style.

---

## Verification Plan

### Automated Verification
- Run the Vite production build to verify no compilation errors.
  `npm run build` inside `frontend/`

### Manual Verification
- Verify the depth chart updates dynamically as WebSocket price/order-book streams are received.
- Check the color styling (emerald for bids, rose for asks) and ensure there is no white-on-white text.
