# Feature Plan: Retrieval Chart Live Tracking

## Objective
Hook the "Pattern Matching & Retrieval Forecast" chart into the live WebSocket price stream to visualize how actual price development compares to the retrieved historical matches and consensus projection in real-time.

## User Needs & Business Logic
Traders using the dashboard want to know in real-time if the asset's actual price path is confirming or diverging from the retrieved historical cycles and the consensus projection. 
By displaying the actual incoming price path directly on top of the forecasted paths, traders can:
1. Make immediate quantitative judgments about the predictive accuracy of the matches.
2. Monitor trend alignment or divergence.
3. Adjust active position exposure or leverage accordingly.

## Proposed Changes

### Frontend Component: `RetrievalVisualizer.jsx`

#### 1. State Management
Add state variables to track the timing and values of actual price data arriving after the query is run:
- `queryEndTime` (Number/timestamp): Holds the timestamp of the last candlestick in the query window (representing the $T_{0}$ baseline).
- `liveActualPrices` (Array of Numbers/null): An array of size `forecastLength` tracking the actual closing prices for each elapsed forecast step.

#### 2. Reset Handling
Ensure that when a new query is run (manual query trigger or token/granularity changes), the live tracking state is reset:
- Clear `liveActualPrices` (reset to an array of size `forecastLength` filled with `null`).
- Update `queryEndTime` to the timestamp of the last query candle.

#### 3. WebSocket Integration
Introduce a `useEffect` hook to subscribe to `webSocketService.onPriceUpdate` when the query is loaded:
- Listen to real-time ticks.
- Extract the tick's price and timestamp.
- Calculate the target step index in the forecast window:
  `stepIndex = Math.floor((tickTime - queryEndTime) / (granularitySeconds * 1000))`
- If `stepIndex >= 0` and `stepIndex < forecastLength`:
  - Update `liveActualPrices[stepIndex]` with the latest tick price.
  - Re-render the chart dynamically.

#### 4. ECharts Data Mapping
In the chart configuration, construct the series data for "Actual Price (Live)":
- The data array should match the length of the x-axis (`segmentLength` historical steps + `forecastLength` forecast steps).
- Fill the historical portion with `'-'` (ECharts null indicator) up to index `segmentLength - 2`.
- At index `segmentLength - 1` (the last historical candle, $T_{0}$), insert the `lastQueryPrice` to act as the pivot point.
- From index `segmentLength` to the end:
  - If a step index is $\le$ the latest recorded step index, insert `liveActualPrices[stepIndex]`. If the price is null (due to missed updates), backfill with the previous valid price.
  - If the step index has not occurred yet, insert `'-'`.
- This creates a line that grows from left to right in real-time.

#### 5. Premium Visual Styling
Style the actual price series to make it stand out:
- Name: `"Actual Price (Live)"`
- Type: `'line'`
- Color: Vibrant rose pink (`#f43f5e`) or neon blue (`#00f2fe`) to ensure immediate contrast against the blue/purple pattern lines.
- Style: Solid, thickness `3.5`, with a glowing shadow:
  - `shadowColor: 'rgba(244, 63, 94, 0.4)'`
  - `shadowBlur: 10`
- Markers: Show a small circle symbol at the latest actual data point.

#### 6. Live Tracking Summary Panel
Add a beautiful UI card to show real-time stats when tracking is active:
- **Elapsed Forecast steps**: "X of Y steps completed"
- **Actual Price vs. Predicted**: Show the current actual price and the predicted consensus price.
- **Direction Consensus**: Show if actual direction matches predictions ("CONFIRMING ✅" in emerald green vs. "DIVERGING ❌" in rose red).
- **Cumulative Returns**: Compare actual returns since $T_{0}$ vs. consensus predicted returns.

## Verification Plan

### Manual Verification
1. Open the Live Trade Room.
2. Select BTC/USDT.
3. Enable live updates on the main chart (which connects the WebSocket).
4. In the Retrieval Forecast panel, click "Trigger Query".
5. Verify the "Actual Price (Live)" line appears and grows to the right as each minute passes.
6. Verify the "Live Performance Tracking" card displays accurate metrics.
7. Switch tokens/granularities and verify all tracking states are cleared and reset properly.
