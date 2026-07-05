# Plan: Remove Mock Results & Fix Input Styling Issues

This plan aims to transition the frontend components away from hardcoded mock/placeholder initial states and styling mismatches, ensuring a premium, consistent dark-mode aesthetic and fully functional data fetching.

## Proposed Changes

### 1. Global Styling & Input Fixes

#### [MODIFY] [index.css](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/index.css)
- Reset dropdown options to render with dark backgrounds (`#0f172a`) and light text (`#f1f5f9`) globally.
- Style general browser autofill, inputs, selects, and textareas to inherit appropriate colors and avoid system overrides that cause white-on-white text.

### 2. Component Design & Theme Alignment

#### [MODIFY] [CandlestickChart.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/CandlestickChart.jsx)
- Transition from light-theme card (`bg-white`, `text-gray-900`) to premium dark-theme container styling (`bg-slate-900/30`, `border-slate-800`, `text-slate-100`).
- Update label colors from `text-gray-700` to `text-slate-400`.
- Style the start/end date inputs and granularity selects to match the dark theme (`bg-slate-950 border-slate-800 text-slate-300`).

#### [MODIFY] [OrderBookPanel.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/OrderBookPanel.jsx)
- Transition container style to match other panels (`bg-slate-900/30`, `border-slate-800`, `text-slate-100`).
- Update headers and sub-headers to use readable text colors (`text-slate-300`, `text-slate-400`).
- Replace light-mode price bucket colors (`bg-green-50 text-green-800` / `bg-red-50 text-red-800`) with semantically clear dark-mode variations (e.g. `bg-emerald-950/20 text-emerald-400` / `bg-rose-950/20 text-rose-400`).

#### [MODIFY] [RetrievalVisualizer.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/RetrievalVisualizer.jsx)
- Update selects and inputs to explicitly define background and border styling (`bg-slate-950 border-slate-800 text-slate-300`).
- Adjust label colors to match the dark-theme design.

### 3. Remove Hardcoded Mock Data States

#### [MODIFY] [SpecializedServicePanels.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/SpecializedServicePanels.jsx)
- **PriceIngestionPanel**: Initialize `feeds` as an empty array (`[]`) and `indexPrice` as `0` or `null`. Render a skeleton or loading state if data is loading.
- **SentimentStreamPanel**: Initialize `tweets` as `[]`, and sentiment indices as `0` or `null`. Display a clean "No sentiment data available" or loading state when no data exists, rather than simulated placeholder tweets.
- **JepaRegimePanel**: Initialize `activeRegime` as `null` or empty string and `leverageMultiplier` as `0`. Render clean loading or empty states.

---

## Verification Plan

### Automated Verification
- Run the frontend build locally to ensure no syntax errors or TypeScript compilation failures.
  `npm run build` inside `frontend/`

### Manual Verification
- Launch the development server and verify the layout visually in the browser.
- Verify that inputs and select boxes do not suffer from white-on-white text issues under any native dropdown render.
- Confirm all components successfully fetch and display real data on startup instead of flashing mock data.
