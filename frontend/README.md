# Frontend

This folder contains the web interface for visualizing data collected and processed by the `src/cryptotrading` library and the `services` backend.

## Overview
- A **Vite + Tailwind CSS** frontend that provides real-time and historical visualizations of cryptocurrency trading data, predictions, and market sentiment.
- Connects to the `services/serve` API to fetch aggregated data from all backend services.

## Structure
```
frontend/
├── dist/           # Production build output
├── src/
│   ├── assets/      # Static assets (images, fonts, etc.)
│   ├── components/  # Reusable UI components
│   ├── pages/       # Page-level components
│   ├── App.vue      # Main application entry
│   └── main.js      # Application bootstrap
├── index.html      # Root HTML file
├── package.json    # Project dependencies and scripts
├── vite.config.js  # Vite configuration
└── tailwind.config.js # Tailwind CSS configuration
```

## Setup
1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

## Configuration
- API endpoints are configured in `src/config.js` (or equivalent). Ensure the `services/serve` API is running and accessible.

## Deployment
- The `dist/` folder is served by the `services/serve` backend in production. For standalone deployment, configure your web server to serve the `dist/` folder.