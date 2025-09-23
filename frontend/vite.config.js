import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {  // Proxy API requests to the FastAPI backend
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ''), // Remove /api prefix
      },
      '/ws': {  // Proxy WebSocket connections
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
        secure: false,
        xfwd: true,
        logLevel: 'debug',
        // Configure WebSocket upgrade
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('[Vite Proxy Error]', err);
          });
          proxy.on('proxyReqWs', (proxyReq, req, socket) => {
            console.log('[Vite Proxy] WebSocket connection established:', req.url);
            socket.on('error', (err) => {
              console.error('[Vite Proxy] WebSocket error:', err);
            });
          });
          proxy.on('proxyRes', (proxyRes, req, _res) => {
            console.log('[Vite Proxy] Received response:', {
              statusCode: proxyRes.statusCode,
              url: req.url,
              headers: proxyRes.headers
            });
          });
        },
        // Additional WebSocket options
        ws: true,
        // Handle connection close
        on: {
          close: (err) => {
            console.log('[Vite Proxy] WebSocket connection closed', err || '');
          },
          error: (err) => {
            console.error('[Vite Proxy] WebSocket error:', err);
          },
          end: () => {
            console.log('[Vite Proxy] WebSocket connection ended');
          }
        }
      },
    },
  },
})