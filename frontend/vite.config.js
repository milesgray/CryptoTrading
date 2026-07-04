import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env variables from the current directory
  const env = loadEnv(mode, process.cwd(), '')
  
  // Resolve backend URL (inside docker network, this will be http://serve:8000)
  const backendUrl = env.VITE_BACKEND_URL || 'http://localhost:8362'
  const trainUrl = env.VITE_TRAIN_URL || 'http://localhost:8389'
  // Derive WebSocket URL from HTTP URL (e.g. ws://serve:8000 or ws://localhost:8362)
  const backendWsUrl = backendUrl.replace(/^http/, 'ws')

  console.log(`[Vite Config] Proxying /api to ${backendUrl}`);
  console.log(`[Vite Config] Proxying /api/train to ${trainUrl}`);
  console.log(`[Vite Config] Proxying /ws to ${backendWsUrl}`);

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api/train': {
          target: trainUrl,
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/api\/train/, ''),
        },
        '/api': {  // Proxy API requests to the FastAPI backend
          target: backendUrl,
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/api/, ''), // Remove /api prefix
        },
        '/ws': {  // Proxy WebSocket connections
          target: backendWsUrl,
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
  }
})