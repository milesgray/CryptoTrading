import axios from 'axios';
import { formatISO } from 'date-fns';

const api = axios.create({
  baseURL: '/api', // Use the proxy we configured in vite.config.js
});

// WebSocket service
class WebSocketService {
  constructor() {
    this.socket = null;
    this.token = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000; // Start with 1 second
    this.maxReconnectDelay = 30000; // Max 30 seconds
    this.isConnecting = false;
    this.callbacks = {
      price: [],
      orderBook: []
    };
    this.fallbackInterval = null;
  }

  getSubscriberCount() {
    return this.callbacks.price.length + this.callbacks.orderBook.length;
  }

  startFallbackPolling() {
    if (this.fallbackInterval) return;
    console.log('[WebSocket] Starting HTTP fallback polling for token:', this.token);
    
    const poll = async () => {
      if (!this.token) return;
      try {
        const priceData = await getLatestPrice(this.token);
        if (priceData) {
          console.log('[WebSocket Fallback] Polled price:', priceData);
          
          this.callbacks.price.forEach(callback => {
            try {
              callback(priceData);
            } catch (err) {
              console.error('[WebSocket Fallback] Error in price callback:', err);
            }
          });

          if (priceData.order_book && this.callbacks.orderBook.length > 0) {
            this.callbacks.orderBook.forEach(callback => {
              try {
                callback(priceData.order_book);
              } catch (err) {
                console.error('[WebSocket Fallback] Error in order book callback:', err);
              }
            });
          }
        }
      } catch (error) {
        console.error('[WebSocket Fallback] Polling failed:', error);
      }
    };

    poll();
    this.fallbackInterval = setInterval(poll, 5000);
  }

  stopFallbackPolling() {
    if (this.fallbackInterval) {
      console.log('[WebSocket] Stopping HTTP fallback polling');
      clearInterval(this.fallbackInterval);
      this.fallbackInterval = null;
    }
  }

  async connect(token) {
    if (this.token && this.token !== token) {
      console.log(`[WebSocket] Token changed from ${this.token} to ${token}, reconnecting...`);
      this.disconnect(true);
    }
    
    this.token = token;

    // Start fallback polling immediately so we have data while connecting/reconnecting
    this.startFallbackPolling();

    // If we already have a valid connection, stop fallback and return
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      console.log('[WebSocket] Already connected');
      this.stopFallbackPolling();
      return true;
    }

    // If we're already trying to connect, don't start another connection
    if (this.isConnecting) {
      console.log('[WebSocket] Connection already in progress');
      return false;
    }

    // If we have a socket in closing state, wait for it to close
    if (this.socket) {
      if (this.socket.readyState === WebSocket.CONNECTING) {
        console.log('[WebSocket] Connection in progress, waiting...');
        return new Promise((resolve) => {
          const checkConnection = () => {
            if (this.socket.readyState === WebSocket.OPEN) {
              this.stopFallbackPolling();
              resolve();
            } else if (this.socket.readyState === WebSocket.CONNECTING) {
              setTimeout(checkConnection, 100);
            } else {
              this.socket = null;
              this.connect(token).then(resolve).catch(console.error);
            }
          };
          setTimeout(checkConnection, 100);
        });
      }
    }

    this.isConnecting = true;
    
    // Clear any existing reconnection timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Create a promise that resolves when the connection is established
    return new Promise((resolve, reject) => {
      try {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/price/${encodeURIComponent(token)}`;
        
        console.log(`[WebSocket] Connecting to ${wsUrl}`);
        
        // Create a new WebSocket connection
        this.socket = new WebSocket(wsUrl);
        
        // Set a connection timeout
        const connectionTimeout = setTimeout(() => {
          if (this.socket && this.socket.readyState === WebSocket.CONNECTING) {
            console.error('[WebSocket] Connection timeout');
            this.socket.close(4000, 'Connection timeout');
            reject(new Error('Connection timeout'));
          }
        }, 10000); // 10 second timeout
        
        // Set up event handlers
        this.socket.onopen = () => {
          clearTimeout(connectionTimeout);
          console.log('[WebSocket] Connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.stopFallbackPolling(); // Successfully connected! Stop fallback HTTP polling.
          this.startKeepAlive();
          resolve();
        };
        
        // Handle connection errors
        this.socket.onerror = (error) => {
          clearTimeout(connectionTimeout);
          const errorMsg = `[WebSocket] Error: ${error.message || 'Unknown error'}`;
          console.error(errorMsg);
          this.isConnecting = false;
          this.stopKeepAlive();
          
          // Make sure fallback polling is active
          this.startFallbackPolling();
          
          // If we have a socket, close it
          if (this.socket) {
            try {
              this.socket.close(4000, 'Connection error');
            } catch (e) {
              console.error('[WebSocket] Error closing socket:', e);
            }
          }
          
          reject(new Error(errorMsg));
        };
        
        // Handle connection close
        this.socket.onclose = (event) => {
          clearTimeout(connectionTimeout);
          console.log(`[WebSocket] Disconnected: ${event.code} ${event.reason || 'No reason provided'}`);
          this.isConnecting = false;
          
          // Start fallback HTTP polling on disconnect
          this.startFallbackPolling();
          
          // Stop keep-alive ping
          this.stopKeepAlive();
          
          // Clean up the current socket
          if (this.socket) {
            try {
              this.socket.onopen = null;
              this.socket.onclose = null;
              this.socket.onerror = null;
              this.socket.onmessage = null;
              if (this.socket.readyState === WebSocket.OPEN) {
                this.socket.close();
              }
            } catch (e) {
              console.error('[WebSocket] Error cleaning up socket:', e);
            } finally {
              this.socket = null;
            }
          }
          
          // Don't attempt to reconnect if we explicitly closed the connection
          if (event.code === 1000 || event.code === 1005) {
            console.log('[WebSocket] Connection closed normally');
            return;
          }
          
          // Only attempt to reconnect if we have a valid token and we're not already reconnecting
          if (this.token && !this.reconnectTimeout) {
            console.log('[WebSocket] Will attempt to reconnect...');
            // Use a small delay before reconnecting
            this.reconnectTimeout = setTimeout(() => {
              this.reconnectTimeout = null;
              if (this.token && (!this.socket || this.socket.readyState === WebSocket.CLOSED)) {
                this.connect(this.token).catch(err => {
                  console.error('[WebSocket] Reconnection attempt failed:', err);
                  // If reconnect fails, schedule another attempt
                  this.attemptReconnect();
                });
              }
            }, 1000);
          }
        };
        
        // Set up message handler
        this.socket.onmessage = (event) => {
          try {
            if (!event.data) {
              console.warn('[WebSocket] Received empty message');
              return;
            }
            
            const data = JSON.parse(event.data);
            console.log('[WebSocket] Received message:', data);
            
            if (data.type === 'price_update' && this.callbacks.price.length > 0) {
              const priceData = data.data || data;
              console.log('[WebSocket] Processing price update:', priceData);
              
              this.callbacks.price.forEach(callback => {
                try {
                  if (priceData.price !== undefined) {
                    callback(priceData);
                  } else if (priceData.close !== undefined) {
                    callback({ price: priceData.close });
                  }
                } catch (err) {
                  console.error('[WebSocket] Error in price update callback:', err);
                }
              });
            } else if (data.type === 'order_book_update' && this.callbacks.orderBook.length > 0) {
              this.callbacks.orderBook.forEach(callback => {
                try {
                  callback(data.data || data);
                } catch (err) {
                  console.error('[WebSocket] Error in order book update callback:', err);
                }
              });
            } else if (data.type === 'pong') {
              this.lastPong = Date.now();
              return;
            }
          } catch (error) {
            console.error('[WebSocket] Error processing message:', error, event.data);
          }
        };
      } catch (error) {
        console.error('Error creating WebSocket:', error);
        this.disconnect(true);
        this.attemptReconnect();
      }
    });
  }

  disconnect(preserveCallbacks = false) {
    console.log('[WebSocket] Disconnecting...');
    this.stopKeepAlive();
    this.stopFallbackPolling();
    
    if (this.socket) {
      try {
        this.socket.close(1000, 'User disconnected');
      } catch (err) {
        console.error('[WebSocket] Error during disconnect:', err);
      } finally {
        this.socket = null;
      }
    }
    
    this.isConnecting = false;
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    if (!preserveCallbacks) {
      this.callbacks.price = [];
      this.callbacks.orderBook = [];
    }
    
    console.log('[WebSocket] Disconnected');
  }

  // Keep-alive mechanism
  startKeepAlive() {
    this.stopKeepAlive();
    this.lastPong = Date.now();
    
    this.keepAliveInterval = setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        try {
          const timeSinceLastPong = Date.now() - this.lastPong;
          if (timeSinceLastPong > 60000) {
            console.warn('[WebSocket] No pong received recently, reconnecting...');
            this.socket.close(4000, 'No pong received');
            return;
          }
          this.socket.send(JSON.stringify({ type: 'ping' }));
        } catch (err) {
          console.error('[WebSocket] Error sending ping:', err);
        }
      }
    }, 25000);
  }
  
  stopKeepAlive() {
    if (this.keepAliveInterval) {
      clearInterval(this.keepAliveInterval);
      this.keepAliveInterval = null;
    }
  }

  async attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnection attempts reached');
      setTimeout(() => {
        this.reconnectAttempts = 0;
      }, 60000);
      return;
    }
    
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts) + 
      Math.random() * 1000,
      this.maxReconnectDelay
    );
    
    console.log(`[WebSocket] Reconnecting in ${Math.round(delay/1000)} seconds...`);
    
    this.reconnectTimeout = setTimeout(() => {
      if (this.token) {
        console.log(`[WebSocket] Reconnection attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts}`);
        this.reconnectAttempts++;
        this.connect(this.token).catch(error => {
          console.error('[WebSocket] Reconnection failed:', error);
          this.attemptReconnect();
        });
      }
    }, delay);
  }

  onPriceUpdate(callback) {
    if (typeof callback !== 'function') {
      console.error('[WebSocket] onPriceUpdate requires a function as callback');
      return () => {};
    }
    
    console.log('[WebSocket] Adding price update callback');
    this.callbacks.price.push(callback);
    
    if (this.token) {
      this.connect(this.token);
    }
    
    return () => {
      this.callbacks.price = this.callbacks.price.filter(cb => cb !== callback);
      if (this.getSubscriberCount() === 0) {
        this.disconnect();
      }
    };
  }

  onOrderBookUpdate(callback) {
    if (typeof callback !== 'function') {
      console.error('[WebSocket] onOrderBookUpdate requires a function as callback');
      return () => {};
    }
    
    console.log('[WebSocket] Adding order book update callback');
    this.callbacks.orderBook.push(callback);
    
    if (this.token) {
      this.connect(this.token);
    }
    
    return () => {
      this.callbacks.orderBook = this.callbacks.orderBook.filter(cb => cb !== callback);
      if (this.getSubscriberCount() === 0) {
        this.disconnect();
      }
    };
  }
}

export const webSocketService = new WebSocketService();

export const getCandlestickData = async (token, start, end, granularity) => {
  console.log('getCandlestickData called with:', { token, start, end, granularity });
  try {
    const formattedStart = formatISO(start);  // Format dates for the API
    const formattedEnd = formatISO(end);
    console.log('Formatted dates:', { formattedStart, formattedEnd });
    
    console.log('Making API request to:', `/candlestick/${token}`, {
      params: { start: formattedStart, end: formattedEnd, granularity, include_book: true }
    });
    
    const response = await api.get(`/candlestick/${token}`, {
      params: {
        start: formattedStart,
        end: formattedEnd,
        granularity,
        include_book: true
      },
    });
    
    console.log('API response received:', {
      status: response.status,
      statusText: response.statusText,
      dataLength: response.data ? response.data.length : 0,
      dataSample: response.data ? response.data.slice(0, 2) : null
    });
    
    if (!response.data) {
      console.warn('No data in API response');
      return [];
    }
    
    return response.data;
  } catch (error) {
    console.error("Error fetching candlestick data:", error);
    if (error.response) {
      console.error('API Error Response:', {
        status: error.response.status,
        statusText: error.response.statusText,
        data: error.response.data
      });
    }
    throw error;
  }
};

export const getLatestPrice = async (token) => {
  console.log('getLatestPrice called for token:', token);
  try {
      const response = await api.get(`/latest_price/${token}`);
      console.log('Latest price response:', {
        status: response.status,
        data: response.data
      });
      return response.data;
  } catch(error) {
      console.error("Error fetching latest price:", error);
      if (error.response) {
        console.error('Latest Price API Error:', {
          status: error.response.status,
          statusText: error.response.statusText,
          data: error.response.data
        });
      }
      throw error;
  }
};

export const getServices = async () => {
  try {
    const response = await api.get('/services');
    return response.data;
  } catch (error) {
    console.error('Error fetching services:', error);
    throw error;
  }
};

export const startService = async (name) => {
  try {
    const response = await api.post(`/services/${name}/start`);
    return response.data;
  } catch (error) {
    console.error(`Error starting service ${name}:`, error);
    throw error;
  }
};

export const stopService = async (name) => {
  try {
    const response = await api.post(`/services/${name}/stop`);
    return response.data;
  } catch (error) {
    console.error(`Error stopping service ${name}:`, error);
    throw error;
  }
};

export const restartService = async (name) => {
  try {
    const response = await api.post(`/services/${name}/restart`);
    return response.data;
  } catch (error) {
    console.error(`Error restarting service ${name}:`, error);
    throw error;
  }
};

export const getServiceLogs = async (name, limit = 100) => {
  try {
    const response = await api.get(`/services/${name}/logs`, { params: { limit } });
    return response.data;
  } catch (error) {
    console.error(`Error fetching logs for service ${name}:`, error);
    throw error;
  }
};

export const updateServiceConfig = async (name, config) => {
  try {
    const response = await api.post(`/services/${name}/config`, { config });
    return response.data;
  } catch (error) {
    console.error(`Error updating config for service ${name}:`, error);
    throw error;
  }
};

export const getFeedsStatus = async (token) => {
  try {
    const response = await api.get(`/feeds/${token}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching feeds for ${token}:`, error);
    throw error;
  }
};

export const getBookPressure = async (token) => {
  try {
    const response = await api.get(`/pressure/${token}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching book pressure for ${token}:`, error);
    throw error;
  }
};

export const searchSimilarSetups = async (prices, symbol = "BTC") => {
  try {
    const response = await api.post('/retrieval/search', { prices, symbol });
    return response.data;
  } catch (error) {
    console.error(`Error searching setups:`, error);
    throw error;
  }
};

export const getSentimentData = async (token) => {
  try {
    const response = await api.get(`/retrieval/sentiment/${token}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching sentiment for ${token}:`, error);
    throw error;
  }
};

export const getJepaRegime = async (token) => {
  try {
    const response = await api.get(`/retrieval/jepa/regime/${token}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching JEPA regime for ${token}:`, error);
    throw error;
  }
};

export const getTradeLedger = async () => {
  try {
    const response = await api.get('/trade/ledger');
    return response.data;
  } catch (error) {
    console.error(`Error fetching trade ledger:`, error);
    throw error;
  }
};

export const executeTrade = async (order) => {
  try {
    const response = await api.post('/trade/order', order);
    return response.data;
  } catch (error) {
    console.error(`Error executing trade:`, error);
    throw error;
  }
};

export const startTrainingTask = async (config) => {
  try {
    const response = await api.post('/train/train', config);
    return response.data;
  } catch (error) {
    console.error('Error starting training task:', error);
    throw error;
  }
};

export const getTrainingTasks = async () => {
  try {
    const response = await api.get('/train/tasks');
    return response.data;
  } catch (error) {
    console.error('Error fetching training tasks:', error);
    throw error;
  }
};

export const getTrainingTaskStatus = async (taskId) => {
  try {
    const response = await api.get(`/train/tasks/${taskId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching status for task ${taskId}:`, error);
    throw error;
  }
};

export const getTrainedModels = async () => {
  try {
    const response = await api.get('/train/models');
    return response.data;
  } catch (error) {
    console.error('Error fetching trained models:', error);
    throw error;
  }
};

export const runModelInference = async (modelId, inputData) => {
  try {
    const response = await api.post(`/train/models/${modelId}/predict`, inputData);
    return response.data;
  } catch (error) {
    console.error(`Error running inference on model ${modelId}:`, error);
    throw error;
  }
};