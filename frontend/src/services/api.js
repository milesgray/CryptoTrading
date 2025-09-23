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
  }

  async connect(token) {
    // If we already have a valid connection, don't reconnect
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      console.log('[WebSocket] Already connected');
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
        // If connecting, wait a bit and try again
        if (this.socket && this.socket.readyState === WebSocket.CONNECTING) {
          console.log('[WebSocket] Connection in progress, waiting...');
          return new Promise((resolve) => {
            const checkConnection = () => {
              if (this.socket.readyState === WebSocket.OPEN) {
                resolve();
              } else if (this.socket.readyState === WebSocket.CONNECTING) {
                setTimeout(checkConnection, 100);
              } else {
                // Connection failed or closed, try again
                this.socket = null;
                this.connect(token).then(resolve).catch(console.error);
              }
            };
            setTimeout(checkConnection, 100);
          });
        }
      }
    }

    this.token = token;
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
          
          // If we have a socket, close it
          if (this.socket) {
            try {
              this.socket.close(1006, 'Connection error');
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
              // The backend sends price data directly in the message
              const priceData = data.data || data;
              console.log('[WebSocket] Processing price update:', priceData);
              
              this.callbacks.price.forEach(callback => {
                try {
                  // The callback expects a price value directly
                  if (priceData.price !== undefined) {
                    callback(priceData);
                  } else if (priceData.close !== undefined) {
                    // If we get a candlestick, use the close price
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
              // Handle pong message (keep-alive response)
              this.lastPong = Date.now();
              return;
            }
          } catch (error) {
            console.error('[WebSocket] Error processing message:', error, event.data);
          }
        };
      } catch (error) {
        console.error('Error creating WebSocket:', error);
        this.disconnect();
        this.attemptReconnect();
      }
    });
  }

  disconnect() {
    console.log('[WebSocket] Disconnecting...');
    this.stopKeepAlive();
    
    if (this.socket) {
      try {
        // Close with normal status code
        this.socket.close(1000, 'User disconnected');
      } catch (err) {
        console.error('[WebSocket] Error during disconnect:', err);
      } finally {
        this.socket = null;
      }
    }
    
    this.isConnecting = false;
    
    // Clear any pending reconnection attempts
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    // Clear callbacks to prevent memory leaks
    this.callbacks.price = [];
    this.callbacks.orderBook = [];
    
    console.log('[WebSocket] Disconnected');
  }

  // Keep-alive mechanism
  startKeepAlive() {
    this.stopKeepAlive();
    this.lastPong = Date.now();
    
    // Send ping every 25 seconds (server should close connection after 30s of inactivity)
    this.keepAliveInterval = setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        try {
          // Check if we've received a pong recently
          const timeSinceLastPong = Date.now() - this.lastPong;
          if (timeSinceLastPong > 60000) { // 60 seconds without a pong
            console.warn('[WebSocket] No pong received recently, reconnecting...');
            this.socket.close(4000, 'No pong received');
            return;
          }
          
          // Send ping
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
      // Reset attempts after a while to allow recovery
      setTimeout(() => {
        this.reconnectAttempts = 0;
      }, 60000); // Reset after 1 minute
      return;
    }
    
    // Exponential backoff with jitter
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts) + 
      Math.random() * 1000, // Add jitter
      this.maxReconnectDelay
    );
    
    console.log(`[WebSocket] Reconnecting in ${Math.round(delay/1000)} seconds...`);
    
    this.reconnectTimeout = setTimeout(() => {
      if (this.token) {
        console.log(`[WebSocket] Reconnection attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts}`);
        this.reconnectAttempts++;
        this.connect(this.token).catch(error => {
          console.error('[WebSocket] Reconnection failed:', error);
          // Schedule next reconnection attempt
          this.attemptReconnect();
        });
      }
    }, delay);
  }

  onPriceUpdate(callback) {
    if (typeof callback !== 'function') {
      console.error('[WebSocket] onPriceUpdate requires a function as callback');
      return () => {}; // Return empty cleanup function
    }
    
    console.log('[WebSocket] Adding price update callback');
    this.callbacks.price.push(callback);
    
    // Return cleanup function
    return () => {
      this.callbacks.price = this.callbacks.price.filter(cb => cb !== callback);
    };
  }

  onOrderBookUpdate(callback) {
    if (typeof callback !== 'function') {
      console.error('[WebSocket] onOrderBookUpdate requires a function as callback');
      return () => {}; // Return empty cleanup function
    }
    
    console.log('[WebSocket] Adding order book update callback');
    this.callbacks.orderBook.push(callback);
    
    // Return cleanup function
    return () => {
      this.callbacks.orderBook = this.callbacks.orderBook.filter(cb => cb !== callback);
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
}