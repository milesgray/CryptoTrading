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

  connect(token) {
    // If we already have a valid connection, don't reconnect
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    // If we're already trying to connect, don't start another connection
    if (this.isConnecting) {
      console.log('WebSocket connection already in progress');
      return;
    }

    // If we have a socket in closing state, wait for it to close
    if (this.socket && (this.socket.readyState === WebSocket.CONNECTING || this.socket.readyState === WebSocket.CLOSING)) {
      console.log('WebSocket is in connecting/closing state, waiting...');
      return;
    }

    this.token = token;
    this.isConnecting = true;
    
    // Clear any existing reconnection timeout to prevent multiple reconnection attempts
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Determine WebSocket protocol and host
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const backendHost = process.env.NODE_ENV === 'development' 
      ? 'localhost:8000' 
      : window.location.host;
    
    const wsUrl = `${wsProtocol}//${backendHost}/ws/price/${token}`;
    console.log(`[WebSocket] Connecting to ${wsUrl}`);
    
    try {
      this.socket = new WebSocket(wsUrl);
      
      // Connection opened
      this.socket.onopen = () => {
        console.log('[WebSocket] Connected successfully');
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this.isConnecting = false;
        
        // Start keep-alive ping
        this.startKeepAlive();
      };

      // Listen for messages
      this.socket.onmessage = (event) => {
        try {
          if (!event.data) {
            console.warn('[WebSocket] Received empty message');
            return;
          }
          
          const data = JSON.parse(event.data);
          //console.debug('[WebSocket] Message received:', data.type || 'unknown');
          
          if (data.type === 'price_update' && this.callbacks.price.length > 0) {
            this.callbacks.price.forEach(callback => {
              try {
                callback(data.data);
              } catch (err) {
                console.error('[WebSocket] Error in price update callback:', err);
              }
            });
          } else if (data.type === 'order_book_update' && this.callbacks.orderBook.length > 0) {
            this.callbacks.orderBook.forEach(callback => {
              try {
                callback(data.data);
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

      // Connection closed
      this.socket.onclose = (event) => {
        console.log(`[WebSocket] Disconnected: ${event.code} ${event.reason || 'No reason provided'}`);
        this.isConnecting = false;
        this.socket = null;
        
        // Stop keep-alive ping
        this.stopKeepAlive();
        
        // Don't attempt to reconnect if we explicitly closed the connection
        if (event.code === 1000) {
          console.log('[WebSocket] Connection closed normally');
          return;
        }
        
        // Attempt to reconnect with exponential backoff
        this.attemptReconnect();
      };

      // Handle errors
      this.socket.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        this.isConnecting = false;
        this.attemptReconnect();
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      this.disconnect();
      this.attemptReconnect();
    }
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

  attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    
    // Exponential backoff with jitter
    const baseDelay = Math.min(
      this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1),
      this.maxReconnectDelay
    );
    const jitter = Math.random() * 1000; // Add up to 1s of jitter
    const delay = Math.min(baseDelay + jitter, this.maxReconnectDelay);
    
    console.log(`[WebSocket] Attempting to reconnect in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    this.reconnectTimeout = setTimeout(() => {
      if (this.token) {
        console.log(`[WebSocket] Reconnecting (attempt ${this.reconnectAttempts})...`);
        this.connect(this.token);
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
      params: { start: formattedStart, end: formattedEnd, granularity }
    });
    
    const response = await api.get(`/candlestick/${token}`, {
      params: {
        start: formattedStart,
        end: formattedEnd,
        granularity,
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