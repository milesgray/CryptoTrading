import React, { useState, useEffect } from 'react';
import { webSocketService } from '../services/api';

const OrderBookPanel = ({ token, latestPriceData }) => {
  const [orderBookData, setOrderBookData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    let cleanupCallback;

    const connectWebSocket = async () => {
      try {
        await webSocketService.connect(token);
        setIsConnected(true);
        
        cleanupCallback = webSocketService.onOrderBookUpdate((data) => {
          console.log('Order book update received:', data);
          setOrderBookData(data);
        });

        // Also listen for price updates that might contain order book data
        const priceCleanup = webSocketService.onPriceUpdate((data) => {
          if (data.order_book) {
            setOrderBookData(data.order_book);
          }
        });

        return () => {
          if (cleanupCallback) cleanupCallback();
          if (priceCleanup) priceCleanup();
        };
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setIsConnected(false);
      }
    };

    connectWebSocket();

    return () => {
      if (cleanupCallback) cleanupCallback();
      webSocketService.disconnect();
    };
  }, [token]);

  // Use order book data from latest price if WebSocket hasn't received data yet
  const displayData = orderBookData || latestPriceData?.order_book;

  const formatNumber = (num, decimals = 2) => {
    if (num === null || num === undefined) return 'N/A';
    return num.toFixed(decimals);
  };

  const renderPriceBucket = (bucket, side) => {
    if (!bucket) return null;
    
    return (
      <div className={`flex justify-between items-center py-1 px-2 text-sm ${
        side === 'bid' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
      }`}>
        <span>{formatNumber(bucket.avg_price)}</span>
        <span className="font-medium">{formatNumber(bucket.volume, 0)}</span>
        <span className="text-xs text-gray-600">{bucket.range}</span>
      </div>
    );
  };

  const renderPriceOutlier = (outlier, side) => {
    if (!outlier) return null;
    
    return (
      <div className={`flex justify-between items-center py-1 px-2 text-xs ${
        side === 'bid' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
      } border-l-4 ${side === 'bid' ? 'border-green-500' : 'border-red-500'}`}>
        <span>{formatNumber(outlier.price)}</span>
        <span>{formatNumber(outlier.volume, 0)}</span>
      </div>
    );
  };

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">Order Book - {token}</h2>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-gray-400'}`}></div>
          <span className="text-xs text-gray-600">
            {isConnected ? 'Live' : 'Offline'}
          </span>
        </div>
      </div>

      {displayData ? (
        <div className="space-y-4">
          {/* Volume Summary */}
          <div className="bg-gray-50 rounded p-3">
            <div className="text-sm font-medium text-gray-700 mb-1">Total Volume</div>
            <div className="text-xl font-bold text-gray-900">
              {formatNumber(displayData.volume, 0)}
            </div>
          </div>

          {/* Bid Buckets */}
          {displayData.bid_buckets && displayData.bid_buckets.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Bid Buckets</h3>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {displayData.bid_buckets.slice(0, 5).map((bucket, index) => (
                  <div key={`bid-${index}`}>
                    {renderPriceBucket(bucket, 'bid')}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Ask Buckets */}
          {displayData.ask_buckets && displayData.ask_buckets.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Ask Buckets</h3>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {displayData.ask_buckets.slice(0, 5).map((bucket, index) => (
                  <div key={`ask-${index}`}>
                    {renderPriceBucket(bucket, 'ask')}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Bid Outliers */}
          {displayData.bid_outliers && displayData.bid_outliers.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Bid Outliers</h3>
              <div className="space-y-1 max-h-24 overflow-y-auto">
                {displayData.bid_outliers.slice(0, 3).map((outlier, index) => (
                  <div key={`bid-outlier-${index}`}>
                    {renderPriceOutlier(outlier, 'bid')}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Ask Outliers */}
          {displayData.ask_outliers && displayData.ask_outliers.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Ask Outliers</h3>
              <div className="space-y-1 max-h-24 overflow-y-auto">
                {displayData.ask_outliers.slice(0, 3).map((outlier, index) => (
                  <div key={`ask-outlier-${index}`}>
                    {renderPriceOutlier(outlier, 'ask')}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Market Stats */}
          {latestPriceData?.metadata && (
            <div className="border-t pt-3">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Market Stats</h3>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {latestPriceData.metadata.spread !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Spread:</span>
                    <span className="font-medium">{formatNumber(latestPriceData.metadata.spread)}</span>
                  </div>
                )}
                {latestPriceData.metadata.midpoint !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Midpoint:</span>
                    <span className="font-medium">{formatNumber(latestPriceData.metadata.midpoint)}</span>
                  </div>
                )}
                {latestPriceData.metadata.lowest_bid !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Best Bid:</span>
                    <span className="font-medium text-green-600">{formatNumber(latestPriceData.metadata.lowest_bid)}</span>
                  </div>
                )}
                {latestPriceData.metadata.lowest_ask !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Best Ask:</span>
                    <span className="font-medium text-red-600">{formatNumber(latestPriceData.metadata.lowest_ask)}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-8">
          <div className="text-gray-500 text-sm">No order book data available</div>
          <div className="text-gray-400 text-xs mt-1">
            {isConnected ? 'Waiting for data...' : 'Connect to see live data'}
          </div>
        </div>
      )}
    </div>
  );
};

export default OrderBookPanel;
