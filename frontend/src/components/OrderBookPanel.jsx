import React, { useState, useEffect } from 'react';
import { webSocketService } from '../services/api';

const OrderBookPanel = ({ token, latestPriceData }) => {
  const [orderBookData, setOrderBookData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    let cleanupCallback = null;
    let priceCleanup = null;

    const connectWebSocket = async () => {
      try {
        await webSocketService.connect(token);
        setIsConnected(true);
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setIsConnected(false);
      }
    };

    connectWebSocket();

    cleanupCallback = webSocketService.onOrderBookUpdate((data) => {
      console.log('Order book update received:', data);
      setOrderBookData(data);
    });

    // Also listen for price updates that might contain order book data
    priceCleanup = webSocketService.onPriceUpdate((data) => {
      if (data.order_book) {
        setOrderBookData(data.order_book);
      }
    });

    return () => {
      if (cleanupCallback) cleanupCallback();
      if (priceCleanup) priceCleanup();
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
      <div className={`flex justify-between items-center py-1 px-2 text-xs font-mono rounded border ${
        side === 'bid' 
          ? 'bg-emerald-950/20 text-emerald-400 border-emerald-500/10' 
          : 'bg-rose-950/20 text-rose-400 border-rose-500/10'
      }`}>
        <span>{formatNumber(bucket.avg_price)}</span>
        <span className="font-semibold">{formatNumber(bucket.volume, 5)}</span>
        <span className="text-[10px] text-slate-500">{bucket.range}</span>
      </div>
    );
  };

  const renderPriceOutlier = (outlier, side) => {
    if (!outlier) return null;
    
    return (
      <div className={`flex justify-between items-center py-1 px-2 text-xs font-mono rounded ${
        side === 'bid' 
          ? 'bg-emerald-950/40 text-emerald-300 border-l-4 border-emerald-500' 
          : 'bg-rose-950/40 text-rose-300 border-l-4 border-rose-500'
      }`}>
        <span>{formatNumber(outlier.price)}</span>
        <span>{formatNumber(outlier.volume, 5)}</span>
      </div>
    );
  };

  return (
    <div className="bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md h-full flex flex-col justify-between">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold text-slate-300">Order Book - {token}</h2>
        <div className="flex items-center space-x-2 bg-slate-950/60 px-2.5 py-1 rounded-lg border border-slate-800">
          <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`}></div>
          <span className="text-[10px] font-semibold font-mono text-slate-400">
            {isConnected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
      </div>

      {displayData ? (
        <div className="space-y-4">
          {/* Volume Summary */}
          <div className="bg-slate-950/50 border border-slate-850 rounded-xl p-3 flex justify-between items-center">
            <div className="text-[10px] text-slate-500 uppercase font-semibold font-mono">Total Volume</div>
            <div className="text-base font-mono font-bold text-slate-350">
              {formatNumber(displayData.volume, 0)}
            </div>
          </div>

          {/* Bid Buckets */}
          {displayData.bid_buckets && displayData.bid_buckets.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <h3 className="text-[10px] text-slate-500 uppercase font-semibold font-mono">Bid Buckets</h3>
              <div className="space-y-1.5 max-h-32 overflow-y-auto custom-scrollbar">
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
            <div className="flex flex-col gap-1.5">
              <h3 className="text-[10px] text-slate-500 uppercase font-semibold font-mono">Ask Buckets</h3>
              <div className="space-y-1.5 max-h-32 overflow-y-auto custom-scrollbar">
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
            <div className="flex flex-col gap-1.5">
              <h3 className="text-[10px] text-slate-500 uppercase font-semibold font-mono">Bid Outliers</h3>
              <div className="space-y-1.5 max-h-24 overflow-y-auto custom-scrollbar">
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
            <div className="flex flex-col gap-1.5">
              <h3 className="text-[10px] text-slate-500 uppercase font-semibold font-mono">Ask Outliers</h3>
              <div className="space-y-1.5 max-h-24 overflow-y-auto custom-scrollbar">
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
            <div className="border-t border-slate-900 pt-3 flex flex-col gap-1.5">
              <h3 className="text-[10px] text-slate-500 uppercase font-semibold font-mono">Market Stats</h3>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                {latestPriceData.metadata.spread !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-slate-500">Spread:</span>
                    <span className="font-semibold text-slate-350">{formatNumber(latestPriceData.metadata.spread)}</span>
                  </div>
                )}
                {latestPriceData.metadata.midpoint !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-slate-500">Midpoint:</span>
                    <span className="font-semibold text-slate-350">{formatNumber(latestPriceData.metadata.midpoint)}</span>
                  </div>
                )}
                {latestPriceData.metadata.lowest_bid !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-slate-500">Best Bid:</span>
                    <span className="font-semibold text-emerald-400">{formatNumber(latestPriceData.metadata.lowest_bid)}</span>
                  </div>
                )}
                {latestPriceData.metadata.lowest_ask !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-slate-500">Best Ask:</span>
                    <span className="font-semibold text-rose-400">{formatNumber(latestPriceData.metadata.lowest_ask)}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-12 flex flex-col justify-center items-center gap-2">
          <div className="text-slate-500 text-sm font-semibold">No order book data available</div>
          <div className="text-slate-600 text-xs font-mono">
            {isConnected ? 'Waiting for updates...' : 'Connect to stream live depth'}
          </div>
        </div>
      )}
    </div>
  );
};

export default OrderBookPanel;
