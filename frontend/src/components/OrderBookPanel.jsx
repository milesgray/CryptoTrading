import React, { useState, useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { webSocketService } from '../services/api';

const OrderBookPanel = ({ token, latestPriceData }) => {
  const [orderBookData, setOrderBookData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

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

  const displayData = orderBookData || latestPriceData?.order_book;

  const formatNumber = (num, decimals = 2) => {
    if (num === null || num === undefined) return 'N/A';
    return num.toFixed(decimals);
  };

  // 1. Calculate pressure stats and depth curves
  const getDepthStats = () => {
    if (!displayData || !displayData.bid_buckets || !displayData.ask_buckets) {
      return { bidsDepth: [], asksDepth: [], bidPercentage: 50, askPercentage: 50, midpoint: 0, spread: 0 };
    }

    const sortedBids = [...displayData.bid_buckets].sort((a, b) => b.avg_price - a.avg_price);
    let bidCumulative = 0;
    const bidsDepth = sortedBids.map(b => {
      bidCumulative += b.volume;
      return [b.avg_price, bidCumulative];
    }).sort((a, b) => a[0] - b[0]);

    const sortedAsks = [...displayData.ask_buckets].sort((a, b) => a.avg_price - b.avg_price);
    let askCumulative = 0;
    const asksDepth = sortedAsks.map(a => {
      askCumulative += a.volume;
      return [a.avg_price, askCumulative];
    }).sort((a, b) => a[0] - b[0]);

    const totalBidVol = displayData.bid_buckets.reduce((acc, curr) => acc + curr.volume, 0);
    const totalAskVol = displayData.ask_buckets.reduce((acc, curr) => acc + curr.volume, 0);
    const totalVol = totalBidVol + totalAskVol;

    const bidPercentage = totalVol > 0 ? (totalBidVol / totalVol) * 100 : 50;
    const askPercentage = totalVol > 0 ? (totalAskVol / totalVol) * 100 : 50;

    const midpoint = latestPriceData?.metadata?.midpoint || 
      (sortedBids.length > 0 && sortedAsks.length > 0 ? (sortedBids[0].avg_price + sortedAsks[0].avg_price) / 2 : 0);
    const spread = latestPriceData?.metadata?.spread ||
      (sortedBids.length > 0 && sortedAsks.length > 0 ? sortedAsks[0].avg_price - sortedBids[0].avg_price : 0);

    return { bidsDepth, asksDepth, bidPercentage, askPercentage, midpoint, spread };
  };

  const { bidsDepth, asksDepth, bidPercentage, askPercentage, midpoint, spread } = getDepthStats();

  // 2. Render and update the ECharts depth chart
  useEffect(() => {
    if (!chartRef.current || bidsDepth.length === 0 || asksDepth.length === 0) return;

    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    const option = {
      grid: { left: '3%', right: '3%', top: '10%', bottom: '15%', containLabel: true },
      xAxis: {
        type: 'value',
        scale: true,
        axisLabel: {
          color: '#64748b',
          fontSize: 9,
          fontFamily: 'monospace',
          formatter: (val) => `$${val.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
        },
        axisLine: { lineStyle: { color: '#334155' } },
        splitLine: { show: false }
      },
      yAxis: {
        type: 'value',
        position: 'right',
        axisLabel: {
          color: '#64748b',
          fontSize: 9,
          fontFamily: 'monospace'
        },
        axisLine: { lineStyle: { color: '#334155' } },
        splitLine: {
          lineStyle: { color: '#1e293b', type: 'dashed' }
        }
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0f172a',
        borderColor: '#1e293b',
        textStyle: { color: '#cbd5e1', fontSize: 11, fontFamily: 'monospace' },
        formatter: (params) => {
          let html = `<div className="font-bold border-b border-slate-800 pb-1 mb-1 text-[10px] text-slate-400">Price: $${params[0].value[0].toFixed(2)}</div>`;
          params.forEach(p => {
            const colorClass = p.seriesName === 'Bids' ? 'text-emerald-400' : 'text-rose-400';
            html += `<div className="flex justify-between gap-4 text-[10px]">
              <span className="${colorClass}">${p.seriesName}:</span>
              <span className="font-bold">${p.value[1].toFixed(4)}</span>
            </div>`;
          });
          return html;
        }
      },
      series: [
        {
          name: 'Bids',
          type: 'line',
          step: 'end',
          data: bidsDepth,
          showSymbol: false,
          lineStyle: { color: '#10b981', width: 1.5 },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(16, 185, 129, 0.25)' },
              { offset: 1, color: 'rgba(16, 185, 129, 0.02)' }
            ])
          }
        },
        {
          name: 'Asks',
          type: 'line',
          step: 'start',
          data: asksDepth,
          showSymbol: false,
          lineStyle: { color: '#ef4444', width: 1.5 },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(239, 68, 68, 0.25)' },
              { offset: 1, color: 'rgba(239, 68, 68, 0.02)' }
            ])
          }
        }
      ],
      backgroundColor: 'transparent'
    };

    // Draw vertical midpoint line if valid
    if (midpoint > 0) {
      option.series[0].markLine = {
        symbol: 'none',
        silent: true,
        lineStyle: { color: '#6366f1', type: 'dashed', width: 1 },
        data: [{ xAxis: midpoint }]
      };
    }

    chartInstance.current.setOption(option);

    const handleResize = () => {
      chartInstance.current?.resize();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [bidsDepth, asksDepth, midpoint]);

  // Clean up chart instance on unmount
  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
    };
  }, []);

  // 3. Find largest whale walls from outlier lists
  const largestBidOutlier = displayData?.bid_outliers && displayData.bid_outliers.length > 0
    ? [...displayData.bid_outliers].sort((a, b) => b.volume - a.volume)[0]
    : null;
  const largestAskOutlier = displayData?.ask_outliers && displayData.ask_outliers.length > 0
    ? [...displayData.ask_outliers].sort((a, b) => b.volume - a.volume)[0]
    : null;

  // Decide market balance sentiment string
  const getPressureLabel = () => {
    if (bidPercentage > 56) return { text: 'BULLISH PRESSURE', style: 'text-emerald-400 border-emerald-500/10 bg-emerald-950/20' };
    if (askPercentage > 56) return { text: 'BEARISH PRESSURE', style: 'text-rose-400 border-rose-500/10 bg-rose-950/20' };
    return { text: 'NEUTRAL BALANCE', style: 'text-slate-400 border-slate-800 bg-slate-900/40' };
  };
  const pressureLabel = getPressureLabel();

  return (
    <div className="bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md h-full flex flex-col gap-5 justify-between">
      {/* Header and status */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-slate-350">Order Depth Map</h2>
          <span className="text-[10px] text-slate-500 font-mono">Cumulative buy & sell walls</span>
        </div>
        
        <div className="flex items-center gap-2 bg-slate-950/60 px-2.5 py-1 rounded-lg border border-slate-800">
          <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`}></div>
          <span className="text-[10px] font-semibold font-mono text-slate-400">
            {isConnected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
      </div>

      {displayData ? (
        <div className="flex flex-col gap-4 flex-1 justify-between">
          
          {/* Pressure Ratio Meter */}
          <div className="flex flex-col gap-1">
            <div className="flex justify-between items-center text-[10px] font-mono font-semibold px-0.5">
              <span className="text-emerald-400">BIDS: {bidPercentage.toFixed(0)}%</span>
              <span className={`px-2 py-0.5 rounded border text-[9px] ${pressureLabel.style}`}>
                {pressureLabel.text}
              </span>
              <span className="text-rose-400">ASKS: {askPercentage.toFixed(0)}%</span>
            </div>
            
            <div className="w-full bg-slate-950 rounded-full h-2 border border-slate-850 overflow-hidden flex">
              <div 
                className="bg-emerald-500 h-full transition-all duration-500" 
                style={{ width: `${bidPercentage}%` }} 
              />
              <div 
                className="bg-rose-500 h-full transition-all duration-500" 
                style={{ width: `${askPercentage}%` }} 
              />
            </div>
          </div>

          {/* Depth Chart Display */}
          <div className="relative bg-slate-950/50 rounded-xl border border-slate-800 p-2 min-h-[220px] flex items-center justify-center">
            {bidsDepth.length > 0 && asksDepth.length > 0 ? (
              <div ref={chartRef} className="w-full h-56" />
            ) : (
              <div className="text-center text-slate-500 text-xs font-mono">
                Generating cumulative depth walls...
              </div>
            )}
          </div>

          {/* Midpoint & Spread summary */}
          <div className="grid grid-cols-2 gap-3 bg-slate-950/30 p-3 rounded-xl border border-slate-850 font-mono text-[11px]">
            <div className="flex flex-col gap-0.5">
              <span className="text-slate-500 text-[9px] uppercase tracking-wider font-semibold">Spread Size</span>
              <span className="font-bold text-slate-300">
                ${formatNumber(spread)}
              </span>
            </div>
            <div className="flex flex-col gap-0.5 text-right">
              <span className="text-slate-500 text-[9px] uppercase tracking-wider font-semibold">Midpoint Price</span>
              <span className="font-bold text-slate-300">
                ${formatNumber(midpoint)}
              </span>
            </div>
          </div>

          {/* Whale Limit Walls Callouts */}
          <div className="flex flex-col gap-2">
            <h3 className="text-[10px] text-slate-500 uppercase font-semibold font-mono tracking-wider">Whale Wall Support / Resistance</h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
              {/* Bid Wall */}
              {largestBidOutlier ? (
                <div className="bg-emerald-950/10 border border-emerald-500/10 p-2 rounded-lg flex items-center gap-2">
                  <div className="h-5 w-5 rounded bg-emerald-500/10 flex items-center justify-center text-emerald-400 font-bold text-[9px]">
                    BUY
                  </div>
                  <div className="flex flex-col">
                    <span className="text-slate-400 text-[9px]">Support Wall</span>
                    <span className="font-bold text-emerald-400">
                      {formatNumber(largestBidOutlier.volume, 2)} {token} @ ${formatNumber(largestBidOutlier.price, 0)}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="bg-slate-950/40 border border-slate-900 p-2 rounded-lg text-[9px] text-slate-500 text-center py-3">
                  No support walls detected
                </div>
              )}

              {/* Ask Wall */}
              {largestAskOutlier ? (
                <div className="bg-rose-950/10 border border-rose-500/10 p-2 rounded-lg flex items-center gap-2">
                  <div className="h-5 w-5 rounded bg-rose-500/10 flex items-center justify-center text-rose-400 font-bold text-[9px]">
                    SELL
                  </div>
                  <div className="flex flex-col">
                    <span className="text-slate-400 text-[9px]">Resistance Wall</span>
                    <span className="font-bold text-rose-400">
                      {formatNumber(largestAskOutlier.volume, 2)} {token} @ ${formatNumber(largestAskOutlier.price, 0)}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="bg-slate-950/40 border border-slate-900 p-2 rounded-lg text-[9px] text-slate-500 text-center py-3">
                  No resistance walls detected
                </div>
              )}
            </div>
          </div>

        </div>
      ) : (
        <div className="text-center py-12 flex flex-col justify-center items-center gap-2 flex-1">
          <div className="text-slate-500 text-sm font-semibold">No order book data available</div>
          <div className="text-slate-650 text-xs font-mono">
            {isConnected ? 'Waiting for updates...' : 'Connect to stream live depth'}
          </div>
        </div>
      )}
    </div>
  );
};

export default OrderBookPanel;
