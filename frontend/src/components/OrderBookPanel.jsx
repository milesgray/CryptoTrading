import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as echarts from 'echarts';
import { webSocketService, getBookPressure } from '../services/api';

const OrderBookPanel = ({ token, latestPriceData }) => {
  const [orderBookData, setOrderBookData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [pressureAnalysis, setPressureAnalysis] = useState(null);
  
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    if (!token) return;
    
    const fetchPressure = async () => {
      try {
        const data = await getBookPressure(token);
        if (data) {
          setPressureAnalysis(data);
        }
      } catch (error) {
        console.error('Error fetching pressure analysis:', error);
      }
    };
    
    fetchPressure();
    const interval = setInterval(fetchPressure, 1000);
    return () => clearInterval(interval);
  }, [token]);

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
  const { bidsDepth, asksDepth, bidPercentage, askPercentage, midpoint, spread } = useMemo(() => {
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
  }, [displayData, latestPriceData]);

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
          let html = `<div class="font-bold border-b border-slate-800 pb-1 mb-1 text-[10px] text-slate-400">Price: $${params[0].value[0].toFixed(2)}</div>`;
          params.forEach(p => {
            const colorClass = p.seriesName === 'Bids' ? 'text-emerald-400' : 'text-rose-400';
            html += `<div class="flex justify-between gap-4 text-[10px]">
              <span class="${colorClass}">${p.seriesName}:</span>
              <span class="font-bold">${p.value[1].toFixed(4)}</span>
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

  // Get recommendation details
  const getRecommendationDetails = (rec) => {
    switch (rec) {
      case 'SCALP_LONG':
        return {
          title: 'SCALP LONG ENTRY',
          subtitle: 'Imbalance detected. High probability long scalp.',
          styleClass: 'border-emerald-500/30 bg-emerald-950/20 text-emerald-400',
          indicatorColor: 'bg-emerald-500',
          badgeText: 'BUY ENTRY'
        };
      case 'SCALP_SHORT':
        return {
          title: 'SCALP SHORT ENTRY',
          subtitle: 'Imbalance detected. High probability short scalp.',
          styleClass: 'border-rose-500/30 bg-rose-950/20 text-rose-400',
          indicatorColor: 'bg-rose-500',
          badgeText: 'SHORT ENTRY'
        };
      case 'SCALP_LONG_CAUTION':
        return {
          title: 'SCALP LONG (CAUTION)',
          subtitle: 'Strong trend in high volatility. Watch stop-loss.',
          styleClass: 'border-amber-500/30 bg-amber-950/20 text-amber-400',
          indicatorColor: 'bg-amber-500',
          badgeText: 'BUY ENTRY (CAUTION)'
        };
      case 'SCALP_SHORT_CAUTION':
        return {
          title: 'SCALP SHORT (CAUTION)',
          subtitle: 'Strong trend in high volatility. Watch stop-loss.',
          styleClass: 'border-amber-500/30 bg-amber-950/20 text-amber-400',
          indicatorColor: 'bg-amber-500',
          badgeText: 'SHORT ENTRY (CAUTION)'
        };
      default:
        return {
          title: 'STANDBY / NO SIGNAL',
          subtitle: 'Order book pressure neutral. Wait for clear imbalance.',
          styleClass: 'border-slate-800 bg-slate-950/40 text-slate-400',
          indicatorColor: 'bg-slate-700',
          badgeText: 'NEUTRAL'
        };
    }
  };

  const recDetails = getRecommendationDetails(pressureAnalysis?.recommendation);

  const getRegimeLabel = (regime) => {
    switch (regime) {
      case 'bull':
        return { text: 'BULLISH REGIME', style: 'text-emerald-400 border-emerald-500/20 bg-emerald-950/30' };
      case 'bear':
        return { text: 'BEARISH REGIME', style: 'text-rose-400 border-rose-500/20 bg-rose-950/30' };
      case 'high_vol':
        return { text: 'HIGH VOLATILITY', style: 'text-amber-400 border-amber-500/20 bg-amber-950/30' };
      case 'low_vol':
        return { text: 'LOW VOLATILITY', style: 'text-sky-400 border-sky-500/20 bg-sky-950/30' };
      default:
        return { text: 'SIDEWAYS RANGE', style: 'text-slate-400 border-slate-850 bg-slate-900/20' };
    }
  };

  const regimeLabel = getRegimeLabel(pressureAnalysis?.market_regime);

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

      {/* SCALP Trading Signals Dashboard Overlay */}
      {pressureAnalysis && (
        <div className={`p-4 rounded-xl border flex flex-col gap-2 transition-all duration-300 ${recDetails.styleClass}`}>
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full animate-pulse ${recDetails.indicatorColor}`} />
              <span className="text-xs font-mono font-extrabold uppercase tracking-wide">{recDetails.title}</span>
            </div>
            <span className={`px-2 py-0.5 rounded text-[8px] font-bold border ${regimeLabel.style}`}>
              {regimeLabel.text}
            </span>
          </div>
          
          <div className="text-[10px] opacity-85 leading-normal font-sans">{recDetails.subtitle}</div>
          
          <div className="flex flex-col gap-1 mt-1">
            <div className="flex justify-between items-center text-[9px] font-mono">
              <span>Signal Confidence:</span>
              <span className="font-bold">{(pressureAnalysis.confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden">
              <div 
                className={`h-full transition-all duration-500 ${recDetails.indicatorColor}`}
                style={{ width: `${pressureAnalysis.confidence * 100}%` }}
              />
            </div>
          </div>
          
          <div className="flex justify-between items-center text-[9px] font-mono opacity-80 pt-1 border-t border-slate-800/40">
            <span>Volatility: {(pressureAnalysis.volatility * 10000).toFixed(1)} bps</span>
            <span className="text-[8px] opacity-70">Index: {pressureAnalysis.total_pressure > 0 ? '+' : ''}{pressureAnalysis.total_pressure.toFixed(2)}</span>
          </div>
        </div>
      )}

      {displayData ? (
        <div className="flex flex-col gap-4 flex-1 justify-between">
          
          {/* Pressure Ratio Meter */}
          <div className="flex flex-col gap-2">
            {/* Raw Depth Meter */}
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-center text-[10px] font-mono font-semibold px-0.5">
                <span className="text-emerald-400">BIDS depth: {bidPercentage.toFixed(0)}%</span>
                <span className={`px-2 py-0.5 rounded border text-[9px] ${pressureLabel.style}`}>
                  {pressureLabel.text}
                </span>
                <span className="text-rose-400">ASKS depth: {askPercentage.toFixed(0)}%</span>
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

            {/* Sigmoid Pressure Balance (Pressure Service Integrated Analysis) */}
            {pressureAnalysis && (
              <div className="flex flex-col gap-1 pt-1 border-t border-slate-850/40">
                <div className="flex justify-between items-center text-[9px] font-mono font-semibold text-slate-450 px-0.5">
                  <span className="text-emerald-400/80">Model Buy: {(pressureAnalysis.buy_pressure * 100).toFixed(0)}%</span>
                  <span className="text-[8px] uppercase tracking-wider text-slate-500">Service Imbalance Analysis</span>
                  <span className="text-rose-400/80">Model Sell: {(pressureAnalysis.sell_pressure * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-1.5 border border-slate-900 overflow-hidden flex">
                  <div 
                    className="bg-emerald-500 h-full transition-all duration-500" 
                    style={{ width: `${pressureAnalysis.buy_pressure * 100}%` }} 
                  />
                  <div 
                    className="bg-rose-500 h-full transition-all duration-500" 
                    style={{ width: `${pressureAnalysis.sell_pressure * 100}%` }} 
                  />
                </div>
              </div>
            )}
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
