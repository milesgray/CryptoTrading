import React, { useEffect, useState, useRef, useCallback } from 'react';
import * as echarts from 'echarts';
import { getCandlestickData } from '../services/api';
import _ from 'lodash';

const RetrievalVisualizer = ({ token }) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  
  // State for Retrieval Settings
  const [k, setK] = useState(3);
  const [segmentLength, setSegmentLength] = useState(60);
  const [frequency, setFrequency] = useState('1m');
  const [orderBookWeight, setOrderBookWeight] = useState(30); // 30% order book, 70% price by default
  
  // Data State
  const [retrievedData, setRetrievedData] = useState([]);
  const [queryPrices, setQueryPrices] = useState([]);
  const [queryCandles, setQueryCandles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Checkbox toggles for each retrieved segment
  const [activeToggles, setActiveToggles] = useState({});

  // Fetch forecast and segment data
  const fetchForecastData = useCallback(async () => {
    if (!token) return;
    
    setLoading(true);
    setError(null);
    try {
      // 1. Fetch live forecast (retrieved segments) from serve proxy
      const forecastRes = await fetch(`/api/retrieval/forecast?symbol=${token}&k=${k}&granularity=${frequency}&window_size=${segmentLength}`);
      if (!forecastRes.ok) {
        let errMsg = "Failed to fetch forecasting data";
        try {
          const errData = await forecastRes.json();
          if (errData && errData.detail) {
            errMsg = errData.detail;
          }
        } catch (_) {}
        throw new Error(errMsg);
      }
      const forecastData = await forecastRes.json();
      
      const segments = forecastData.retrieved || [];
      
      // 2. Fetch the recent price history (query segment) to show as baseline
      const freqToSec = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '1h': 3600
      };
      const granularitySec = freqToSec[frequency] || 60;
      const totalSec = segmentLength * granularitySec;
      const end = new Date();
      // Add 30% buffer to make sure we have enough points and it's robust
      const start = new Date(end.getTime() - (totalSec * 1.3) * 1000);
      const candles = await getCandlestickData(token, start, end, granularitySec);
      
      if (candles && candles.length > 0) {
        const slicedCandles = candles.slice(-segmentLength);
        const recentPrices = slicedCandles.map(c => parseFloat(c.close));
        if (recentPrices.length < segmentLength) {
          throw new Error(`Insufficient live query data. Need ${segmentLength} points, but only found ${recentPrices.length}.`);
        }
        setQueryPrices(recentPrices);
        setQueryCandles(slicedCandles);
      } else {
        throw new Error("No recent price history found to establish current pattern baseline.");
      }

      // Process similarity scores and metrics
      const processedSegments = segments.map((seg, idx) => {
        const similarity = seg.similarity || 0;
        const prices = seg.prices || [];
        const startPrice = prices[0] || 1;
        const endPrice = prices[prices.length - 1] || 1;
        const pctReturn = ((endPrice - startPrice) / startPrice) * 100;
        
        return {
          ...seg,
          index: idx,
          similarity: similarity,
          pctReturn: pctReturn,
          direction: pctReturn >= 0 ? 'BULLISH' : 'BEARISH'
        };
      });

      setRetrievedData(processedSegments);
      
      // Initialize toggles: by default all retrieved segments are checked (active)
      const initialToggles = {};
      processedSegments.forEach(seg => {
        initialToggles[seg.id || seg.index] = true;
      });
      setActiveToggles(initialToggles);
      
    } catch (err) {
      console.error(err);
      setError(err.message || "Unable to load forecasting patterns. Ensure retrieval service is running.");
    } finally {
      setLoading(false);
    }
  }, [token, k, frequency, segmentLength]);

  // Handle toggle change
  const handleToggleChange = (id) => {
    setActiveToggles(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  // Trigger data fetch on token, k, frequency, or segmentLength change
  useEffect(() => {
    fetchForecastData();
  }, [token, k, frequency, segmentLength, fetchForecastData]);

  // Recalculate and render the chart whenever active segments or data changes
  useEffect(() => {
    if (!chartRef.current || queryCandles.length === 0) return;

    // Initialize ECharts instance if not done
    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    // Colors for the retrieved segments
    const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];
    
    const activeSegments = retrievedData.filter(seg => activeToggles[seg.id || seg.index]);
    const forecastLength = activeSegments.length > 0
      ? Math.min(...activeSegments.map(s => (s.prices || []).length))
      : segmentLength; // fallback

    const stepUnit = frequency.endsWith('m') ? 'm' : 'h';
    const stepMultiplier = parseInt(frequency) || 1;

    // X-axis: segmentLength historical steps + forecastLength forecasted steps
    const xAxisData = [];
    for (let i = 0; i < segmentLength; i++) {
      xAxisData.push(`-${(segmentLength - i) * stepMultiplier}${stepUnit}`);
    }
    for (let i = 1; i <= forecastLength; i++) {
      xAxisData.push(`+${i * stepMultiplier}${stepUnit}`);
    }

    // Calculate average relative shadows from queryCandles
    let avgRelUpper = 0.001; // default fallback (0.1%)
    let avgRelLower = 0.001; // default fallback (0.1%)
    if (queryCandles.length > 0) {
      let sumRelUpper = 0;
      let sumRelLower = 0;
      queryCandles.forEach(c => {
        const o = parseFloat(c.open);
        const h = parseFloat(c.high);
        const l = parseFloat(c.low);
        const cl = parseFloat(c.close);
        
        const upperShadow = h - Math.max(o, cl);
        const lowerShadow = Math.min(o, cl) - l;
        
        sumRelUpper += cl > 0 ? (upperShadow / cl) : 0;
        sumRelLower += cl > 0 ? (lowerShadow / cl) : 0;
      });
      avgRelUpper = sumRelUpper / queryCandles.length;
      avgRelLower = sumRelLower / queryCandles.length;
      
      // Enforce realistic bounds/fallbacks
      if (isNaN(avgRelUpper) || avgRelUpper < 0.0001) avgRelUpper = 0.0005;
      if (isNaN(avgRelLower) || avgRelLower < 0.0001) avgRelLower = 0.0005;
    }

    // Format historical candles for ECharts: [open, close, low, high]
    const historicalCandleData = queryCandles.map(c => [
      parseFloat(c.open),
      parseFloat(c.close),
      parseFloat(c.low),
      parseFloat(c.high)
    ]);
    
    // Pad historical candle data with nulls for the forecast steps
    const paddedHistoricalData = [...historicalCandleData];
    for (let i = 0; i < forecastLength; i++) {
      paddedHistoricalData.push(null);
    }

    // 1. Historical baseline series
    const series = [
      {
        name: 'Current Pattern',
        type: 'candlestick',
        data: paddedHistoricalData,
        itemStyle: {
          color: '#10b981',        // Bullish fill (Emerald Green)
          color0: '#f43f5e',       // Bearish fill (Rose Red)
          borderColor: '#10b981',  // Bullish border
          borderColor0: '#f43f5e'  // Bearish border
        },
        zIndex: 10
      }
    ];

    const lastQueryPrice = queryPrices[queryPrices.length - 1] || 0;

    // 2. Add retrieved patterns (only if checked/active)
    activeSegments.forEach((segment) => {
      const prices = segment.prices || [];
      if (prices.length === 0) return;

      // Rescale and align segment to start exactly at the last historical price point
      const meanSeg = prices.reduce((a, b) => a + b, 0) / prices.length;
      const stdSeg = Math.sqrt(prices.reduce((a, b) => a + Math.pow(b - meanSeg, 2), 0) / prices.length) || 1e-8;
      
      // Standardize and rescale using a 1.5% volatility multiplier
      const scaleMultiplier = lastQueryPrice * 0.015;
      const alignedForecast = prices.map(p => {
        const standardized = (p - meanSeg) / stdSeg;
        return lastQueryPrice + standardized * scaleMultiplier;
      });

      // Construct forecasted candles continuing from history's last close
      const retrievedCandles = Array(segmentLength).fill(null);
      let prevClose = lastQueryPrice;
      for (let t = 0; t < forecastLength; t++) {
        const currentClose = alignedForecast[t];
        const open = prevClose;
        const close = currentClose;
        const high = Math.max(open, close) + close * avgRelUpper;
        const low = Math.min(open, close) - close * avgRelLower;
        retrievedCandles.push([open, close, low, high]);
        prevClose = currentClose;
      }

      series.push({
        name: `Pattern #${segment.index + 1}`,
        type: 'candlestick',
        data: retrievedCandles,
        itemStyle: {
          color: '#22d3ee',         // Bullish fill (Cyan)
          color0: '#0891b2',        // Bearish fill (Dark Cyan)
          borderColor: '#22d3ee',   // Bullish border
          borderColor0: '#0891b2',  // Bearish border
          opacity: 0.35             // Partially opaque
        },
        zIndex: 2
      });
    });

    // 3. Consensus Forecast: Average of all checked segments
    if (activeSegments.length > 0) {
      if (forecastLength > 0) {
        const consensusPrices = [];
        
        for (let t = 0; t < forecastLength; t++) {
          let sum = 0;
          activeSegments.forEach(seg => {
            const prices = seg.prices || [];
            const meanSeg = prices.reduce((a, b) => a + b, 0) / prices.length;
            const stdSeg = Math.sqrt(prices.reduce((a, b) => a + Math.pow(b - meanSeg, 2), 0) / prices.length) || 1e-8;
            const scaleMultiplier = lastQueryPrice * 0.015;
            
            const alignedPrice = lastQueryPrice + ((prices[t] - meanSeg) / stdSeg) * scaleMultiplier;
            sum += alignedPrice;
          });
          consensusPrices.push(sum / activeSegments.length);
        }

        const consensusCandles = Array(segmentLength).fill(null);
        let consensusPrevClose = lastQueryPrice;
        for (let t = 0; t < forecastLength; t++) {
          const currentClose = consensusPrices[t];
          const open = consensusPrevClose;
          const close = currentClose;
          const high = Math.max(open, close) + close * avgRelUpper;
          const low = Math.min(open, close) - close * avgRelLower;
          consensusCandles.push([open, close, low, high]);
          consensusPrevClose = currentClose;
        }
        
        series.push({
          name: 'Consensus Projection',
          type: 'candlestick',
          data: consensusCandles,
          itemStyle: {
            color: '#a78bfa',        // Bullish fill (Lavender/Purple)
            color0: '#7c3aed',       // Bearish fill (Deep Violet)
            borderColor: '#a78bfa',  // Bullish border
            borderColor0: '#7c3aed'  // Bearish border
          },
          zIndex: 5
        });
      }
    }

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0f172a', // slate-900
        borderColor: '#1e293b',     // slate-800
        textStyle: { color: '#f8fafc' },
        formatter: (params) => {
          let html = `<div class="font-semibold text-slate-200 border-b border-slate-700 pb-1 mb-1 font-mono">${params[0].name}</div>`;
          params.forEach(p => {
            if (p.value !== undefined && p.value !== null) {
              let valueStr = '';
              if (Array.isArray(p.value)) {
                const val = p.value;
                const offset = val.length === 5 ? 1 : 0;
                const open = parseFloat(val[offset]);
                const close = parseFloat(val[offset + 1]);
                const low = parseFloat(val[offset + 2]);
                const high = parseFloat(val[offset + 3]);
                
                if (!isNaN(open) && !isNaN(close)) {
                  valueStr = `
                    <div class="flex flex-col ml-4 border-l border-slate-700 pl-2 text-[10px] text-slate-400">
                      <div>Open: <span class="font-mono text-slate-350">$${open.toFixed(2)}</span></div>
                      <div>Close: <span class="font-mono text-slate-200 font-bold">$${close.toFixed(2)}</span></div>
                      <div>Low: <span class="font-mono text-slate-450">$${low.toFixed(2)}</span></div>
                      <div>High: <span class="font-mono text-slate-450">$${high.toFixed(2)}</span></div>
                    </div>
                  `;
                }
              } else if (typeof p.value === 'number') {
                valueStr = `<span class="font-mono font-medium" style="color:${p.color}">$${p.value.toFixed(2)}</span>`;
              } else {
                valueStr = `<span class="font-mono font-medium" style="color:${p.color}">${p.value}</span>`;
              }
              
              html += `<div class="flex flex-col gap-0.5 text-xs mt-1">
                <div class="flex justify-between gap-4 font-semibold" style="color:${p.color}">
                  <span>${p.seriesName}:</span>
                  ${!Array.isArray(p.value) ? valueStr : ''}
                </div>
                ${Array.isArray(p.value) ? valueStr : ''}
              </div>`;
            }
          });
          return html;
        }
      },
      legend: {
        data: series.map(s => s.name),
        bottom: 0,
        icon: 'roundRect',
        textStyle: { color: '#94a3b8', fontSize: 11 }
      },
      grid: {
        top: '8%',
        left: '4%',
        right: '4%',
        bottom: '12%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        boundaryGap: true,
        data: xAxisData,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 }
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLine: { show: false },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 }
      },
      series: series
    };

    chartInstance.current.setOption(option);

    // Responsive resize
    const handleResize = () => {
      chartInstance.current?.resize();
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);

  }, [queryPrices, queryCandles, retrievedData, activeToggles, segmentLength, frequency]);

  // Clean up chart on unmount
  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
    };
  }, []);

  // Compute Summary Statistics based on active (checked) segments
  const getSummaryStats = () => {
    const activeSegments = retrievedData.filter(seg => activeToggles[seg.id || seg.index]);
    const lastQueryPrice = queryPrices[queryPrices.length - 1] || 0;

    if (activeSegments.length === 0) {
      return {
        expectedReturn: 0,
        bullRatio: 0,
        volatility: 0,
        avgSimilarity: 0,
        nextCandleColor: 'NEUTRAL',
        nextCandleConfidence: 50,
        upTicks: 0,
        activeCount: 0
      };
    }

    const returns = activeSegments.map(s => s.pctReturn);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const bulls = activeSegments.filter(s => s.pctReturn >= 0).length;
    const bullRatio = (bulls / activeSegments.length) * 100;
    
    // Volatility (std dev of returns)
    const avgRet = avgReturn;
    const variance = returns.reduce((a, b) => a + Math.pow(b - avgRet, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance);

    // Similarity
    const similarities = activeSegments.map(s => s.similarity);
    const avgSimilarity = (similarities.reduce((a, b) => a + b, 0) / similarities.length) * 100;

    // Next Candle Color Prediction (Up/Down consensus on the very first forecasted step)
    let upTicks = 0;
    activeSegments.forEach(seg => {
      const prices = seg.prices || [];
      if (prices.length === 0) return;
      
      const meanSeg = prices.reduce((a, b) => a + b, 0) / prices.length;
      const stdSeg = Math.sqrt(prices.reduce((a, b) => a + Math.pow(b - meanSeg, 2), 0) / prices.length) || 1e-8;
      const scaleMultiplier = lastQueryPrice * 0.015;
      
      const firstAlignedPrice = lastQueryPrice + ((prices[0] - meanSeg) / stdSeg) * scaleMultiplier;
      if (firstAlignedPrice >= lastQueryPrice) {
        upTicks++;
      }
    });

    const downTicks = activeSegments.length - upTicks;
    let nextCandleColor = 'NEUTRAL';
    let nextCandleConfidence = 50;

    if (upTicks > downTicks) {
      nextCandleColor = 'GREEN';
      nextCandleConfidence = (upTicks / activeSegments.length) * 100;
    } else if (downTicks > upTicks) {
      nextCandleColor = 'RED';
      nextCandleConfidence = (downTicks / activeSegments.length) * 100;
    }

    return {
      expectedReturn: avgReturn,
      bullRatio: bullRatio,
      volatility: volatility,
      avgSimilarity: avgSimilarity,
      nextCandleColor,
      nextCandleConfidence,
      upTicks,
      activeCount: activeSegments.length
    };
  };

  const stats = getSummaryStats();
  const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];

  return (
    <div className="bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md flex flex-col gap-6">
      <div className="flex flex-wrap justify-between items-center gap-4 border-b border-slate-850 pb-4">
        <div>
          <h2 className="text-base font-semibold text-slate-350">Pattern Matching & Retrieval Forecast</h2>
          <p className="text-xs text-slate-500">Retrieves historical cycles matching the current price momentum and order book depth</p>
        </div>
        <button 
          onClick={fetchForecastData}
          disabled={loading}
          className="px-4 py-1.5 bg-indigo-950/40 text-indigo-400 border border-indigo-800/20 hover:bg-indigo-900/40 font-semibold rounded-lg text-xs transition disabled:opacity-50"
        >
          {loading ? 'Retrieving...' : 'Trigger Query'}
        </button>
      </div>

      {error ? (
        <div className="p-4 bg-rose-950/40 text-rose-400 border border-rose-800/20 rounded-xl text-xs font-mono">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left: Chart & Stats */}
        <div className="xl:col-span-2 flex flex-col gap-4">
          {/* Chart Container */}
          <div className="relative border border-slate-800 rounded-xl p-4 bg-slate-950/50">
            {loading && (
              <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm flex items-center justify-center z-20 rounded-xl">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-650 mx-auto"></div>
                  <p className="mt-2 text-xs text-slate-400 font-mono">Retrieving similar sequences...</p>
                </div>
              </div>
            )}
            <div ref={chartRef} className="w-full" style={{ minHeight: '380px', height: '380px' }}></div>
          </div>

          {/* Next Candle Predictor Hero Card */}
          <div className="flex flex-col sm:flex-row gap-4 p-4 border border-slate-850 rounded-xl bg-gradient-to-r from-slate-950/60 to-slate-900/60 items-center justify-between shadow-inner">
            <div className="flex items-center gap-4">
              <div className={`w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold border shadow-sm ${
                stats.nextCandleColor === 'GREEN' ? 'bg-emerald-950/40 text-emerald-400 border-emerald-500/10 shadow-emerald-950/10' : 
                stats.nextCandleColor === 'RED' ? 'bg-rose-950/40 text-rose-400 border-rose-500/10 shadow-rose-950/10' : 
                'bg-slate-950 text-slate-500 border-slate-800'
              }`}>
                {stats.nextCandleColor === 'GREEN' ? '▲' : stats.nextCandleColor === 'RED' ? '▼' : '◆'}
              </div>
              <div>
                <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider font-mono">PREDICTED NEXT CANDLE COLOR</span>
                <h3 className={`text-base font-extrabold tracking-tight ${
                  stats.nextCandleColor === 'GREEN' ? 'text-emerald-400' : 
                  stats.nextCandleColor === 'RED' ? 'text-rose-400' : 
                  'text-slate-500'
                }`}>
                  {stats.nextCandleColor === 'GREEN' ? 'GREEN (UP)' : stats.nextCandleColor === 'RED' ? 'RED (DOWN)' : 'NEUTRAL (FLAT)'}
                </h3>
              </div>
            </div>
            <div className="flex flex-col items-start sm:items-end gap-1 w-full sm:w-auto">
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider font-mono">CONSENSUS CONFIDENCE</span>
              <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                <span className="text-base font-extrabold text-slate-300 font-mono">{stats.nextCandleConfidence.toFixed(0)}%</span>
                <div className="w-24 bg-slate-900 rounded-full h-2 border border-slate-850">
                  <div 
                    className={`h-2 rounded-full ${stats.nextCandleColor === 'GREEN' ? 'bg-emerald-500' : stats.nextCandleColor === 'RED' ? 'bg-rose-500' : 'bg-slate-500'}`}
                    style={{ width: `${stats.nextCandleConfidence}%` }}
                  ></div>
                </div>
              </div>
              <span className="text-[9px] text-slate-500 font-mono">
                ({stats.upTicks} of {stats.activeCount} active historical matches agree)
              </span>
            </div>
          </div>

          {/* Summary Statistics Card */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 border border-slate-850 rounded-xl bg-gradient-to-r from-slate-950/60 to-slate-900/60">
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-500 font-semibold font-mono uppercase tracking-wider">EXPECTED RETURN</span>
              <span className={`text-base font-bold font-mono mt-0.5 ${stats.expectedReturn >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {stats.expectedReturn >= 0 ? '+' : ''}{stats.expectedReturn.toFixed(2)}%
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-500 font-semibold font-mono uppercase tracking-wider">BULLISH CONSENSUS</span>
              <span className="text-base font-bold font-mono mt-0.5 text-slate-300">
                {stats.bullRatio.toFixed(0)}%
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-500 font-semibold font-mono uppercase tracking-wider">UNCERTAINTY (VOL)</span>
              <span className="text-base font-bold font-mono mt-0.5 text-indigo-400">
                {stats.volatility.toFixed(2)}%
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-500 font-semibold font-mono uppercase tracking-wider">MATCH STRENGTH</span>
              <span className="text-base font-bold font-mono mt-0.5 text-emerald-450">
                {stats.avgSimilarity.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* Right: Settings & Toggles */}
        <div className="xl:col-span-1 flex flex-col gap-6 border-t xl:border-t-0 xl:border-l border-slate-850 xl:pl-6 pt-6 xl:pt-0">
          
          {/* Settings Section */}
          <div className="flex flex-col gap-4">
            <h3 className="text-xs font-bold text-slate-350 uppercase tracking-wider border-b border-slate-850 pb-2 font-mono">Retrieval Settings</h3>
            
            {/* number of segments (k) */}
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-xs font-semibold text-gray-600">
                <label htmlFor="k-slider">Number of Patterns (k)</label>
                <span className="font-mono text-indigo-600">{k} segments</span>
              </div>
              <input 
                id="k-slider"
                type="range" 
                min="1" 
                max="5" 
                value={k}
                onChange={(e) => setK(parseInt(e.target.value))}
                className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
            </div>

            {/* length of segment */}
            <div className="flex flex-col gap-1">
              <label htmlFor="len-select" className="text-[10px] text-slate-500 uppercase font-semibold">Segment Length</label>
              <select 
                id="len-select"
                value={segmentLength}
                onChange={(e) => setSegmentLength(parseInt(e.target.value))}
                className="mt-1 block w-full rounded-lg bg-slate-950 border border-slate-800 text-slate-300 text-xs py-1.5 px-3 focus:outline-none focus:border-slate-700"
              >
                <option value={15}>15 steps (Short-term)</option>
                <option value={30}>30 steps (Standard)</option>
                <option value={60}>60 steps (Recommended)</option>
                <option value={120}>120 steps (Macro)</option>
              </select>
            </div>

            {/* frequency */}
            <div className="flex flex-col gap-1">
              <label htmlFor="freq-select" className="text-[10px] text-slate-500 uppercase font-semibold">Retrieval Frequency</label>
              <select 
                id="freq-select"
                value={frequency}
                onChange={(e) => setFrequency(e.target.value)}
                className="mt-1 block w-full rounded-lg bg-slate-950 border border-slate-800 text-slate-300 text-xs py-1.5 px-3 focus:outline-none focus:border-slate-700"
              >
                <option value="1m">1 Minute</option>
                <option value="5m">5 Minutes</option>
                <option value="15m">15 Minutes</option>
                <option value="1h">1 Hour</option>
              </select>
            </div>

            {/* weight */}
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-[10px] text-slate-500 uppercase font-semibold">
                <label htmlFor="weight-slider">Order Book Weight</label>
                <span className="font-mono text-indigo-400">{orderBookWeight}%</span>
              </div>
              <input 
                id="weight-slider"
                type="range" 
                min="0" 
                max="100" 
                value={orderBookWeight}
                onChange={(e) => setOrderBookWeight(parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-650"
              />
              <div className="flex justify-between text-[9px] font-mono text-slate-500">
                <span>100% Price Shape</span>
                <span>100% Depth Imbalance</span>
              </div>
            </div>
          </div>

          {/* Toggles Section */}
          <div className="flex flex-col gap-4">
            <h3 className="text-xs font-bold text-slate-350 uppercase tracking-wider border-b border-slate-850 pb-2 font-mono">Overlay Segments</h3>
            
            <div className="flex flex-col gap-2.5 max-h-[220px] overflow-y-auto pr-1">
              {retrievedData.map((segment) => {
                const segId = segment.id || segment.index;
                const isActive = !!activeToggles[segId];
                const color = colors[segment.index % colors.length];
                
                return (
                  <div 
                    key={segId} 
                    className="flex items-center justify-between p-2.5 rounded-lg border border-slate-850 bg-slate-950/20 hover:bg-slate-900/40 transition"
                  >
                    <div className="flex items-center gap-3">
                      <input 
                        type="checkbox" 
                        id={`seg-${segId}`}
                        checked={isActive}
                        onChange={() => handleToggleChange(segId)}
                        className="h-4 w-4 text-indigo-600 border-slate-800 bg-slate-950 rounded focus:ring-indigo-500 cursor-pointer"
                      />
                      <label 
                        htmlFor={`seg-${segId}`}
                        className="flex items-center gap-2 text-xs font-semibold text-slate-350 cursor-pointer"
                      >
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }}></span>
                        Pattern #{segment.index + 1}
                      </label>
                    </div>
                    
                    <div className="flex items-center gap-2 font-mono">
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold border ${
                        segment.direction === 'BULLISH' 
                          ? 'bg-emerald-950/40 text-emerald-400 border-emerald-500/10' 
                          : 'bg-rose-950/40 text-rose-400 border-rose-500/10'
                      }`}>
                        {segment.pctReturn >= 0 ? '+' : ''}{segment.pctReturn.toFixed(1)}%
                      </span>
                      <span className="text-xs font-bold text-indigo-400">
                        {(segment.similarity * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                );
              })}
              
              {retrievedData.length === 0 && !loading && (
                <div className="text-center py-6 text-xs text-slate-500 font-mono">
                  No segments retrieved. Press "Trigger Query".
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default RetrievalVisualizer;