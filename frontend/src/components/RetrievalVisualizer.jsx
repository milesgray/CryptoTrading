import React, { useEffect, useState, useRef, useCallback } from 'react';
import * as echarts from 'echarts';
import { getCandlestickData, webSocketService } from '../services/api';
import _ from 'lodash';

const getScaleMultiplier = (queryPrices, lastQueryPrice) => {
  if (!queryPrices || queryPrices.length === 0 || lastQueryPrice === 0) {
    return lastQueryPrice * 0.015; // fallback
  }
  const meanQuery = queryPrices.reduce((a, b) => a + b, 0) / queryPrices.length;
  const varianceQuery = queryPrices.reduce((a, b) => a + Math.pow(b - meanQuery, 2), 0) / queryPrices.length;
  const stdQuery = Math.sqrt(varianceQuery);
  
  // Floor at 0.05% of last price, ceiling at 2.0% of last price to handle extreme conditions
  const minMultiplier = lastQueryPrice * 0.0005;
  const maxMultiplier = lastQueryPrice * 0.02;
  return Math.max(minMultiplier, Math.min(maxMultiplier, stdQuery));
};

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

  // Real-time tracking of actual prices vs forecast
  const [queryEndTime, setQueryEndTime] = useState(null);
  const [liveActualPrices, setLiveActualPrices] = useState([]);
  const [archiveStatus, setArchiveStatus] = useState('idle'); // 'idle', 'saving', 'saved', 'error'

  const queryPricesRef = useRef([]);
  const liveActualPricesRef = useRef([]);
  const archiveStatusRef = useRef('idle');
  const tokenRef = useRef(token);
  const frequencyRef = useRef(frequency);

  // Keep refs synchronized
  useEffect(() => { tokenRef.current = token; }, [token]);
  useEffect(() => { frequencyRef.current = frequency; }, [frequency]);
  useEffect(() => { queryPricesRef.current = queryPrices; }, [queryPrices]);
  useEffect(() => { liveActualPricesRef.current = liveActualPrices; }, [liveActualPrices]);
  useEffect(() => { archiveStatusRef.current = archiveStatus; }, [archiveStatus]);

  // Fetch forecast and segment data
  const fetchForecastData = useCallback(async () => {
    if (!token) return;
    
    // Auto-archive active tracking session before loading new forecast if sufficient data has elapsed
    const currentPrices = queryPricesRef.current;
    const actualPrices = liveActualPricesRef.current;
    const currentStatus = archiveStatusRef.current;
    
    if (currentPrices.length > 0 && actualPrices.length > 0 && currentStatus === 'idle') {
      const validPrices = actualPrices.filter(p => p !== null && p !== undefined);
      if (validPrices.length >= 5) {
        archiveStatusRef.current = 'saving'; // synchronously prevent duplicate archive triggers
        console.log(`[RetrievalVisualizer] Auto-archiving active tracking run of ${validPrices.length} steps for ${tokenRef.current}...`);
        fetch('/api/retrieval/setup/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            symbol: tokenRef.current,
            timeframe: frequencyRef.current,
            prices: currentPrices,
            actual_future_prices: validPrices,
            leverage: 1.0
          })
        }).catch(err => console.error('[RetrievalVisualizer] Auto-archive failed:', err));
      }
    }
    
    setArchiveStatus('idle');
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
        const lastCandle = slicedCandles[slicedCandles.length - 1];
        setQueryEndTime(new Date(lastCandle.timestamp).getTime());
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

      // Initialize live tracking prices array
      const forecastLengths = processedSegments.map(s => (s.prices || []).length);
      const computedForecastLength = forecastLengths.length > 0 ? Math.min(...forecastLengths) : segmentLength;
      setLiveActualPrices(Array(computedForecastLength).fill(null));
      
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

  // Hook into live WebSocket updates for comparing actual values against forecast in real-time
  useEffect(() => {
    if (!token || !queryEndTime) return;

    const freqToSec = {
      '1m': 60,
      '5m': 300,
      '15m': 900,
      '1h': 3600
    };
    const granularitySec = freqToSec[frequency] || 60;

    const handlePriceUpdate = (priceData) => {
      if (!priceData || priceData.price === undefined) return;

      const tickTime = new Date(priceData.timestamp || Date.now()).getTime();
      const diffMs = tickTime - queryEndTime;

      if (diffMs <= 0) {
        // Price update corresponds to the query window or before, ignore
        return;
      }

      // Calculate step index in the forecast window
      const stepIndex = Math.floor(diffMs / (granularitySec * 1000));

      setLiveActualPrices(prev => {
        if (stepIndex >= 0 && stepIndex < prev.length) {
          const nextPrices = [...prev];
          nextPrices[stepIndex] = parseFloat(priceData.price);
          return nextPrices;
        }
        return prev;
      });
    };

    console.log(`[RetrievalVisualizer] Subscribing to live updates for ${token} to track forecast performance`);
    const unsubscribe = webSocketService.onPriceUpdate(handlePriceUpdate);

    return () => {
      console.log(`[RetrievalVisualizer] Unsubscribing from live updates for ${token}`);
      unsubscribe();
    };
  }, [token, queryEndTime, frequency]);

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
      paddedHistoricalData.push('-');
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
    const scaleMultiplier = getScaleMultiplier(queryPrices, lastQueryPrice);

    // Build the Actual Price series line from live price stream
    const actualLineData = Array(segmentLength - 1).fill('-');
    actualLineData.push(lastQueryPrice);

    let lastValidIdx = -1;
    for (let i = liveActualPrices.length - 1; i >= 0; i--) {
      if (liveActualPrices[i] !== null && liveActualPrices[i] !== undefined) {
        lastValidIdx = i;
        break;
      }
    }

    for (let t = 0; t < forecastLength; t++) {
      if (t <= lastValidIdx) {
        let val = liveActualPrices[t];
        if (val === null || val === undefined) {
          let prevVal = lastQueryPrice;
          for (let j = t - 1; j >= 0; j--) {
            if (liveActualPrices[j] !== null && liveActualPrices[j] !== undefined) {
              prevVal = liveActualPrices[j];
              break;
            }
          }
          val = prevVal;
        }
        actualLineData.push(val);
      } else {
        actualLineData.push('-');
      }
    }

    const hasActualPrice = actualLineData.some(val => val !== '-');
    if (hasActualPrice) {
      series.push({
        name: 'Actual Price (Live)',
        type: 'line',
        data: actualLineData,
        smooth: true,
        showSymbol: true,
        symbolSize: 6,
        lineStyle: {
          width: 3.5,
          color: '#f43f5e',
          shadowColor: 'rgba(244, 63, 94, 0.4)',
          shadowBlur: 10,
          shadowOffsetY: 2
        },
        itemStyle: {
          color: '#f43f5e'
        },
        zIndex: 20
      });
    }

    // 2. Add retrieved patterns (only if checked/active)
    activeSegments.forEach((segment) => {
      const prices = segment.prices || [];
      if (prices.length === 0) return;

      // Rescale and align segment to start exactly at the last historical price point
      const histPrices = segment.historical_prices || [];
      const histLast = histPrices.length > 0 ? histPrices[histPrices.length - 1] : (prices[0] || 1);
      const meanSeg = prices.reduce((a, b) => a + b, 0) / prices.length;
      const stdSeg = Math.sqrt(prices.reduce((a, b) => a + Math.pow(b - meanSeg, 2), 0) / prices.length) || 1e-8;
      
      // Standardize and rescale using the dynamic query-based volatility scale relative to history's last close
      const alignedForecast = prices.map(p => {
        const standardized = (p - histLast) / stdSeg;
        return lastQueryPrice + standardized * scaleMultiplier;
      });

      // Construct forecasted line continuing from history's last close
      const retrievedLineData = Array(segmentLength - 1).fill('-');
      retrievedLineData.push(lastQueryPrice);
      for (let t = 0; t < forecastLength; t++) {
        retrievedLineData.push(alignedForecast[t]);
      }

      const color = colors[segment.index % colors.length];

      series.push({
        name: `Pattern #${segment.index + 1}`,
        type: 'line',
        data: retrievedLineData,
        smooth: true,
        showSymbol: false,
        lineStyle: {
          width: 1.5,
          color: color,
          opacity: 0.45
        },
        itemStyle: {
          color: color
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
            const histPrices = seg.historical_prices || [];
            const histLast = histPrices.length > 0 ? histPrices[histPrices.length - 1] : (prices[0] || 1);
            const meanSeg = prices.reduce((a, b) => a + b, 0) / prices.length;
            const stdSeg = Math.sqrt(prices.reduce((a, b) => a + Math.pow(b - meanSeg, 2), 0) / prices.length) || 1e-8;
            
            const alignedPrice = lastQueryPrice + ((prices[t] - histLast) / stdSeg) * scaleMultiplier;
            sum += alignedPrice;
          });
          consensusPrices.push(sum / activeSegments.length);
        }

        const consensusCandles = Array(segmentLength).fill('-');
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

  }, [queryPrices, queryCandles, retrievedData, activeToggles, segmentLength, frequency, liveActualPrices]);

  // Clean up chart and auto-archive on unmount
  useEffect(() => {
    return () => {
      // Auto-archive active session on unmount
      const currentPrices = queryPricesRef.current;
      const actualPrices = liveActualPricesRef.current;
      const currentStatus = archiveStatusRef.current;
      
      if (currentPrices.length > 0 && actualPrices.length > 0 && currentStatus === 'idle') {
        const validPrices = actualPrices.filter(p => p !== null && p !== undefined);
        if (validPrices.length >= 5) {
          console.log(`[RetrievalVisualizer] Auto-archiving active tracking run on unmount...`);
          const payloadStr = JSON.stringify({
            symbol: tokenRef.current,
            timeframe: frequencyRef.current,
            prices: currentPrices,
            actual_future_prices: validPrices,
            leverage: 1.0
          });
          if (navigator.sendBeacon) {
            const blob = new Blob([payloadStr], { type: 'application/json' });
            navigator.sendBeacon('/api/retrieval/setup/add', blob);
          } else {
            fetch('/api/retrieval/setup/add', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: payloadStr
            }).catch(err => console.error(err));
          }
        }
      }

      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
    };
  }, []);

  // Auto-archive when the forecast window is fully completed
  useEffect(() => {
    if (liveActualPrices.length === 0 || archiveStatus !== 'idle') return;

    const isComplete = liveActualPrices.every(p => p !== null && p !== undefined);
    if (isComplete) {
      const saveComplete = async () => {
        setArchiveStatus('saving');
        try {
          const response = await fetch('/api/retrieval/setup/add', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              symbol: token,
              timeframe: frequency,
              prices: queryPrices,
              actual_future_prices: liveActualPrices,
              leverage: 1.0
            })
          });
          if (response.ok) {
            setArchiveStatus('saved');
            console.log('[RetrievalVisualizer] Full tracking run archived to database.');
          } else {
            setArchiveStatus('error');
          }
        } catch (error) {
          setArchiveStatus('error');
          console.error('[RetrievalVisualizer] Error saving full setup:', error);
        }
      };
      saveComplete();
    }
  }, [liveActualPrices, queryPrices, archiveStatus, token, frequency]);

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
    const scaleMultiplier = getScaleMultiplier(queryPrices, lastQueryPrice);
    
    activeSegments.forEach(seg => {
      const prices = seg.prices || [];
      if (prices.length === 0) return;
      
      const histPrices = seg.historical_prices || [];
      const histLast = histPrices.length > 0 ? histPrices[histPrices.length - 1] : (prices[0] || 1);
      const meanSeg = prices.reduce((a, b) => a + b, 0) / prices.length;
      const stdSeg = Math.sqrt(prices.reduce((a, b) => a + Math.pow(b - meanSeg, 2), 0) / prices.length) || 1e-8;
      
      const firstAlignedPrice = lastQueryPrice + ((prices[0] - histLast) / stdSeg) * scaleMultiplier;
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

  const getTrackingStats = () => {
    const lastQueryPrice = queryPrices[queryPrices.length - 1] || 0;
    if (lastQueryPrice === 0 || liveActualPrices.length === 0) return null;

    let lastValidIdx = -1;
    for (let i = liveActualPrices.length - 1; i >= 0; i--) {
      if (liveActualPrices[i] !== null && liveActualPrices[i] !== undefined) {
        lastValidIdx = i;
        break;
      }
    }

    if (lastValidIdx === -1) return null;

    const actualPrice = liveActualPrices[lastValidIdx];
    const actualReturn = ((actualPrice - lastQueryPrice) / lastQueryPrice) * 100;

    const activeSegments = retrievedData.filter(seg => activeToggles[seg.id || seg.index]);
    if (activeSegments.length === 0) return null;

    const scaleMultiplier = getScaleMultiplier(queryPrices, lastQueryPrice);
    
    let sum = 0;
    activeSegments.forEach(seg => {
      const prices = seg.prices || [];
      const histPrices = seg.historical_prices || [];
      const histLast = histPrices.length > 0 ? histPrices[histPrices.length - 1] : (prices[0] || 1);
      const meanSeg = prices.reduce((a, b) => a + b, 0) / prices.length;
      const stdSeg = Math.sqrt(prices.reduce((a, b) => a + Math.pow(b - meanSeg, 2), 0) / prices.length) || 1e-8;
      
      const alignedPrice = lastQueryPrice + ((prices[lastValidIdx] - histLast) / stdSeg) * scaleMultiplier;
      sum += alignedPrice;
    });
    const consensusPrice = sum / activeSegments.length;
    const predictedReturn = ((consensusPrice - lastQueryPrice) / lastQueryPrice) * 100;

    const errorPct = Math.abs((actualPrice - consensusPrice) / consensusPrice) * 100;
    const directionMatches = (actualReturn >= 0 && predictedReturn >= 0) || (actualReturn < 0 && predictedReturn < 0);

    return {
      elapsedSteps: lastValidIdx + 1,
      totalSteps: liveActualPrices.length,
      actualPrice,
      actualReturn,
      predictedReturn,
      errorPct,
      directionMatches
    };
  };

  const trackingStats = getTrackingStats();

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

          {/* Live Performance Tracking Card */}
          {trackingStats && (
            <div className="flex flex-col sm:flex-row gap-4 p-4 border border-indigo-900/40 rounded-xl bg-gradient-to-r from-indigo-950/20 to-slate-950/60 items-center justify-between shadow-md">
              <div className="flex items-center gap-4">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold border shadow-sm ${
                  trackingStats.directionMatches 
                    ? 'bg-emerald-950/40 text-emerald-400 border-emerald-500/20 shadow-emerald-950/20' 
                    : 'bg-rose-950/40 text-rose-400 border-rose-500/20 shadow-rose-950/20'
                }`}>
                  {trackingStats.directionMatches ? '✓' : '✗'}
                </div>
                <div>
                  <span className="text-[10px] text-indigo-400 font-bold uppercase tracking-wider font-mono flex items-center gap-1.5 flex-wrap">
                    <span>LIVE FORECAST TRACKING ({trackingStats.elapsedSteps} / {trackingStats.totalSteps} steps)</span>
                    {archiveStatus === 'saving' && <span className="text-amber-500 animate-pulse font-extrabold">(Archiving to DB...)</span>}
                    {archiveStatus === 'saved' && <span className="text-emerald-400 font-extrabold">(Archived to DB ✅)</span>}
                    {archiveStatus === 'error' && <span className="text-rose-500 font-extrabold">(DB Sync Failed ❌)</span>}
                  </span>
                  <h3 className={`text-base font-extrabold tracking-tight ${
                    trackingStats.directionMatches ? 'text-emerald-400' : 'text-rose-400'
                  }`}>
                    {trackingStats.directionMatches ? 'CONFIRMING FORECAST (UP/DOWN MATCH)' : 'DIVERGING FROM FORECAST'}
                  </h3>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-x-6 gap-y-1 w-full sm:w-auto font-mono text-xs border-t sm:border-t-0 sm:border-l border-slate-800/80 pt-2 sm:pt-0 sm:pl-6">
                <div className="flex justify-between gap-4">
                  <span className="text-slate-500">Actual Price:</span>
                  <span className="text-slate-350 font-semibold">${trackingStats.actualPrice.toFixed(2)}</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-slate-500">Actual Return:</span>
                  <span className={`font-semibold ${trackingStats.actualReturn >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {trackingStats.actualReturn >= 0 ? '+' : ''}{trackingStats.actualReturn.toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-slate-500">Consensus Pred:</span>
                  <span className={`font-semibold ${trackingStats.predictedReturn >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {trackingStats.predictedReturn >= 0 ? '+' : ''}{trackingStats.predictedReturn.toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-slate-500">Mean Abs Error:</span>
                  <span className="text-slate-350 font-semibold">{trackingStats.errorPct.toFixed(2)}%</span>
                </div>
              </div>
            </div>
          )}
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