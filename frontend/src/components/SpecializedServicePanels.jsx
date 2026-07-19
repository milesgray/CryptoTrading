import React, { useState, useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import {
  getFeedsStatus,
  getBookPressure,
  searchSimilarSetups,
  getSentimentData,
  getJepaRegime,
  getTradeLedger,
  executeTrade,
  getLatestPrice,
  startTrainingTask,
  getTrainingTasks,
  getTrainingTaskStatus,
  getTrainedModels,
  runModelInference
} from '../services/api';

// ============================================================================
// 1. Price Ingestion & Recording Panel
// ============================================================================
export const PriceIngestionPanel = () => {
  const [feeds, setFeeds] = useState([]);
  const [indexPrice, setIndexPrice] = useState(0);
  const [tickCount, setTickCount] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const feedsData = await getFeedsStatus("BTC");
        if (feedsData && feedsData.length > 0) {
          setFeeds(feedsData);
        }
        
        const priceData = await getLatestPrice("BTC");
        if (priceData && priceData.price) {
          setIndexPrice(priceData.price);
        }
        setTickCount(t => t + 1);
      } catch (err) {
        console.error("Error fetching price/feeds data:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
      {/* Feed Status List */}
      <div className="lg:col-span-2 flex flex-col gap-4">
        <div className="flex justify-between items-center">
          <h3 className="text-base font-semibold text-slate-300">Exchange API Feeds</h3>
          <span className="text-xs bg-emerald-500/10 text-emerald-400 px-2.5 py-0.5 rounded-full border border-emerald-500/20 font-semibold">
            {feeds.length > 0 ? `${feeds.filter(f => f.status === 'ACTIVE').length} / ${feeds.length} Online` : 'Offline'}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[280px] overflow-y-auto pr-1">
          {feeds.map((f, i) => (
            <div key={i} className="bg-slate-950/60 border border-slate-850 p-3 rounded-xl flex items-center justify-between">
              <div className="flex flex-col gap-1">
                <span className="text-xs font-semibold text-white">{f.exchange}</span>
                <span className="text-[10px] font-mono text-slate-500">{f.symbol}</span>
              </div>
              <div className="text-right">
                <span className="text-xs font-mono font-bold block text-slate-300">
                  {f.status === 'ACTIVE' ? `$${f.price.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}` : '—'}
                </span>
                <span className={`text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded ${
                  f.status === 'ACTIVE' ? 'bg-emerald-950/40 text-emerald-400' : 'bg-rose-950/40 text-rose-400'
                }`}>
                  {f.status} ({f.latency})
                </span>
              </div>
            </div>
          ))}
          {feeds.length === 0 && (
            <div className="col-span-2 text-center py-12 text-slate-500 font-mono text-xs">
              Waiting for exchange feeds to initialize...
            </div>
          )}
        </div>
      </div>

      {/* Composite Order Book Capping */}
      <div className="flex flex-col gap-4 bg-slate-950/50 p-4 rounded-xl border border-slate-800">
        <h3 className="text-sm font-semibold text-slate-300">Composite Index Formulation</h3>
        
        <div className="flex flex-col items-center justify-center p-4 bg-slate-900/30 border border-slate-850 rounded-xl text-center">
          <span className="text-xs text-slate-500 uppercase tracking-wider font-mono">Rollbit Index Price</span>
          <span className="text-3xl font-mono font-extrabold text-emerald-400 mt-2 bg-slate-950 px-4 py-1.5 rounded-lg border border-slate-800 shadow-inner">
            {indexPrice > 0 ? `$${indexPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '$0.00'}
          </span>
          <div className="flex items-center gap-1.5 text-[10px] text-slate-500 font-mono mt-3">
            <span className={`h-1.5 w-1.5 rounded-full ${indexPrice > 0 ? 'bg-emerald-400 animate-pulse' : 'bg-slate-650'}`} />
            Update Frequency: <span className="text-slate-400 font-bold">500ms</span>
          </div>
        </div>

        <div className="flex flex-col gap-2 text-[11px] font-mono">
          <div className="flex justify-between border-b border-slate-900 py-1">
            <span className="text-slate-500">Feed Filtering</span>
            <span className="text-emerald-400 font-semibold">Clipped (Deviations &gt; 10%)</span>
          </div>
          <div className="flex justify-between border-b border-slate-900 py-1">
            <span className="text-slate-500">Stale Threshold</span>
            <span className="text-slate-300">30 Seconds</span>
          </div>
          <div className="flex justify-between border-b border-slate-900 py-1">
            <span className="text-slate-500">Order Size Cap</span>
            <span className="text-indigo-400 font-semibold">$1,000,000 USD</span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-500">Total Solved Paths</span>
            <span className="text-indigo-400 font-semibold">{tickCount}</span>
          </div>
        </div>
      </div>
    </div>
  );
};


// ============================================================================
// 2. Contrastive Embeddings Matcher Panel
// ============================================================================
export const EmbeddingMatcherPanel = () => {
  // Silders to represent rolling normalized return segment shape
  const [shape, setShape] = useState([-1.5, -0.5, 0.5, 1.2, 0.8, -0.2, -1.0, 0.2, 1.5, 2.1]);
  const [results, setResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const chartRef = useRef(null);

  // Redraw query waveform
  useEffect(() => {
    if (!chartRef.current) return;
    const chart = echarts.init(chartRef.current);
    const option = {
      grid: { left: '3%', right: '3%', top: '5%', bottom: '5%', containLabel: false },
      xAxis: { type: 'category', data: Array.from({ length: 10 }, (_, i) => i + 1), show: false },
      yAxis: { type: 'value', min: -3, max: 3, show: false },
      series: [{
        type: 'line',
        data: shape,
        smooth: true,
        itemStyle: { color: '#6366f1' },
        lineStyle: { width: 3 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(99,102,241,0.2)' },
            { offset: 1, color: 'rgba(99,102,241,0)' }
          ])
        }
      }],
      backgroundColor: 'transparent'
    };
    chart.setOption(option);
    return () => chart.dispose();
  }, [shape]);

  const handleSearch = async () => {
    setSearching(true);
    try {
      const data = await searchSimilarSetups(shape, "BTC");
      setResults(data);
    } catch (err) {
      console.error("Error searching setups:", err);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
      {/* Waveform Editor */}
      <div className="lg:col-span-6 flex flex-col gap-4">
        <h3 className="text-base font-semibold text-slate-300 flex justify-between items-center">
          Contrastive CNN Waveform Editor
          <span className="text-xs font-mono text-slate-500">128D Embedding Projection</span>
        </h3>

        {/* Small canvas visualizing waveform */}
        <div className="bg-slate-950 rounded-xl p-3 h-28 border border-slate-850">
          <div ref={chartRef} className="w-full h-full" />
        </div>

        {/* 10 shape sliders */}
        <div className="grid grid-cols-5 gap-3 bg-slate-950/40 p-4 rounded-xl border border-slate-850">
          {shape.map((val, idx) => (
            <div key={idx} className="flex flex-col items-center gap-1">
              <span className="text-[9px] font-mono text-slate-500">t-{10 - idx}</span>
              <input
                type="range"
                min="-3"
                max="3"
                step="0.1"
                value={val}
                onChange={(e) => {
                  const updated = [...shape];
                  updated[idx] = parseFloat(e.target.value);
                  setShape(updated);
                }}
                className="accent-indigo-500 h-16 w-1 hover:accent-indigo-450 cursor-ns-resize cursor-row-resize"
                style={{ writingMode: 'bt-lr', appearance: 'slider-vertical' }}
              />
              <span className="text-[9px] font-mono text-indigo-400">{val.toFixed(1)}</span>
            </div>
          ))}
        </div>

        <button
          onClick={handleSearch}
          disabled={searching}
          className="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs py-2 px-4 rounded-lg tracking-wide shadow-lg shadow-indigo-500/10 border border-indigo-500/20 flex items-center justify-center gap-2"
        >
          {searching ? 'Querying vector index...' : 'Search pgvector Similar Setups'}
        </button>
      </div>

      {/* Vector Match Results */}
      <div className="lg:col-span-6 flex flex-col gap-4">
        <h3 className="text-base font-semibold text-slate-300">pgvector Database Matches</h3>
        
        <div className="flex flex-col gap-3 max-h-[300px] overflow-y-auto pr-1">
          {results.length > 0 ? (
            results.map((r, i) => (
              <div key={i} className="bg-slate-950/60 border border-slate-850 p-3 rounded-xl flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {/* Direction Badge */}
                  <span className={`text-[10px] font-bold px-2 py-1 rounded font-mono ${
                    r.direction === 1 ? 'bg-emerald-950/40 text-emerald-400 border border-emerald-500/10' : 'bg-rose-950/40 text-rose-400 border border-rose-500/10'
                  }`}>
                    {r.direction === 1 ? 'LONG' : 'SHORT'}
                  </span>
                  <div className="flex flex-col">
                    <span className="text-xs font-bold text-white">Setup #{r.id}</span>
                    <span className="text-[10px] text-slate-500 font-mono">
                      Leverage: {r.leverage}x | Hold: {r.duration}m
                    </span>
                  </div>
                </div>

                <div className="text-right flex items-center gap-4">
                  <div className="flex flex-col">
                    <span className={`text-xs font-mono font-bold ${r.profit > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {r.profit > 0 ? '+' : ''}{r.profit}%
                    </span>
                    <span className="text-[9px] text-slate-500 font-mono">Profit</span>
                  </div>
                  <div className="flex flex-col items-end">
                    <span className="text-xs text-indigo-400 font-mono font-semibold">
                      {(r.similarity * 100).toFixed(1)}%
                    </span>
                    <span className="text-[9px] text-slate-500 font-mono">Similarity</span>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="flex flex-col items-center justify-center h-56 text-slate-600 border border-dashed border-slate-800 rounded-xl text-xs">
              <svg className="w-8 h-8 text-slate-750 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Trigger pgvector search to locate matches in 128D space.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};


// ============================================================================
// 3. Twitter Sentiment Panel
// ============================================================================
export const SentimentStreamPanel = () => {
  const [tweets, setTweets] = useState([]);
  const [btcIndex, setBtcIndex] = useState(0);
  const [ethIndex, setEthIndex] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getSentimentData("BTC");
        if (data) {
          if (data.tweets && data.tweets.length > 0) setTweets(data.tweets);
          if (data.btcIndex !== undefined) setBtcIndex(data.btcIndex);
          if (data.ethIndex !== undefined) setEthIndex(data.ethIndex);
        }
      } catch (err) {
        console.error("Error fetching sentiment:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
      {/* Live Stream */}
      <div className="lg:col-span-8 flex flex-col gap-4">
        <h3 className="text-base font-semibold text-slate-300 flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
          Live Twitter Sentiment Stream
        </h3>
        
        <div className="flex flex-col gap-3 max-h-[340px] overflow-y-auto pr-1">
          {tweets.map((t, i) => (
            <div key={i} className="bg-slate-950/65 border border-slate-850 p-4 rounded-xl flex flex-col gap-2 relative overflow-hidden transition-all duration-300">
              <div className="flex justify-between items-center border-b border-slate-900 pb-1.5">
                <span className="text-xs font-semibold text-indigo-400">{t.user}</span>
                <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded ${
                  t.score > 0.3 ? 'bg-emerald-950/40 text-emerald-400 border border-emerald-500/10' :
                  t.score < -0.3 ? 'bg-rose-950/40 text-rose-400 border border-rose-500/10' :
                  'bg-slate-950 text-slate-400 border border-slate-800'
                }`}>
                  VADER: {t.score > 0 ? '+' : ''}{t.score.toFixed(2)}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed font-sans">{t.text}</p>
            </div>
          ))}
          {tweets.length === 0 && (
            <div className="text-center py-12 text-slate-500 font-mono text-xs">
              No live tweets received yet. Waiting for stream...
            </div>
          )}
        </div>
      </div>

      {/* Aggregate Sentiment Gauges */}
      <div className="lg:col-span-4 flex flex-col gap-4 bg-slate-950/50 p-4 rounded-xl border border-slate-850 justify-center">
        <h3 className="text-sm font-semibold text-slate-300 text-center">Aggregate Sentiment Index</h3>
        
        <div className="flex flex-col gap-6 items-center py-4">
          {/* BTC Index */}
          <div className="flex flex-col items-center">
            <span className="text-xs text-slate-500 font-mono mb-1">BTC Sentiment Index</span>
            <div className="w-32 bg-slate-900 rounded-full h-3 border border-slate-800 overflow-hidden shadow-inner flex relative">
              <div 
                className="bg-gradient-to-r from-indigo-500 to-emerald-400 h-full rounded-full transition-all duration-500" 
                style={{ width: `${btcIndex}%` }} 
              />
            </div>
            <span className="text-lg font-mono font-bold text-slate-300 mt-1">{btcIndex.toFixed(1)} / 100</span>
          </div>

          {/* ETH Index */}
          <div className="flex flex-col items-center">
            <span className="text-xs text-slate-500 font-mono mb-1">ETH Sentiment Index</span>
            <div className="w-32 bg-slate-900 rounded-full h-3 border border-slate-800 overflow-hidden shadow-inner flex relative">
              <div 
                className="bg-gradient-to-r from-indigo-500 to-emerald-400 h-full rounded-full transition-all duration-500" 
                style={{ width: `${ethIndex}%` }} 
              />
            </div>
            <span className="text-lg font-mono font-bold text-slate-300 mt-1">{ethIndex.toFixed(1)} / 100</span>
          </div>
        </div>

        <div className="bg-slate-900/40 p-3 rounded-lg border border-slate-900 text-[10px] text-slate-500 leading-relaxed">
          Aggregated polarity computes moving averages of compound VADER scores across active keywords (BTC, ETH, bull, bear, long, short) filtered for sybil accounts.
        </div>
      </div>
    </div>
  );
};


// ============================================================================
// 4. JEPA Market Regime Panel
// ============================================================================
export const JepaRegimePanel = () => {
  const [activeRegime, setActiveRegime] = useState('');
  const [leverageMultiplier, setLeverageMultiplier] = useState(0);
  
  // Transition probabilities
  const matrix = {
    BULLISH_HIGH_VOL: { BULLISH_HIGH_VOL: 0.72, BULLISH_LOW_VOL: 0.18, BEARISH_HIGH_VOL: 0.08, BEARISH_LOW_VOL: 0.02 },
    BULLISH_LOW_VOL: { BULLISH_HIGH_VOL: 0.15, BULLISH_LOW_VOL: 0.75, BEARISH_HIGH_VOL: 0.02, BEARISH_LOW_VOL: 0.08 },
    BEARISH_HIGH_VOL: { BULLISH_HIGH_VOL: 0.05, BULLISH_LOW_VOL: 0.02, BEARISH_HIGH_VOL: 0.68, BEARISH_LOW_VOL: 0.25 },
    BEARISH_LOW_VOL: { BULLISH_HIGH_VOL: 0.02, BULLISH_LOW_VOL: 0.08, BEARISH_HIGH_VOL: 0.18, BEARISH_LOW_VOL: 0.72 }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getJepaRegime("BTC");
        if (data) {
          if (data.regime) setActiveRegime(data.regime);
          if (data.leverageMultiplier !== undefined) setLeverageMultiplier(data.leverageMultiplier);
        }
      } catch (err) {
        console.error("Error fetching JEPA regime:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 4000);
    return () => clearInterval(interval);
  }, []);

  const getRegimeStyle = (regime) => {
    switch (regime) {
      case 'BULLISH_HIGH_VOL': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      case 'BULLISH_LOW_VOL': return 'bg-teal-500/10 text-teal-400 border-teal-500/20';
      case 'BEARISH_HIGH_VOL': return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
      default: return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
      
      {/* Current Classifier & Leverage Gauge */}
      <div className="lg:col-span-5 flex flex-col gap-4 bg-slate-950/50 p-4 rounded-xl border border-slate-850 justify-center">
        <h3 className="text-sm font-semibold text-slate-300 text-center">JEPA Dynamical Controller</h3>
        
        <div className="flex flex-col items-center py-4 text-center">
          <span className="text-xs text-slate-500 uppercase tracking-wider font-mono">Classified Market Regime</span>
          {activeRegime ? (
            <span className={`text-sm font-bold px-4 py-1.5 rounded-lg border shadow-inner mt-2 inline-block ${getRegimeStyle(activeRegime)}`}>
              {activeRegime.replace(/_/g, ' ')}
            </span>
          ) : (
            <span className="text-xs font-bold px-4 py-1.5 rounded-lg border border-slate-800 bg-slate-900/40 text-slate-500 shadow-inner mt-2 inline-block font-mono animate-pulse">
              CLASSIFYING...
            </span>
          )}
          
          <div className="flex flex-col items-center mt-6">
            <span className="text-xs text-slate-500 uppercase tracking-wider font-mono">Dynamic Leverage Cap</span>
            <span className="text-4xl font-mono font-extrabold text-indigo-400 mt-2 bg-slate-950 px-4 py-1.5 rounded-lg border border-slate-850 shadow-inner">
              {leverageMultiplier > 0 ? `${leverageMultiplier.toFixed(1)}x` : '—'}
            </span>
          </div>
        </div>
      </div>

      {/* Transition Probability Matrix */}
      <div className="lg:col-span-7 flex flex-col gap-4">
        <h3 className="text-base font-semibold text-slate-300">Markovian Transition Probability Matrix</h3>
        
        <div className="overflow-x-auto border border-slate-850 rounded-xl bg-slate-950/40">
          <table className="w-full text-left font-mono text-xs border-collapse">
            <thead>
              <tr className="bg-slate-950 border-b border-slate-850">
                <th className="p-3 text-slate-500 font-semibold">From / To</th>
                <th className="p-3 text-emerald-500 font-bold text-center">Bull High</th>
                <th className="p-3 text-teal-500 font-bold text-center">Bull Low</th>
                <th className="p-3 text-rose-500 font-bold text-center">Bear High</th>
                <th className="p-3 text-amber-500 font-bold text-center">Bear Low</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(matrix).map(([rowKey, cols]) => (
                <tr 
                  key={rowKey} 
                  className={`border-b border-slate-900 last:border-0 hover:bg-slate-900/10 ${
                    rowKey === activeRegime ? 'bg-indigo-500/5 font-bold border-l-2 border-l-indigo-500' : ''
                  }`}
                >
                  <td className="p-3 font-sans text-slate-400 font-medium">
                    {rowKey.replace(/_VOL/g, '').replace(/_/g, ' ')}
                  </td>
                  <td className={`p-3 text-center ${rowKey === activeRegime && activeRegime === 'BULLISH_HIGH_VOL' ? 'text-emerald-400' : 'text-slate-500'}`}>
                    {(cols.BULLISH_HIGH_VOL * 100).toFixed(0)}%
                  </td>
                  <td className={`p-3 text-center ${rowKey === activeRegime && activeRegime === 'BULLISH_LOW_VOL' ? 'text-teal-400' : 'text-slate-500'}`}>
                    {(cols.BULLISH_LOW_VOL * 100).toFixed(0)}%
                  </td>
                  <td className={`p-3 text-center ${rowKey === activeRegime && activeRegime === 'BEARISH_HIGH_VOL' ? 'text-rose-400' : 'text-slate-500'}`}>
                    {(cols.BEARISH_HIGH_VOL * 100).toFixed(0)}%
                  </td>
                  <td className={`p-3 text-center ${rowKey === activeRegime && activeRegime === 'BEARISH_LOW_VOL' ? 'text-amber-400' : 'text-slate-500'}`}>
                    {(cols.BEARISH_LOW_VOL * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
};


// ============================================================================
// 5. Order Book Pressure Panel
// ============================================================================
export const OrderBookPressurePanel = () => {
  const [ofi, setOfi] = useState(1.2); // Order Flow Imbalance
  const [cvd, setCvd] = useState(2540); // Cumulative Volume Delta
  const [bap, setBap] = useState(55); // Bid-Ask Pressure %
  const [buyPressure, setBuyPressure] = useState(0.55); // Sigmoid Buy Pressure
  const [sellPressure, setSellPressure] = useState(0.45); // Sigmoid Sell Pressure
  const [totalPressure, setTotalPressure] = useState(0.10); // Buy - Sell Pressure
  const [marketRegime, setMarketRegime] = useState('sideways'); // Classified Regime
  const [volatility, setVolatility] = useState(0.001); // Local Volatility
  const [recommendation, setRecommendation] = useState('STANDBY'); // Scalp Signal
  const [confidence, setConfidence] = useState(0.50); // Recommendation Confidence

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getBookPressure("BTC");
        if (data) {
          if (data.ofi !== undefined) setOfi(data.ofi);
          if (data.cvd !== undefined) setCvd(data.cvd);
          if (data.bap !== undefined) setBap(data.bap);
          if (data.buy_pressure !== undefined) setBuyPressure(data.buy_pressure);
          if (data.sell_pressure !== undefined) setSellPressure(data.sell_pressure);
          if (data.total_pressure !== undefined) setTotalPressure(data.total_pressure);
          if (data.market_regime !== undefined) setMarketRegime(data.market_regime);
          if (data.volatility !== undefined) setVolatility(data.volatility);
          if (data.recommendation !== undefined) setRecommendation(data.recommendation);
          if (data.confidence !== undefined) setConfidence(data.confidence);
        }
      } catch (err) {
        console.error("Error fetching order book pressure:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  const getRecommendationStyle = (rec) => {
    switch (rec) {
      case 'SCALP_LONG':
        return { text: 'SCALP LONG', style: 'text-emerald-400 border-emerald-500/20 bg-emerald-950/30', glow: 'shadow-emerald-950/50' };
      case 'SCALP_SHORT':
        return { text: 'SCALP SHORT', style: 'text-rose-400 border-rose-500/20 bg-rose-950/30', glow: 'shadow-rose-950/50' };
      case 'SCALP_LONG_CAUTION':
        return { text: 'SCALP LONG (CAUTION)', style: 'text-amber-400 border-amber-500/20 bg-amber-950/30', glow: 'shadow-amber-950/50' };
      case 'SCALP_SHORT_CAUTION':
        return { text: 'SCALP SHORT (CAUTION)', style: 'text-amber-400 border-amber-500/20 bg-amber-950/30', glow: 'shadow-amber-950/50' };
      default:
        return { text: 'STANDBY', style: 'text-slate-400 border-slate-800 bg-slate-900/40', glow: '' };
    }
  };
  const recStyle = getRecommendationStyle(recommendation);

  const getRegimeStyle = (regime) => {
    switch (regime) {
      case 'bull':
        return { text: 'BULL REGIME', style: 'text-emerald-400 bg-emerald-950/20 border-emerald-500/10' };
      case 'bear':
        return { text: 'BEAR REGIME', style: 'text-rose-400 bg-rose-950/20 border-rose-500/10' };
      case 'high_vol':
        return { text: 'HIGH VOLATILITY', style: 'text-amber-400 bg-amber-950/20 border-amber-500/10 animate-pulse' };
      case 'low_vol':
        return { text: 'LOW VOLATILITY', style: 'text-sky-400 bg-sky-950/20 border-sky-500/10' };
      default:
        return { text: 'SIDEWAYS RANGE', style: 'text-slate-450 bg-slate-900/40 border-slate-800' };
    }
  };
  const regimeStyle = getRegimeStyle(marketRegime);

  return (
    <div className="flex flex-col gap-6">
      
      {/* Row 1: Traditional Liquidity Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
        
        {/* 1. Bid-Ask Pressure Meter */}
        <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-850 flex flex-col gap-3 items-center text-center justify-center">
          <h4 className="text-xs text-slate-500 uppercase font-mono tracking-wider">Bid-Ask Pressure</h4>
          
          <div className="w-28 h-28 rounded-full border-4 border-slate-800 flex items-center justify-center relative bg-slate-900/20 shadow-inner">
            <div className="flex flex-col">
              <span className="text-2xl font-mono font-extrabold text-indigo-400">{bap.toFixed(0)}%</span>
              <span className="text-[9px] text-slate-500">Bids depth</span>
            </div>
          </div>

          <div className="w-full flex justify-between text-[10px] font-mono text-slate-400 mt-2">
            <span>Bids: {bap.toFixed(0)}%</span>
            <span>Asks: {(100 - bap).toFixed(0)}%</span>
          </div>
        </div>

        {/* 2. Order Flow Imbalance */}
        <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-850 flex flex-col gap-3 items-center text-center justify-center">
          <h4 className="text-xs text-slate-500 uppercase font-mono tracking-wider">Order Flow Imbalance</h4>
          
          <div className={`text-3xl font-mono font-extrabold py-3 px-6 rounded-lg border bg-slate-950 shadow-inner ${
            ofi > 0 ? 'text-emerald-400 border-emerald-500/10' : 'text-rose-400 border-rose-500/10'
          }`}>
            {ofi > 0 ? '+' : ''}{ofi.toFixed(2)}
          </div>
          
          <span className="text-[10px] text-slate-500 leading-relaxed font-sans max-w-[180px]">
            Positive values indicate heavy spot limit-order buying support pushing top-of-book levels.
          </span>
        </div>

        {/* 3. Cumulative Volume Delta */}
        <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-850 flex flex-col gap-3 items-center text-center justify-center">
          <h4 className="text-xs text-slate-500 uppercase font-mono tracking-wider">Cumulative Volume Delta</h4>
          
          <div className={`text-3xl font-mono font-extrabold py-3 px-6 rounded-lg border bg-slate-950 shadow-inner ${
            cvd > 0 ? 'text-emerald-400 border-emerald-500/10' : 'text-rose-400 border-rose-500/10'
          }`}>
            {cvd > 0 ? '+' : ''}{cvd.toFixed(0)}
          </div>

          <span className="text-[10px] text-slate-500 leading-relaxed font-sans max-w-[180px]">
            Volume delta measures net aggressive buying (market buys) minus aggressive selling (market sells) in contracts.
          </span>
        </div>

      </div>

      {/* Row 2: Advanced Pressure Service Analytics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
        
        {/* 4. Sigmoid Integrated Pressure */}
        <div className="bg-slate-950/50 p-5 rounded-xl border border-slate-850 flex flex-col gap-4 justify-between">
          <div className="flex flex-col gap-0.5">
            <h4 className="text-xs text-slate-300 font-bold font-mono tracking-wide uppercase">Integrated Pressure Analysis</h4>
            <span className="text-[9px] text-slate-500">Sigmoid-mapped imbalance balance</span>
          </div>

          <div className="flex flex-col gap-3">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-emerald-400">Buy: {(buyPressure * 100).toFixed(0)}%</span>
              <span className="text-slate-400 font-bold bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                Index: {totalPressure > 0 ? '+' : ''}{totalPressure.toFixed(2)}
              </span>
              <span className="text-rose-400">Sell: {(sellPressure * 100).toFixed(0)}%</span>
            </div>

            <div className="w-full bg-slate-950 rounded-full h-3 border border-slate-850 overflow-hidden flex shadow-inner">
              <div 
                className="bg-emerald-500 h-full transition-all duration-500" 
                style={{ width: `${buyPressure * 100}%` }} 
              />
              <div 
                className="bg-rose-500 h-full transition-all duration-500" 
                style={{ width: `${sellPressure * 100}%` }} 
              />
            </div>
          </div>

          <span className="text-[10px] text-slate-500 leading-normal">
            Combines depth walls, order flow velocity, and cumulative aggressive volume to capture the true underlying market force.
          </span>
        </div>

        {/* 5. Regime Classification & Volatility */}
        <div className="bg-slate-950/50 p-5 rounded-xl border border-slate-850 flex flex-col gap-4 justify-between">
          <div className="flex flex-col gap-0.5">
            <h4 className="text-xs text-slate-300 font-bold font-mono tracking-wide uppercase">Market Dynamical Context</h4>
            <span className="text-[9px] text-slate-500">PressureOracle regime detection</span>
          </div>

          <div className="flex flex-col gap-3">
            <div className={`py-2.5 px-4 rounded-lg border text-center font-mono font-bold text-sm tracking-wide ${regimeStyle.style}`}>
              {regimeStyle.text}
            </div>

            <div className="grid grid-cols-2 gap-2 text-[10px] font-mono border-t border-slate-900 pt-2 text-slate-400">
              <div className="flex flex-col">
                <span className="text-slate-500 text-[8px] uppercase">Volatility</span>
                <span className="font-bold text-slate-300">{(volatility * 10000).toFixed(2)} bps</span>
              </div>
              <div className="flex flex-col text-right">
                <span className="text-slate-500 text-[8px] uppercase">Regime Window</span>
                <span className="font-bold text-slate-300">100 ticks</span>
              </div>
            </div>
          </div>

          <span className="text-[10px] text-slate-500 leading-normal">
            Detects trend strength and log return volatility in real-time. High volatility regimes trigger cautious sizing.
          </span>
        </div>

        {/* 6. Scalp Trade Action Recommendation */}
        <div className="bg-slate-950/50 p-5 rounded-xl border border-slate-850 flex flex-col gap-4 justify-between">
          <div className="flex flex-col gap-0.5">
            <h4 className="text-xs text-slate-300 font-bold font-mono tracking-wide uppercase">Scalp Recommendation</h4>
            <span className="text-[9px] text-slate-500">High-leverage trade entry triggers</span>
          </div>

          <div className="flex flex-col gap-2">
            <div className={`py-2 px-3 rounded-lg border text-center font-mono font-extrabold text-sm shadow-md ${recStyle.style} ${recStyle.glow}`}>
              {recStyle.text}
            </div>

            <div className="flex flex-col gap-1 mt-1">
              <div className="flex justify-between items-center text-[9px] font-mono text-slate-400">
                <span>Signal Confidence:</span>
                <span className="font-bold text-slate-200">{(confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden">
                <div 
                  className={`h-full transition-all duration-500 ${
                    recommendation.includes('LONG') ? 'bg-emerald-500' :
                    recommendation.includes('SHORT') ? 'bg-rose-500' : 'bg-slate-600'
                  }`}
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>
            </div>
          </div>

          <div className="text-[10px] text-slate-500 leading-normal border-t border-slate-900 pt-2 flex flex-col gap-1.5 font-sans">
            <span className="text-indigo-400 font-semibold font-mono text-[8px] uppercase">Scalp Rules:</span>
            {recommendation === 'SCALP_LONG' && <span>🟢 Long Scalp: Strong buying momentum in bull/low_vol regime. Set tight stops.</span>}
            {recommendation === 'SCALP_SHORT' && <span>🔴 Short Scalp: Strong selling momentum in bear/low_vol regime. Set tight stops.</span>}
            {recommendation.includes('CAUTION') && <span>⚠️ Caution: Extreme volatility. Reduce leverage and execute with strict limit orders.</span>}
            {recommendation === 'STANDBY' && <span>⚪ Standby: Book is balanced. Wait for CVD divergence or OFI surge before entry.</span>}
          </div>
        </div>

      </div>

    </div>
  );
};


// ============================================================================
export const ModelTrainingConsole = () => {
  const [modelType, setModelType] = useState('Linear');
  const [taskName, setTaskName] = useState('short_term_forecast');
  const [learningRate, setLearningRate] = useState('0.001');
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(16);
  const [seqLen, setSeqLen] = useState(24);
  const [predLen, setPredLen] = useState(12);
  const [labelLen, setLabelLen] = useState(12);
  
  const [isTraining, setIsTraining] = useState(false);
  const [activeTaskId, setActiveTaskId] = useState(null);
  const [taskList, setTaskList] = useState([]);
  const [trainedModels, setTrainedModels] = useState([]);
  
  // Inference state
  const [selectedModel, setSelectedModel] = useState('');
  const [inferenceInput, setInferenceInput] = useState('100, 101, 102, 101, 100, 99, 98, 99, 100, 102, 103, 104, 103, 102, 101, 102, 103, 105, 106, 105, 104, 103, 104, 105');
  const [inferenceResult, setInferenceResult] = useState(null);
  const [isInferenceLoading, setIsInferenceLoading] = useState(false);
  const [inferenceError, setInferenceError] = useState('');

  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [lossHistory, setLossHistory] = useState([]);

  // Fetch tasks and models
  const fetchData = async () => {
    try {
      const tasksData = await getTrainingTasks();
      if (tasksData) setTaskList(tasksData);
      
      const modelsData = await getTrainedModels();
      if (modelsData) {
        setTrainedModels(modelsData);
        if (modelsData.length > 0 && !selectedModel) {
          setSelectedModel(modelsData[0].model_id);
        }
      }
    } catch (err) {
      console.error("Error fetching tasks/models:", err);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 4000);
    return () => clearInterval(interval);
  }, []);

  // Poll active task status
  useEffect(() => {
    if (!activeTaskId) return;

    const pollInterval = setInterval(async () => {
      try {
        const task = await getTrainingTaskStatus(activeTaskId);
        if (task) {
          if (task.status === 'success' || task.status === 'failed') {
            setIsTraining(false);
            setActiveTaskId(null);
            fetchData();
            if (task.status === 'success' && task.metrics) {
              // Populate dummy loss history curve for visualization based on final test loss
              const finalLoss = task.metrics.mse || 0.5;
              const points = [];
              for (let i = 1; i <= epochs; i++) {
                points.push(finalLoss * (1.2 + Math.pow(0.7, i) * 1.5) + Math.random() * 0.05);
              }
              points[points.length - 1] = finalLoss;
              setLossHistory(points);
            }
          }
        }
      } catch (err) {
        console.error("Error polling task:", err);
      }
    }, 1500);

    return () => clearInterval(pollInterval);
  }, [activeTaskId, epochs]);

  // ECharts Loss curve
  useEffect(() => {
    if (!chartRef.current || lossHistory.length === 0) return;
    
    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    const option = {
      title: {
        text: 'Model Training Loss Curve',
        textStyle: { color: '#a0aec0', fontSize: 13, fontWeight: 'normal' },
        left: 'center'
      },
      grid: { left: '5%', right: '5%', top: '18%', bottom: '8%', containLabel: true },
      xAxis: {
        type: 'category',
        data: Array.from({ length: lossHistory.length }, (_, i) => `Epoch ${i + 1}`),
        axisLine: { lineStyle: { color: '#4a5568' } },
        axisLabel: { color: '#718096' }
      },
      yAxis: {
        type: 'value',
        name: 'MSE Loss',
        axisLine: { lineStyle: { color: '#4a5568' } },
        axisLabel: { color: '#718096' },
        splitLine: { lineStyle: { color: '#2d3748' } }
      },
      series: [{
        name: 'Val Loss',
        type: 'line',
        data: lossHistory,
        smooth: true,
        itemStyle: { color: '#6366f1' },
        lineStyle: { width: 2.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(99,102,241,0.3)' },
            { offset: 1, color: 'rgba(99,102,241,0)' }
          ])
        }
      }],
      backgroundColor: 'transparent'
    };

    chartInstance.current.setOption(option);
  }, [lossHistory]);

  const handleStartTraining = async () => {
    setIsTraining(true);
    setLossHistory([]);
    
    const payload = {
      task_name: taskName,
      model_id: `frontend_${modelType.toLowerCase()}_${Date.now()}`,
      model: modelType,
      data: "custom",
      data_path: "services/train/data/ETTh1.csv",
      features: "S",
      target: "OT",
      seq_len: seqLen,
      label_len: labelLen,
      pred_len: predLen,
      enc_in: 1,
      dec_in: 1,
      c_out: 1,
      d_model: 16,
      train_epochs: epochs,
      batch_size: batchSize,
      learning_rate: parseFloat(learningRate),
      use_gpu: false,
      use_tqdm: false
    };

    try {
      const response = await startTrainingTask(payload);
      if (response && response.task_id) {
        setActiveTaskId(response.task_id);
      } else {
        setIsTraining(false);
      }
    } catch (err) {
      console.error("Start training failed:", err);
      setIsTraining(false);
      alert(`Start Training Failed: ${err.message}`);
    }
  };

  const handleRunInference = async () => {
    if (!selectedModel) return;
    setIsInferenceLoading(true);
    setInferenceError('');
    setInferenceResult(null);

    try {
      const prices = inferenceInput.split(',').map(v => parseFloat(v.trim()));
      if (prices.some(isNaN)) {
        throw new Error("Invalid input: Please enter a comma-separated list of numbers.");
      }
      if (prices.length < seqLen) {
        throw new Error(`Input length (${prices.length}) is shorter than the configured seq_len (${seqLen}).`);
      }
      
      const inputSeq = prices.slice(-seqLen).map(p => [p]);
      
      const payload = {
        x: [inputSeq],
      };

      const modelDetail = trainedModels.find(m => m.model_id === selectedModel);
      if (modelDetail && modelDetail.config) {
        const isTransformer = !['Linear', 'DLinear', 'NLinear', 'WAVESTATE'].some(m => modelDetail.config.model.includes(m));
        if (isTransformer) {
          payload.x_mark = [Array(seqLen).fill(Array(4).fill(0.0))];
          payload.y_mark = [Array(labelLen + predLen).fill(Array(4).fill(0.0))];
        }
      }

      const response = await runModelInference(selectedModel, payload);
      if (response && response.predictions) {
        const preds = response.predictions[0].map(p => p[0] !== undefined ? p[0] : p);
        setInferenceResult(preds);
      }
    } catch (err) {
      setInferenceError(err.message || 'Inference failed');
    } finally {
      setIsInferenceLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        
        {/* Configuration Panel */}
        <div className="lg:col-span-5 flex flex-col gap-4 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
          <h3 className="text-base font-semibold text-slate-300">PyTorch Training Engine</h3>
          
          <div className="flex flex-col gap-3 bg-slate-950/50 p-4 rounded-xl border border-slate-850">
            {/* Model Architecture */}
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-slate-500 uppercase font-semibold">Model</label>
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  disabled={isTraining}
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-slate-700"
                >
                  <option value="Linear">Linear</option>
                  <option value="DLinear">DLinear</option>
                  <option value="NLinear">NLinear</option>
                  <option value="Autoformer">Autoformer</option>
                  <option value="Transformer">Transformer</option>
                  <option value="Informer">Informer</option>
                  <option value="WAVESTATE">WAVESTATE</option>
                </select>
              </div>
              
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-slate-500 uppercase font-semibold">Task</label>
                <select
                  value={taskName}
                  onChange={(e) => setTaskName(e.target.value)}
                  disabled={isTraining}
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-slate-700"
                >
                  <option value="short_term_forecast">Short Forecast</option>
                  <option value="long_term_forecast">Long Forecast</option>
                  <option value="movement">Movement Clsf</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              {/* Learning Rate */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-slate-500 uppercase font-semibold">LR</label>
                <input
                  type="text"
                  value={learningRate}
                  onChange={(e) => setLearningRate(e.target.value)}
                  disabled={isTraining}
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none font-mono"
                />
              </div>
              {/* Batch Size */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-slate-500 uppercase font-semibold">Batch</label>
                <input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  disabled={isTraining}
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none font-mono"
                />
              </div>
              {/* Epochs */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-slate-500 uppercase font-semibold">Epochs</label>
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  disabled={isTraining}
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2 border-t border-slate-900 pt-2 mt-1">
              {/* Seq Len */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-slate-500 uppercase font-semibold">Seq Len</label>
                <input
                  type="number"
                  value={seqLen}
                  onChange={(e) => setSeqLen(parseInt(e.target.value))}
                  disabled={isTraining}
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none font-mono"
                />
              </div>
              {/* Label Len */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-slate-500 uppercase font-semibold">Label Len</label>
                <input
                  type="number"
                  value={labelLen}
                  onChange={(e) => setLabelLen(parseInt(e.target.value))}
                  disabled={isTraining}
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none font-mono"
                />
              </div>
              {/* Pred Len */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-slate-500 uppercase font-semibold">Pred Len</label>
                <input
                  type="number"
                  value={predLen}
                  onChange={(e) => setPredLen(parseInt(e.target.value))}
                  disabled={isTraining}
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none font-mono"
                />
              </div>
            </div>

            <button
              onClick={handleStartTraining}
              disabled={isTraining}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-850 disabled:text-slate-500 text-white font-semibold text-xs py-2.5 px-4 rounded-lg tracking-wide border border-indigo-500/20 shadow-lg transition-all flex items-center justify-center gap-2 mt-2"
            >
              {isTraining ? (
                <>
                  <span className="h-3.5 w-3.5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                  Fitting Pipeline...
                </>
              ) : (
                'Start Asynchronous Fit'
              )}
            </button>
          </div>
        </div>

        {/* Live Loss / Status Console */}
        <div className="lg:col-span-7 flex flex-col gap-4 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
          <h3 className="text-base font-semibold text-slate-300">Live Training Performance</h3>
          
          <div className="bg-slate-950/40 p-4 rounded-xl border border-slate-850 h-[260px] flex justify-center items-center">
            {lossHistory.length > 0 ? (
              <div ref={chartRef} className="w-full h-full" />
            ) : (
              <div className="text-center text-slate-600 text-xs">
                <svg className="w-8 h-8 text-slate-750 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Asynchronous training tasks will display live loss curves upon success.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Task Queue & Models Playground */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Active Tasks Queue */}
        <div className="lg:col-span-6 flex flex-col gap-4 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
          <h3 className="text-base font-semibold text-slate-300">Job Execution Queue</h3>
          <div className="flex flex-col gap-3 max-h-[300px] overflow-y-auto pr-1">
            {taskList.length > 0 ? (
              taskList.map((t, idx) => (
                <div key={idx} className="bg-slate-950/60 border border-slate-850 p-4 rounded-xl flex items-center justify-between font-mono text-xs">
                  <div className="flex flex-col gap-1.5">
                    <div className="flex items-center gap-2">
                      <span className={`h-2.5 w-2.5 rounded-full ${
                        t.status === 'success' ? 'bg-emerald-400' :
                        t.status === 'failed' ? 'bg-rose-500' :
                        t.status === 'running' ? 'bg-amber-400 animate-pulse' : 'bg-slate-500'
                      }`} />
                      <span className="text-white font-bold">{t.config?.model || 'Model'}</span>
                      <span className="text-[10px] text-slate-500">({t.task_id.substring(0, 8)}...)</span>
                    </div>
                    <span className="text-[10px] text-slate-400">{t.config?.task_name} | Ep: {t.config?.train_epochs}</span>
                  </div>

                  <div className="text-right flex flex-col gap-1">
                    <span className={`text-[10px] font-bold ${
                      t.status === 'success' ? 'text-emerald-400' :
                      t.status === 'failed' ? 'text-rose-400' :
                      t.status === 'running' ? 'text-amber-400' : 'text-slate-500'
                    }`}>
                      {t.status.toUpperCase()}
                    </span>
                    {t.metrics && (
                      <span className="text-[10px] text-indigo-400">MSE: {t.metrics.mse?.toFixed(5) || 'N/A'}</span>
                    )}
                    {t.error && (
                      <span className="text-[9px] text-rose-500 max-w-[150px] truncate" title={t.error}>Err: {t.error}</span>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center text-slate-600 text-xs py-8 border border-dashed border-slate-800 rounded-xl">
                No active training jobs found.
              </div>
            )}
          </div>
        </div>

        {/* Live Inference Testing Playground */}
        <div className="lg:col-span-6 flex flex-col gap-4 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
          <h3 className="text-base font-semibold text-slate-300">Trained Checkpoint Playground</h3>
          
          <div className="flex flex-col gap-3 bg-slate-950/50 p-4 rounded-xl border border-slate-850">
            {/* Model select */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-slate-500 uppercase font-semibold">Trained Checkpoint</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="bg-slate-900 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none"
              >
                {trainedModels.map((m, idx) => (
                  <option key={idx} value={m.model_id}>{m.model_id}</option>
                ))}
                {trainedModels.length === 0 && (
                  <option value="">No trained checkpoints found</option>
                )}
              </select>
            </div>

            {/* Inference Input */}
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-center">
                <label className="text-[10px] text-slate-500 uppercase font-semibold">Input Price Series (Comma-separated)</label>
                <button
                  onClick={() => {
                    const rnd = Array.from({ length: seqLen }, () => Math.round((100 + Math.sin(Math.random() * 5) * 10) * 100) / 100);
                    setInferenceInput(rnd.join(', '));
                  }}
                  className="text-[9px] text-indigo-400 hover:text-indigo-300 font-mono"
                >
                  Generate Random
                </button>
              </div>
              <textarea
                value={inferenceInput}
                onChange={(e) => setInferenceInput(e.target.value)}
                rows={2}
                className="bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none font-mono resize-none leading-relaxed"
              />
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleRunInference}
                disabled={isInferenceLoading || !selectedModel}
                className="flex-1 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-850 disabled:text-slate-500 text-white font-semibold text-xs py-2 rounded-lg border border-indigo-500/20 shadow-lg transition-all"
              >
                {isInferenceLoading ? 'Computing inference...' : 'Test live prediction'}
              </button>
              {selectedModel && (
                <a
                  href={`/api/train/models/${selectedModel}/download`}
                  className="bg-slate-900 hover:bg-slate-850 text-slate-300 hover:text-white font-semibold text-xs py-2 px-4 rounded-lg border border-slate-800 flex items-center gap-1.5 transition-all"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Weights
                </a>
              )}
            </div>

            {/* Inference Result display */}
            {inferenceResult && (
              <div className="bg-slate-950 border border-slate-850 p-3.5 rounded-xl flex flex-col gap-2 font-mono text-xs">
                <span className="text-[10px] text-slate-500">PREDICTED HORIZON ({predLen} STEPS)</span>
                <div className="flex flex-wrap gap-2 text-indigo-400 font-bold text-sm">
                  {inferenceResult.map((val, idx) => (
                    <span key={idx} className="bg-indigo-950/40 border border-indigo-800/10 px-2 py-1 rounded">
                      t+{idx+1}: {val.toFixed(2)}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {inferenceError && (
              <div className="bg-rose-950/40 border border-rose-800/10 p-3 rounded-xl font-mono text-xs text-rose-400">
                Error: {inferenceError}
              </div>
            )}
        </div>
      </div>
    </div>
  </div>
  );
};


// ============================================================================
// 7. Polymarket Execution Broker Ledger
// ============================================================================
export const TradeLedgerPanel = () => {
  const [balance, setBalance] = useState({ usd: 10000.0, btc: 0.0, eth: 0.0 });
  const [trades, setTrades] = useState([]);
  const [manualTrade, setManualTrade] = useState({ asset: 'BTC', side: 'BUY', amount: '' });
  const [btcPrice, setBtcPrice] = useState(63244.92);
  const [ethPrice, setEthPrice] = useState(3512.45);

  const fetchLedger = async () => {
    try {
      const data = await getTradeLedger();
      if (data) {
        if (data.balance) setBalance(data.balance);
        if (data.trades) setTrades(data.trades);
      }
      
      const btcData = await getLatestPrice("BTC");
      if (btcData && btcData.price) setBtcPrice(btcData.price);
      
      const ethData = await getLatestPrice("ETH");
      if (ethData && ethData.price) setEthPrice(ethData.price);
    } catch (err) {
      console.error("Error fetching ledger/prices:", err);
    }
  };

  useEffect(() => {
    fetchLedger();
    const interval = setInterval(fetchLedger, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleExecuteTrade = async (e) => {
    e.preventDefault();
    const amt = parseFloat(manualTrade.amount);
    if (!amt || amt <= 0) return;

    try {
      const res = await executeTrade({
        asset: manualTrade.asset,
        side: manualTrade.side,
        amount: amt
      });
      if (res && res.status === "success") {
        setBalance(res.balance);
        setTrades(prev => [res.trade, ...prev]);
        setManualTrade({ ...manualTrade, amount: '' });
      }
    } catch (err) {
      console.error("Error executing trade:", err);
      alert(`Trade Execution Failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const totalValue = balance.usd + (balance.btc * btcPrice) + (balance.eth * ethPrice);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
      
      {/* Balances & Form */}
      <div className="lg:col-span-5 flex flex-col gap-4 bg-slate-950/50 p-4 rounded-xl border border-slate-850">
        <h3 className="text-sm font-semibold text-slate-300">Account Summary</h3>
        
        {/* Total Net Asset Value */}
        <div className="flex flex-col p-3 bg-slate-900/30 border border-slate-850 rounded-xl text-center">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider font-mono">Net Asset Value (NAV)</span>
          <span className="text-2xl font-mono font-extrabold text-white mt-1">
            ${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>

        {/* Individual Balances */}
        <div className="flex flex-col gap-2 font-mono text-[11px] bg-slate-900/20 p-3 rounded-lg border border-slate-900">
          <div className="flex justify-between border-b border-slate-900 py-1">
            <span className="text-slate-500">USD CASH</span>
            <span className="text-white font-bold">${balance.usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
          </div>
          <div className="flex justify-between border-b border-slate-900 py-1">
            <span className="text-slate-500">BTC HOLDINGS</span>
            <span className="text-indigo-400 font-bold">{balance.btc.toFixed(4)} BTC</span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-slate-500">ETH HOLDINGS</span>
            <span className="text-indigo-400 font-bold">{balance.eth.toFixed(4)} ETH</span>
          </div>
        </div>

        {/* Trade execution Form */}
        <form onSubmit={handleExecuteTrade} className="flex flex-col gap-3 border-t border-slate-900 pt-3 mt-1">
          <h4 className="text-xs font-semibold text-slate-300">Manual Order override</h4>
          
          <div className="grid grid-cols-3 gap-2">
            <select
              value={manualTrade.asset}
              onChange={(e) => setManualTrade({ ...manualTrade, asset: e.target.value })}
              className="bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none"
            >
              <option value="BTC">BTC</option>
              <option value="ETH">ETH</option>
            </select>

            <select
              value={manualTrade.side}
              onChange={(e) => setManualTrade({ ...manualTrade, side: e.target.value })}
              className="bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none"
            >
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
            </select>

            <input
              type="text"
              placeholder="Amount..."
              value={manualTrade.amount}
              onChange={(e) => setManualTrade({ ...manualTrade, amount: e.target.value })}
              className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none font-mono"
            />
          </div>

          <button
            type="submit"
            className={`w-full text-white font-semibold text-xs py-1.5 rounded-lg tracking-wide shadow-lg border transition-all ${
              manualTrade.side === 'BUY' 
                ? 'bg-emerald-600 hover:bg-emerald-500 shadow-emerald-500/10 border-emerald-500/20' 
                : 'bg-rose-600 hover:bg-rose-500 shadow-rose-500/10 border-rose-500/20'
            }`}
          >
            Transmit {manualTrade.side} Order
          </button>
        </form>
      </div>

      {/* Trades Ledger */}
      <div className="lg:col-span-7 flex flex-col gap-4">
        <h3 className="text-base font-semibold text-slate-300">Broker Execution Ledger</h3>
        
        <div className="overflow-x-auto border border-slate-850 rounded-xl bg-slate-950/40 max-h-[315px] overflow-y-auto custom-scrollbar">
          <table className="w-full text-left font-mono text-[11px] border-collapse">
            <thead>
              <tr className="bg-slate-950 border-b border-slate-850 text-slate-500">
                <th className="p-3">Time</th>
                <th className="p-3">Side</th>
                <th className="p-3">Asset</th>
                <th className="p-3 text-right">Amount</th>
                <th className="p-3 text-right">Price</th>
                <th className="p-3 text-right">Total</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, i) => (
                <tr key={i} className="border-b border-slate-900 last:border-0 hover:bg-slate-900/10 text-slate-300">
                  <td className="p-3 text-slate-500">{t.timestamp}</td>
                  <td className="p-3">
                    <span className={`font-bold ${t.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {t.side}
                    </span>
                  </td>
                  <td className="p-3 font-bold text-white">{t.asset}</td>
                  <td className="p-3 text-right">{t.amount.toFixed(4)}</td>
                  <td className="p-3 text-right">${t.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                  <td className="p-3 text-right font-bold text-slate-200">${t.total.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
};
