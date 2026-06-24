import React, { useState, useEffect, useRef } from 'react';
import * as echarts from 'echarts';

// ============================================================================
// 1. Price Ingestion & Recording Panel
// ============================================================================
export const PriceIngestionPanel = () => {
  const [feeds, setFeeds] = useState([
    { exchange: 'Binance Spot', symbol: 'BTC/USDT', status: 'ACTIVE', latency: '42ms', price: 63245.5 },
    { exchange: 'Coinbase Spot', symbol: 'BTC/USD', status: 'ACTIVE', latency: '68ms', price: 63248.2 },
    { exchange: 'OKX Swap', symbol: 'BTC/USDT-SWAP', status: 'ACTIVE', latency: '52ms', price: 63243.0 },
    { exchange: 'Bybit Linear', symbol: 'BTC/USDT', status: 'ACTIVE', latency: '55ms', price: 63244.8 },
    { exchange: 'Bitmex Inverse', symbol: 'BTC/USD', status: 'STALE', latency: '820ms', price: 63238.0 },
    { exchange: 'Deribit Option', symbol: 'BTC-USD-PERP', status: 'ACTIVE', latency: '72ms', price: 63246.1 }
  ]);
  const [indexPrice, setIndexPrice] = useState(63244.92);
  const [tickCount, setTickCount] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      // Mock price updates
      setFeeds(prev => prev.map(f => {
        if (f.status === 'STALE' && Math.random() < 0.1) {
          return { ...f, status: 'ACTIVE', latency: '95ms', price: indexPrice + (Math.random() - 0.5) * 15 };
        }
        if (f.status === 'ACTIVE' && Math.random() < 0.05) {
          return { ...f, status: 'STALE', latency: '1200ms' };
        }
        if (f.status === 'ACTIVE') {
          return { ...f, price: f.price + (Math.random() - 0.5) * 10, latency: `${Math.floor(Math.random() * 40 + 30)}ms` };
        }
        return f;
      }));

      // Calculate composite price index
      setIndexPrice(prev => prev + (Math.random() - 0.5) * 4);
      setTickCount(t => t + 1);
    }, 800);
    return () => clearInterval(interval);
  }, [indexPrice]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
      {/* Feed Status List */}
      <div className="lg:col-span-2 flex flex-col gap-4">
        <div className="flex justify-between items-center">
          <h3 className="text-base font-semibold text-slate-300">Exchange API Feeds</h3>
          <span className="text-xs bg-emerald-500/10 text-emerald-400 px-2.5 py-0.5 rounded-full border border-emerald-500/20 font-semibold">
            {feeds.filter(f => f.status === 'ACTIVE').length} / {feeds.length} Online
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
        </div>
      </div>

      {/* Composite Order Book Capping */}
      <div className="flex flex-col gap-4 bg-slate-950/50 p-4 rounded-xl border border-slate-800">
        <h3 className="text-sm font-semibold text-slate-300">Composite Index Formulation</h3>
        
        <div className="flex flex-col items-center justify-center p-4 bg-slate-900/30 border border-slate-850 rounded-xl text-center">
          <span className="text-xs text-slate-500 uppercase tracking-wider font-mono">Rollbit Index Price</span>
          <span className="text-3xl font-mono font-extrabold text-emerald-400 mt-2 bg-slate-950 px-4 py-1.5 rounded-lg border border-slate-800 shadow-inner">
            ${indexPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <div className="flex items-center gap-1.5 text-[10px] text-slate-500 font-mono mt-3">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping" />
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

  const handleSearch = () => {
    setSearching(true);
    setTimeout(() => {
      // Mock pgvector retrieval results
      const mockSetups = [
        { id: 1042, similarity: 0.942, direction: 1, profit: 4.85, leverage: 10, duration: 15, symbol: 'BTC' },
        { id: 3120, similarity: 0.891, direction: 1, profit: 3.20, leverage: 8, duration: 25, symbol: 'BTC' },
        { id: 894, similarity: 0.865, direction: -1, profit: -1.42, leverage: 5, duration: 40, symbol: 'BTC' },
        { id: 4503, similarity: 0.838, direction: 1, profit: 2.10, leverage: 10, duration: 12, symbol: 'BTC' }
      ];
      setResults(mockSetups);
      setSearching(false);
    }, 600);
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
  const [tweets, setTweets] = useState([
    { user: '@CryptoGains', text: 'BTC holding the support levels perfectly. Strong order book pressure showing up on spot, looks highly bullish for the next candle! 🚀', score: 0.85 },
    { user: '@MacroCrypto', text: 'Markets choppy today. Volumes declining. Better stay on sidelines until break of consolidation range.', score: 0.05 },
    { user: '@WhaleAlert', text: 'Significant sell pressure coming from exchange inflows. Volatility rising, expect sharp downward move soon.', score: -0.68 }
  ]);
  const [btcIndex, setBtcIndex] = useState(68.5);
  const [ethIndex, setEthIndex] = useState(54.2);

  useEffect(() => {
    const handleNewTweet = () => {
      const users = ['@AlphaTrader', '@CryptoWizard', '@BlockNews', '@DefiWhale', '@BitKing'];
      const phrases = [
        { text: 'Unbelievable bids piling up on ETH futures. Spot spread narrowing. Bull run in progress! 🔥', score: 0.91 },
        { text: 'USDT inflows spiking on exchanges. Margin traders expanding leverage. Market is extremely primed.', score: 0.76 },
        { text: 'Minor liquidation squeeze on BTC shorts. Expected consolidation before next leg down.', score: -0.15 },
        { text: 'Regulatory FUD creeping back. Macro environment unfavorable. Spot volume dry, reducing positions.', score: -0.55 }
      ];
      
      const newPhrase = phrases[Math.floor(Math.random() * phrases.length)];
      setTweets(prev => [
        { user: users[Math.floor(Math.random() * users.length)], ...newPhrase },
        ...prev.slice(0, 4)
      ]);

      // Shift indices slightly
      setBtcIndex(b => Math.max(0, Math.min(100, b + newPhrase.score * 5)));
      setEthIndex(e => Math.max(0, Math.min(100, e + newPhrase.score * 4)));
    };

    const interval = setInterval(handleNewTweet, 6000);
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
                  'bg-gray-950 text-gray-400 border border-gray-800'
                }`}>
                  VADER: {t.score > 0 ? '+' : ''}{t.score.toFixed(2)}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed font-sans">{t.text}</p>
            </div>
          ))}
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
  const [activeRegime, setActiveRegime] = useState('BULLISH_HIGH_VOL');
  const [leverageMultiplier, setLeverageMultiplier] = useState(8.5);
  
  // Transition probabilities
  const matrix = {
    BULLISH_HIGH_VOL: { BULLISH_HIGH_VOL: 0.72, BULLISH_LOW_VOL: 0.18, BEARISH_HIGH_VOL: 0.08, BEARISH_LOW_VOL: 0.02 },
    BULLISH_LOW_VOL: { BULLISH_HIGH_VOL: 0.15, BULLISH_LOW_VOL: 0.75, BEARISH_HIGH_VOL: 0.02, BEARISH_LOW_VOL: 0.08 },
    BEARISH_HIGH_VOL: { BULLISH_HIGH_VOL: 0.05, BULLISH_LOW_VOL: 0.02, BEARISH_HIGH_VOL: 0.68, BEARISH_LOW_VOL: 0.25 },
    BEARISH_LOW_VOL: { BULLISH_HIGH_VOL: 0.02, BULLISH_LOW_VOL: 0.08, BEARISH_HIGH_VOL: 0.18, BEARISH_LOW_VOL: 0.72 }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate regime transitions
      const probs = matrix[activeRegime];
      const rand = Math.random();
      let cumulative = 0;
      let nextRegime = activeRegime;
      
      for (const [regime, prob] of Object.entries(probs)) {
        cumulative += prob;
        if (rand <= cumulative) {
          nextRegime = regime;
          break;
        }
      }
      
      if (nextRegime !== activeRegime) {
        setActiveRegime(nextRegime);
        // Adjust leverage ceilings based on regime
        if (nextRegime === 'BULLISH_HIGH_VOL') setLeverageMultiplier(8.5 + Math.random());
        else if (nextRegime === 'BULLISH_LOW_VOL') setLeverageMultiplier(10.0 - Math.random());
        else if (nextRegime === 'BEARISH_HIGH_VOL') setLeverageMultiplier(3.0 + Math.random());
        else setLeverageMultiplier(5.5 - Math.random());
      }
    }, 8000);
    return () => clearInterval(interval);
  }, [activeRegime]);

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
          <span className={`text-sm font-bold px-4 py-1.5 rounded-lg border shadow-inner mt-2 inline-block ${getRegimeStyle(activeRegime)}`}>
            {activeRegime.replace(/_/g, ' ')}
          </span>
          
          <div className="flex flex-col items-center mt-6">
            <span className="text-xs text-slate-500 uppercase tracking-wider font-mono">Dynamic Leverage Cap</span>
            <span className="text-4xl font-mono font-extrabold text-indigo-400 mt-2 bg-slate-950 px-4 py-1.5 rounded-lg border border-slate-850 shadow-inner">
              {leverageMultiplier.toFixed(1)}x
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

  useEffect(() => {
    const interval = setInterval(() => {
      // Mock book pressure fluctuations
      setOfi(prev => Math.max(-10, Math.min(10, prev + (Math.random() - 0.5) * 2)));
      setCvd(prev => prev + (Math.random() - 0.4) * 300);
      setBap(prev => Math.max(10, Math.min(90, prev + (Math.random() - 0.5) * 6)));
    }, 1200);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
      
      {/* 1. Bid-Ask Pressure Meter */}
      <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-850 flex flex-col gap-3 items-center text-center justify-center">
        <h4 className="text-xs text-slate-500 uppercase font-mono tracking-wider">Bid-Ask Pressure</h4>
        
        <div className="w-32 h-32 rounded-full border-4 border-slate-800 flex items-center justify-center relative bg-slate-900/20 shadow-inner">
          <div className="flex flex-col">
            <span className="text-3xl font-mono font-extrabold text-indigo-400">{bap.toFixed(0)}%</span>
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
        
        <div className={`text-4xl font-mono font-extrabold py-3 px-6 rounded-lg border bg-slate-950 shadow-inner ${
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
        
        <div className={`text-4xl font-mono font-extrabold py-3 px-6 rounded-lg border bg-slate-950 shadow-inner ${
          cvd > 0 ? 'text-emerald-400 border-emerald-500/10' : 'text-rose-400 border-rose-500/10'
        }`}>
          {cvd > 0 ? '+' : ''}{cvd.toFixed(0)}
        </div>

        <span className="text-[10px] text-slate-500 leading-relaxed font-sans max-w-[180px]">
          Volume delta measures net aggressive buying (market buys) minus aggressive selling (market sells) in contracts.
        </span>
      </div>

    </div>
  );
};


// ============================================================================
// 6. PyTorch Model Training Console
// ============================================================================
export const ModelTrainingConsole = () => {
  const [modelType, setModelType] = useState('TimesNet');
  const [learningRate, setLearningRate] = useState('0.001');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(32);
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(0.0);
  
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [lossHistory, setLossHistory] = useState([]);

  // Initialize and update training loss chart
  useEffect(() => {
    if (!chartRef.current) return;
    
    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    const option = {
      title: {
        text: 'Live Training Loss Curves',
        textStyle: { color: '#a0aec0', fontSize: 13, fontWeight: 'normal' },
        left: 'center'
      },
      grid: { left: '5%', right: '5%', top: '18%', bottom: '8%', containLabel: true },
      xAxis: {
        type: 'category',
        data: Array.from({ length: lossHistory.length }, (_, i) => i + 1),
        axisLine: { lineStyle: { color: '#4a5568' } },
        axisLabel: { color: '#718096' }
      },
      yAxis: {
        type: 'value',
        name: 'Loss',
        axisLine: { lineStyle: { color: '#4a5568' } },
        axisLabel: { color: '#718096' },
        splitLine: { lineStyle: { color: '#2d3748' } }
      },
      series: [{
        name: 'Total Loss',
        type: 'line',
        data: lossHistory,
        smooth: true,
        itemStyle: { color: '#818cf8' },
        lineStyle: { width: 2.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(129,140,248,0.3)' },
            { offset: 1, color: 'rgba(129,140,248,0)' }
          ])
        }
      }],
      backgroundColor: 'transparent'
    };

    chartInstance.current.setOption(option);
  }, [lossHistory]);

  const handleStartTraining = () => {
    setIsTraining(true);
    setCurrentEpoch(0);
    setLossHistory([]);
    
    let epoch = 1;
    let baseLoss = modelType === 'TimesNet' ? 0.85 : modelType === 'JEPA' ? 1.45 : 0.65;
    
    const trainStep = () => {
      if (epoch > epochs) {
        setIsTraining(false);
        return;
      }
      
      const newLoss = baseLoss * Math.pow(0.82, epoch) + Math.random() * 0.04;
      setCurrentEpoch(epoch);
      setCurrentLoss(newLoss);
      setLossHistory(prev => [...prev, newLoss]);
      
      epoch++;
      setTimeout(trainStep, 1200); // Wait 1.2s per epoch to simulate step
    };

    setTimeout(trainStep, 600);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-slate-900/30 p-6 rounded-2xl border border-slate-800 backdrop-blur-md">
      
      {/* Parameter Configuration */}
      <div className="lg:col-span-5 flex flex-col gap-4">
        <h3 className="text-base font-semibold text-slate-300">PyTorch Trainer Console</h3>
        
        <div className="flex flex-col gap-3 bg-slate-950/50 p-4 rounded-xl border border-slate-850">
          {/* Model Select */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-slate-500 uppercase font-semibold">Architecture</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              disabled={isTraining}
              className="bg-slate-900 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-slate-700"
            >
              <option value="TimesNet">TimesNet (Forecaster)</option>
              <option value="Autoformer">Autoformer (Vol Dynamics)</option>
              <option value="JEPA">Koopman-JEPA (Regimes)</option>
            </select>
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

          <button
            onClick={handleStartTraining}
            disabled={isTraining}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:border-slate-800 disabled:text-slate-500 text-white font-semibold text-xs py-2 px-4 rounded-lg tracking-wide shadow-lg shadow-indigo-500/10 border border-indigo-500/20 transition-all flex items-center justify-center gap-2 mt-2"
          >
            {isTraining ? (
              <>
                <span className="h-3.5 w-3.5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                Fitting Epoch {currentEpoch}/{epochs}...
              </>
            ) : (
              'Initialize Fit Pipeline'
            )}
          </button>
        </div>

        {/* Live Metrics Card */}
        {isTraining && (
          <div className="grid grid-cols-2 gap-3 bg-slate-950 p-4 rounded-xl border border-slate-850 font-mono text-xs">
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-500">CURRENT LOSS</span>
              <span className="text-indigo-400 font-extrabold text-lg mt-1">{currentLoss.toFixed(4)}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-500">PROGRESS</span>
              <span className="text-white font-extrabold text-lg mt-1">
                {((currentEpoch / epochs) * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Loss Curves Graph */}
      <div className="lg:col-span-7 flex flex-col gap-4 bg-slate-950/40 p-4 rounded-xl border border-slate-850 h-[280px] justify-center items-center">
        {lossHistory.length > 0 ? (
          <div ref={chartRef} className="w-full h-full" />
        ) : (
          <div className="text-center text-slate-600 text-xs">
            <svg className="w-8 h-8 text-slate-750 mx-auto mb-2 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 12l3-3 3 3 4-4M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
            Fit your deep learning model to inspect real-time training loss curves.
          </div>
        )}
      </div>

    </div>
  );
};


// ============================================================================
// 7. Polymarket Execution Broker Ledger
// ============================================================================
export const TradeLedgerPanel = () => {
  const [balance, setBalance] = useState({ usd: 10000.0, btc: 0.0, eth: 0.0 });
  const [trades, setTrades] = useState([
    { timestamp: new Date(Date.now() - 3600000).toLocaleTimeString(), side: 'BUY', asset: 'BTC', amount: 0.0820, price: 60975.60, total: 5000.0 },
    { timestamp: new Date(Date.now() - 1800000).toLocaleTimeString(), side: 'BUY', asset: 'ETH', amount: 0.8571, price: 3500.00, total: 3000.0 }
  ]);
  const [manualTrade, setManualTrade] = useState({ asset: 'BTC', side: 'BUY', amount: '' });
  const [btcPrice] = useState(63244.92);
  const [ethPrice] = useState(3512.45);

  const handleExecuteTrade = (e) => {
    e.preventDefault();
    const amt = parseFloat(manualTrade.amount);
    if (!amt || amt <= 0) return;

    const price = manualTrade.asset === 'BTC' ? btcPrice : ethPrice;
    const cost = amt * price;

    if (manualTrade.side === 'BUY') {
      if (cost > balance.usd) {
        alert('Insufficient USD Cash Balance');
        return;
      }
      setBalance(prev => ({
        usd: prev.usd - cost,
        btc: prev.btc + (manualTrade.asset === 'BTC' ? amt : 0),
        eth: prev.eth + (manualTrade.asset === 'ETH' ? amt : 0)
      }));
    } else {
      const pos = manualTrade.asset === 'BTC' ? balance.btc : balance.eth;
      if (amt > pos) {
        alert(`Insufficient ${manualTrade.asset} position to sell`);
        return;
      }
      setBalance(prev => ({
        usd: prev.usd + cost,
        btc: prev.btc - (manualTrade.asset === 'BTC' ? amt : 0),
        eth: prev.eth - (manualTrade.asset === 'ETH' ? amt : 0)
      }));
    }

    setTrades(prev => [
      {
        timestamp: new Date().toLocaleTimeString(),
        side: manualTrade.side,
        asset: manualTrade.asset,
        amount: amt,
        price: price,
        total: cost
      },
      ...prev
    ]);
    setManualTrade({ ...manualTrade, amount: '' });
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
