import React, { useState, useEffect } from 'react';
import CandlestickChart from './components/CandlestickChart';
import OrderBookPanel from './components/OrderBookPanel';
import RetrievalVisualizer from './components/RetrievalVisualizer';
import ServiceControlDashboard from './components/ServiceControlDashboard';
import { 
  PriceIngestionPanel, 
  EmbeddingMatcherPanel, 
  SentimentStreamPanel, 
  JepaRegimePanel, 
  OrderBookPressurePanel,
  ModelTrainingConsole,
  TradeLedgerPanel
} from './components/SpecializedServicePanels';
import { getLatestPrice } from './services/api';

const App = () => {
  const [selectedToken, setSelectedToken] = useState('BTC');
  const [latestPriceData, setLatestPriceData] = useState(null);
  const [activeTab, setActiveTab] = useState('market');
  const [analyticsSubTab, setAnalyticsSubTab] = useState('ingestion');

  useEffect(() => {
    const fetchPrice = async () => {
      try {
        const priceData = await getLatestPrice(selectedToken);
        setLatestPriceData(priceData);
      } catch (error) {
        console.error("Failed to fetch latest price:", error);
      }
    };
    fetchPrice();
    const interval = setInterval(fetchPrice, 5000);
    return () => clearInterval(interval);
  }, [selectedToken]);

  const handleTokenChange = async (event) => {
    const newToken = event.target.value;
    setSelectedToken(newToken);
    try {
      const priceData = await getLatestPrice(newToken);
      setLatestPriceData(priceData);
    } catch (error) {
      console.error("Failed to fetch latest price:", error);
      setLatestPriceData(null);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased selection:bg-indigo-500 selection:text-white">
      {/* Global Navbar */}
      <header className="bg-slate-900/80 border-b border-slate-800/80 sticky top-0 z-40 backdrop-blur-md">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          
          {/* Brand Logo & Live Info */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="h-9 w-9 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-600/30">
                <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <div>
                <h1 className="text-lg font-bold tracking-tight text-white leading-none">CryptoTrading</h1>
                <span className="text-[10px] text-slate-500 font-mono tracking-widest uppercase">Quant Framework</span>
              </div>
            </div>
            
            {latestPriceData?.price && (
              <div className="hidden md:flex items-center gap-3 pl-4 border-l border-slate-800">
                <span className="text-xs text-slate-400 font-semibold font-mono">
                  {selectedToken}/USDT:
                </span>
                <span className="text-sm font-mono font-bold text-emerald-400">
                  ${latestPriceData.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </span>
                {latestPriceData.volume && (
                  <span className="text-xs text-slate-500 font-mono">
                    Vol: {latestPriceData.volume.toFixed(0)}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Navigation Tabs */}
          <nav className="flex bg-slate-950/80 p-1 rounded-xl border border-slate-800/80 text-xs font-semibold">
            <button
              onClick={() => setActiveTab('market')}
              className={`px-4 py-2 rounded-lg transition-all duration-150 ${
                activeTab === 'market' 
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/15' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Live Trade Room
            </button>
            <button
              onClick={() => setActiveTab('orchestrator')}
              className={`px-4 py-2 rounded-lg transition-all duration-150 ${
                activeTab === 'orchestrator' 
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/15' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Service Orchestrator
            </button>
            <button
              onClick={() => setActiveTab('analytics')}
              className={`px-4 py-2 rounded-lg transition-all duration-150 ${
                activeTab === 'analytics' 
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/15' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Pipeline Analytics
            </button>
            <button
              onClick={() => setActiveTab('execution')}
              className={`px-4 py-2 rounded-lg transition-all duration-150 ${
                activeTab === 'execution' 
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/15' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Trainer & Trade
            </button>
          </nav>

          {/* Token Selector & Connection Pulse */}
          <div className="flex items-center gap-3">
            <select
              id="token-select"
              value={selectedToken}
              onChange={handleTokenChange}
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs font-semibold text-slate-300 focus:outline-none focus:border-slate-700 font-mono"
            >
              <option value="BTC">BTC/USDT</option>
            </select>

            <div className="flex items-center gap-1.5 text-xs text-slate-500 font-semibold font-mono bg-slate-950/60 px-3 py-1.5 rounded-lg border border-slate-800">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              LIVE
            </div>
          </div>

        </div>
      </header>

      {/* Main Content Body */}
      <main className="mx-auto max-w-7xl p-4 sm:p-6 lg:p-8">
        
        {/* Tab 1: Live Market Trading Charts */}
        {activeTab === 'market' && (
          <div className="flex flex-col gap-6">
            {/* Candlestick & Book */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              <div className="lg:col-span-8 flex flex-col">
                <CandlestickChart token={selectedToken} />
              </div>
              <div className="lg:col-span-4 flex flex-col">
                <OrderBookPanel token={selectedToken} latestPriceData={latestPriceData} />
              </div>
            </div>
            
            {/* Pattern Retrieval Forecast */}
            <div className="w-full">
              <RetrievalVisualizer token={selectedToken} />
            </div>
          </div>
        )}

        {/* Tab 2: Service Orchestration lifecycle manager */}
        {activeTab === 'orchestrator' && (
          <div className="w-full">
            <ServiceControlDashboard />
          </div>
        )}

        {/* Tab 3: Detailed Pipeline Analytics */}
        {activeTab === 'analytics' && (
          <div className="flex flex-col gap-6">
            
            {/* Analytics Navigation Bar */}
            <div className="flex bg-slate-900/60 p-1.5 rounded-xl border border-slate-800/80 self-start text-xs font-bold font-mono">
              <button
                onClick={() => setAnalyticsSubTab('ingestion')}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  analyticsSubTab === 'ingestion' ? 'bg-indigo-950/40 text-indigo-400 border border-indigo-800/20' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                1. Price Ingestion
              </button>
              <button
                onClick={() => setAnalyticsSubTab('embeddings')}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  analyticsSubTab === 'embeddings' ? 'bg-indigo-950/40 text-indigo-400 border border-indigo-800/20' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                2. Contrastive CNN
              </button>
              <button
                onClick={() => setAnalyticsSubTab('pressure')}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  analyticsSubTab === 'pressure' ? 'bg-indigo-950/40 text-indigo-400 border border-indigo-800/20' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                3. Book Pressure
              </button>
              <button
                onClick={() => setAnalyticsSubTab('sentiment')}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  analyticsSubTab === 'sentiment' ? 'bg-indigo-950/40 text-indigo-400 border border-indigo-800/20' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                4. Twitter Sentiment
              </button>
              <button
                onClick={() => setAnalyticsSubTab('jepa')}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  analyticsSubTab === 'jepa' ? 'bg-indigo-950/40 text-indigo-400 border border-indigo-800/20' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                5. JEPA Regimes
              </button>
            </div>

            {/* Sub-tab views */}
            {analyticsSubTab === 'ingestion' && <PriceIngestionPanel />}
            {analyticsSubTab === 'embeddings' && <EmbeddingMatcherPanel />}
            {analyticsSubTab === 'pressure' && <OrderBookPressurePanel />}
            {analyticsSubTab === 'sentiment' && <SentimentStreamPanel />}
            {analyticsSubTab === 'jepa' && <JepaRegimePanel />}
          </div>
        )}

        {/* Tab 4: Trainer & Trade Ledger */}
        {activeTab === 'execution' && (
          <div className="flex flex-col gap-6">
            {/* PyTorch Model Training */}
            <div className="w-full">
              <ModelTrainingConsole />
            </div>

            {/* Simulated Polymarket Broker */}
            <div className="w-full">
              <TradeLedgerPanel />
            </div>
          </div>
        )}

      </main>
    </div>
  );
};

export default App;