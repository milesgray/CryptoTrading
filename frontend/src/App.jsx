import React, { useState } from 'react';
import CandlestickChart from './components/CandlestickChart';
import OrderBookPanel from './components/OrderBookPanel';
import { getLatestPrice } from './services/api';

const App = () => {
    const [selectedToken, setSelectedToken] = useState('BTC');
    const [latestPriceData, setLatestPriceData] = useState(null);


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
        <div className="min-h-screen bg-gray-100">
             <header className="bg-white shadow">
                <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
                    <h1 className="text-3xl font-bold tracking-tight text-gray-900 flex items-center">
                        Crypto Price Dashboard
                        {latestPriceData?.price && (
                            <span className="ml-4 text-xl text-gray-600">
                                Latest Price: ${latestPriceData.price.toFixed(2)}
                                {latestPriceData.volume && (
                                    <span className="ml-2 text-sm text-gray-500">
                                        Vol: {latestPriceData.volume.toFixed(0)}
                                    </span>
                                )}
                            </span>
                        )}
                    </h1>

                </div>
            </header>

            <main>
                <div className="mx-auto max-w-7xl py-6 sm:px-6 lg:px-8">
                      <div className="mb-4">
                        <label htmlFor="token-select" className="block text-sm font-medium text-gray-700">Select Token:</label>
                        <select
                          id="token-select"
                          value={selectedToken}
                          onChange={handleTokenChange}
                          className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                        >
                          <option value="BTC">BTC/USDT</option>                          

                        </select>
                    </div>
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <div className="lg:col-span-2">
                            <CandlestickChart token={selectedToken} />
                        </div>
                        <div className="lg:col-span-1">
                            <OrderBookPanel token={selectedToken} latestPriceData={latestPriceData} />
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default App;