import React, { useState } from 'react';
import CandlestickChart from './components/CandlestickChart';
import { getLatestPrice } from './services/api';

const App = () => {
    const [selectedToken, setSelectedToken] = useState('BTC');
    const [latestPrice, setLatestPrice] = useState(null);


    const handleTokenChange = async (event) => {
        const newToken = event.target.value;
        setSelectedToken(newToken);
        try {
          const price = await getLatestPrice(newToken);
          setLatestPrice(price);
        } catch (error) {
            console.error("Failed to fetch latest price:", error);
            setLatestPrice(null); // Reset or indicate an error
        }
    };

    return (
        <div className="min-h-screen bg-gray-100">
             <header className="bg-white shadow">
                <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
                    <h1 className="text-3xl font-bold tracking-tight text-gray-900 flex items-center">
                        Crypto Price Dashboard
                        {latestPrice !== null && (
                            <span className="ml-4 text-xl text-gray-600">
                                Latest Price: {latestPrice.toFixed(2)}
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
                    <CandlestickChart token={selectedToken} />
                </div>
            </main>
        </div>
    );
};

export default App;