import React, { useEffect, useState, useRef, useCallback } from 'react';
import { getCandlestickData, getLatestPrice, webSocketService } from '../services/api';
import { format, subDays, isAfter, parseISO } from 'date-fns';
import _ from 'lodash';

const ChartLoading = ({token}) => {
  return (
    <div className="container mx-auto p-4">
      <h2 className="text-2xl font-bold mb-4">Candlestick Chart for {token}</h2>
      <div className="bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 rounded">
        <p className="font-bold">Loading Chart Data</p>
        <p>Fetching {token} price history...</p>
        <div className="mt-2 w-full bg-gray-200 rounded-full h-2.5">
          <div className="bg-blue-600 h-2.5 rounded-full animate-pulse w-3/4"></div>
        </div>
      </div>
    </div>
  );
}

const CandlestickChart = ({ token }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [startDate, setStartDate] = useState(subDays(new Date(), 7)); // Default: 7 days ago
  const [endDate, setEndDate] = useState(new Date());
  const [granularity, setGranularity] = useState(3600); // Default: 1 hour
  const [isLiveUpdating, setIsLiveUpdating] = useState(true); // Default: live updates enabled
  const [latestPrice, setLatestPrice] = useState(null);
  const chartContainer = useRef(null);
  const chart = useRef(null);
  const dataTable = useRef(null);
  const updateInterval = useRef(null);
  const chartData = useRef([]);
  
  // Update frequency in milliseconds (5 seconds by default)
  const UPDATE_FREQUENCY = 5000;

  // Store the latest data point for live updates
  const latestDataPoint = useRef(null);
  const isMounted = useRef(true);

  // Handle live price updates
  useEffect(() => {
    if (!isLiveUpdating || !token) return;

    const handlePriceUpdate = (priceData) => {
      if (!isMounted.current || !priceData || !priceData.price) return;
      
      //console.log('Received price update:', priceData);
      setLatestPrice(priceData.price);
      
      // Only update the chart if we have historical data and the chart is initialized
      if (chartData.current.length > 0 && chart.current) {
        const now = new Date();
        const lastPoint = chartData.current[chartData.current.length - 1];
        
        // If we're still in the same time window, update the last point
        if (isAfter(now, lastPoint.x) && isAfter(now, new Date(lastPoint.x.getTime() + granularity * 1000))) {
          // Create a new point for the current time window
          const newPoint = {
            x: now,
            open: lastPoint.close,
            high: Math.max(lastPoint.close, priceData.price),
            low: Math.min(lastPoint.close, priceData.price),
            close: priceData.price,
            volume: 0 // We don't have volume for live updates
          };
          
          // Add the new point to our data
          const newData = [...chartData.current, newPoint];
          chartData.current = newData;
          
          // Update the chart
          if (chart.current) {
            chart.current.data(newData);
            chart.current.draw();
          }
        } else {
          // Update the last point with the latest price
          const updatedPoint = {
            ...lastPoint,
            high: Math.max(lastPoint.high, priceData.price),
            low: Math.min(lastPoint.low, priceData.price),
            close: priceData.price
          };
          
          // Update the last point in our data
          const newData = [...chartData.current];
          newData[newData.length - 1] = updatedPoint;
          chartData.current = newData;
          
          // Update the chart
          if (chart.current) {
            chart.current.data(newData);
            chart.current.draw();
          }
        }
      }
    };

    // Connect to WebSocket and listen for updates
    webSocketService.connect(token);
    const unsubscribe = webSocketService.onPriceUpdate(handlePriceUpdate);

    // Clean up on unmount or when dependencies change
    return () => {
      unsubscribe();
      webSocketService.disconnect();
    };
  }, [token, isLiveUpdating, granularity]);

  // Clean up on unmount
  useEffect(() => {
    isMounted.current = true;
    
    return () => {
      isMounted.current = false;
      webSocketService.disconnect();
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
        updateInterval.current = null;
      }
      if (chart.current) {
        try {
          chart.current.dispose();
        } catch (e) {
          console.warn('Error disposing chart on unmount:', e);
        }
        chart.current = null;
      }
    };
  }, []);

  const fetchData = async () => {
    console.log('fetchData called');
    if (!isMounted.current) {
      console.log('fetchData: Component not mounted, returning');
      return;
    }
    
    // Skip if no token or container not ready
    if (!token) {
      console.log('fetchData: No token, skipping');
      return;
    }
    
    if (!chartContainer.current) {
      console.log('fetchData: chartContainer not ready, skipping');
      // Don't return here, we'll try to initialize anyway
    }
    
    console.log('Starting data fetch...');
    setLoading(true);
    setError(null);
    
    try {
      console.log('Fetching candlestick data...', { token, startDate, endDate, granularity });
      const data = await getCandlestickData(token, startDate, endDate, granularity);
      console.log('Received data from API:', data ? `Array(${data.length})` : 'null', data);
      
      if (!data || !Array.isArray(data) || data.length === 0) {
        console.warn('No data received or empty array from API');
        const errorMsg = "No data available for the selected time range.";
        console.warn(errorMsg);
        setError(errorMsg);
        setLoading(false);
        return;
      }
      
      console.log(`Received ${data.length} data points`);
      
      // Format data for the chart
      const mappedData = data.map(item => ({
        x: new Date(item.timestamp), // Convert timestamp to Date object
        open: parseFloat(item.open),
        high: parseFloat(item.high),
        low: parseFloat(item.low),
        close: parseFloat(item.close),
        volume: parseFloat(item.volume) || 0
      }));

      // Store the data for later use
      chartData.current = mappedData;
      
      console.log('Rendering chart with data:', mappedData);
      console.log('Data mapped, rendering chart...');
      renderChart(mappedData);
    } catch (err) {
      console.error('Error in fetchData:', err);
      setError(err.message || "An error occurred while fetching data.");
    } finally {
      console.log('Fetch completed, setting loading to false');
      setLoading(false);
    }
  };

  const renderChart = (data) => {
    if (!data || data.length === 0) {
      console.error('No data provided to render chart');
      setError('No data available to display the chart.');
      return;
    }

    // Ensure chart container is available and has dimensions
    if (!chartContainer.current) {
      console.error('Chart container not found');
      return;
    }

    // Make sure the container has dimensions
    chartContainer.current.style.height = '600px';
    chartContainer.current.style.width = '100%';

    // Clean up previous chart if it exists
    if (chart.current) {
      try {
        chart.current.dispose();
      } catch (e) {
        console.warn('Error disposing previous chart:', e);
      }
      chart.current = null;
    }

    console.log('Loading AnyStock...');
    
    // Load AnyStock dynamically
    import('anychart').then((anychart) => {
      console.log('AnyStock loaded, initializing chart...');
      
      try {
        // Set theme
        anychart.theme('lightBlue');
        
        // Create stock chart
        const stockChart = anychart.stock();
        
        // Enable stock tools UI
        //stockChart.toolbar().enabled(true);
        
        // Create data table
        const table = anychart.data.table('x');
        
        // Ensure data is in the correct format
        const formattedData = data.map(item => ({
          x: item.x.getTime ? item.x.getTime() : new Date(item.x).getTime(),
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
          volume: item.volume || 0
        }));
        
        console.log('Adding data to table:', formattedData);
        table.addData(formattedData);
        dataTable.current = table;
        
        // Mapping for candlestick series
        const mapping = table.mapAs({
          x: 'x',
          open: 'open',
          high: 'high',
          low: 'low',
          close: 'close',
          value: 'close'
        });
        
        // Create first plot with candlestick series
        const plot = stockChart.plot(0);
        plot.yGrid(true).xGrid(true);
        plot.yMinorGrid(true).xMinorGrid(true);
        
        // Configure axes
        // plot.xAxis().title('Time');
        //plot.yAxis().title('Price');
        
        // Create candlestick series
        const candlestickSeries = plot.candlestick(mapping);
        candlestickSeries.name(token + ' Price');
        
        // Customize the appearance of candlesticks
        candlestickSeries.risingStroke('#0f9d58');
        candlestickSeries.risingFill('#0f9d58');
        candlestickSeries.fallingStroke('#db4437');
        candlestickSeries.fallingFill('#db4437');
        
        // Set chart title
        stockChart.title(`${token} Price Chart${isLiveUpdating ? ' (Live)' : ''}`);
        
        // Create second plot for volume if available
        if (data[0].volume) {
          const volumeMapping = table.mapAs({
            x: 'x',
            value: 'volume'
          });
          
          const volumePlot = stockChart.plot(1);
          volumePlot.height('30%');
          volumePlot.yAxis().title('Volume');
          volumePlot.yAxis().labels().format('{%Value}{scale:(1)(K)(M)(B)}');
          
          const volumeSeries = volumePlot.column(volumeMapping);
          volumeSeries.name('Volume');
          volumeSeries.zIndex(100);
          
          // Color volume based on price change
          volumeSeries.colorScale(anychart.scales.ordinalColor([
            { less: 0, color: '#db4437' },
            { from: 0, color: '#0f9d58' }
          ]));
          
          // Add scroller
          stockChart.scroller().candlestick(mapping);
        }
        
        // Configure the chart container
        console.log('Setting up chart container...');
        
        // Draw the chart
        console.log('Drawing chart...');
        stockChart.container(chartContainer.current);
        stockChart.draw();
        
        // Save chart reference for cleanup
        chart.current = stockChart;
        console.log('Chart rendered successfully');
        
      } catch (error) {
        console.error('Error rendering chart:', error);
        setError(`Failed to render chart: ${error.message}`);
      }
      }).catch(err => {
        console.error('Failed to load AnyStock:', err);
        setError(`Failed to load charting library: ${err.message}. Please try refreshing the page.`);
      });
  };

  // Function to update the latest price data
  const updateLatestPrice = useCallback(async () => {
    if (!isLiveUpdating || !token || !dataTable.current || !chart.current) return;
    
    try {
      const latestPrice = await getLatestPrice(token);
      
      if (!latestPrice) return;
      
      // Format for the data table
      const formattedPrice = {
        x: latestPrice.timestamp,
        open: latestPrice.open || latestPrice.price,
        high: latestPrice.high || latestPrice.price,
        low: latestPrice.low || latestPrice.price,
        close: latestPrice.close || latestPrice.price,
        volume: latestPrice.volume || 0
      };
      
      // Get the last existing candle
      const lastCandle = chartData.current[chartData.current.length - 1];
      
      // Check if the new price belongs to the current candle or is a new one
      if (lastCandle && 
          new Date(lastCandle.x).getTime() + granularity * 1000 > new Date(formattedPrice.x).getTime()) {
        // Update the existing candle
        lastCandle.close = formattedPrice.close;
        lastCandle.high = Math.max(lastCandle.high, formattedPrice.high);
        lastCandle.low = Math.min(lastCandle.low, formattedPrice.low);
        lastCandle.volume = (lastCandle.volume || 0) + (formattedPrice.volume || 0);
        
        // Replace the last candle with the updated one
        dataTable.current.remove(lastCandle.x);
        dataTable.current.addData([lastCandle]);
      } else {
        // Add as a new candle
        dataTable.current.addData([formattedPrice]);
        chartData.current.push(formattedPrice);
      }
      
      // Update the chart title to indicate live status
      chart.current.title(`${token} Price Chart (Live)`);
    } catch (error) {
      console.error('Error updating latest price:', error);
    }
  }, [token, granularity, isLiveUpdating]);

  // Set up the initial data load
  useEffect(() => {
    // Skip if component is not mounted or no token
    if (!isMounted.current || !token) return;
    
    // Set a flag to track if the component is still mounted
    let isActive = true;
    
    const loadData = async () => {
      try {
        await fetchData();
      } catch (error) {
        console.error('Error loading chart data:', error);
        if (isActive) {
          setError(`Failed to load chart data: ${error.message}`);
        }
      }
    };
    
    console.log('Chart: ', chart.current)
    console.log('Chart container: ', chartContainer.current)
    // Ensure the container is mounted before trying to render
    if (chartContainer.current) {
      console.log('Chart container is mounted, loading data...', chartContainer.current)
      
      loadData();
    } else {
      console.log('Chart container is not mounted, waiting...')
      // If container isn't ready, wait for a short time and try again
      const timer = setTimeout(() => {
        if (isMounted.current && chartContainer.current) {
          console.log('Chart container is mounted, loading data...')
          loadData();
        } else {
          console.log("ERROR! Chart never mounted?!?")
        }
      }, 100);
      
      return () => clearTimeout(timer);
    }
    
    // Cleanup
    return () => {
      console.log('Chart cleanup...')
      isActive = false;
      if (chart.current) {
        console.log('Disposing chart...')
        try {
          chart.current.dispose();
        } catch (e) {
          console.warn('Error disposing chart:', e);
        }
        chart.current = null;
      }
      if (updateInterval.current) {
        console.log('Clearing update interval...')
        clearInterval(updateInterval.current);
        updateInterval.current = null;
      }
    };
  }, [token, startDate, endDate, granularity]);

  // Set up the live update interval
  useEffect(() => {
    if (!isMounted.current) return;
    
    if (isLiveUpdating && token) {
      // Clear any existing interval
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
        updateInterval.current = null;
      }
      
      // Create a new interval
      updateInterval.current = setInterval(() => {
        if (isMounted.current) {
          updateLatestPrice().catch(error => {
            console.error('Error in live update:', error);
          });
        }
      }, UPDATE_FREQUENCY);
      
      // Start with an immediate update
      updateLatestPrice().catch(error => {
        console.error('Error in initial live update:', error);
      });
    } else if (updateInterval.current) {
      clearInterval(updateInterval.current);
      updateInterval.current = null;
      
      // Update the chart title to remove live indication
      if (chart.current) {
        try {
          chart.current.title(`${token} Price Chart`);
        } catch (e) {
          console.warn('Error updating chart title:', e);
        }
      }
    }
    
    return () => {
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
        updateInterval.current = null;
      }
    };
  }, [isLiveUpdating, token, updateLatestPrice]);

  const handleStartDateChange = (event) => {
    setStartDate(new Date(event.target.value));
  };

  const handleEndDateChange = (event) => {
    setEndDate(new Date(event.target.value));
  };

  const handleGranularityChange = (event) => {
    setGranularity(parseInt(event.target.value, 10));
  };

  const toggleLiveUpdates = () => {
    const newState = !isLiveUpdating;
    setIsLiveUpdating(newState);
    
    if (newState) {
      // If enabling live updates, ensure we're connected
      if (token) {
        webSocketService.connect(token);
      }
    } else {
      // If disabling live updates, disconnect
      webSocketService.disconnect();
    }
  };

  if (!token) {
    return <div>Please select a token.</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  // Add a ref to track if we've attempted to initialize
  const hasInitialized = useRef(false);
  
  // Effect to handle initial render and container availability
  useEffect(() => {
    console.log('Initialization effect running', { 
      hasInitialized: hasInitialized.current, 
      token, 
      hasContainer: !!chartContainer.current 
    });
    
    if (!hasInitialized.current && token) {
      console.log('Initializing chart...');
      hasInitialized.current = true;
      
      // Use a small timeout to ensure the container is mounted
      const timer = setTimeout(() => {
        console.log('Delayed initialization - fetching data');
        fetchData().catch(error => {
          console.error('Error in initial fetch:', error);
          setError(`Failed to load chart data: ${error.message}`);
        });
      }, 100);
      
      return () => clearTimeout(timer);
    }
  }, [token, chartContainer.current]);
  
  return (
    <div className="container mx-auto p-4">
      <h2 className="text-2xl font-bold mb-4">Candlestick Chart for {token}</h2>
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4 rounded" role="alert">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      )}
      {loading ? (
         <ChartLoading token={token} />
      ) : (
        <div className="flex flex-wrap gap-4 mb-4">
          <div>
            <label htmlFor="start-date" className="block text-sm font-medium text-gray-700">Start Date:</label>
            <input
              type="date"
              id="start-date"
              value={format(startDate, 'yyyy-MM-dd')}
              onChange={handleStartDateChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            />
          </div>
          <div>
            <label htmlFor="end-date" className="block text-sm font-medium text-gray-700">End Date:</label>
            <input
              type="date"
              id="end-date"
              value={format(endDate, 'yyyy-MM-dd')}
              onChange={handleEndDateChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            />
          </div>
          <div>
            <label htmlFor="granularity" className="block text-sm font-medium text-gray-700">Granularity:</label>
            <select
              id="granularity"
              value={granularity}
              onChange={handleGranularityChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            >
              <option value={60}>1 Minute</option>
              <option value={300}>5 Minutes</option>
              <option value={900}>15 Minutes</option>
              <option value={3600}>1 Hour</option>
              <option value={86400}>1 Day</option>
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={toggleLiveUpdates}
              className={`px-4 py-2 rounded-md text-white font-medium ${
                isLiveUpdating ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
              }`}
            >
              {isLiveUpdating ? 'Pause Live Updates' : 'Enable Live Updates'}
            </button>
          </div>
        </div>
      )}
      <div className="bg-white rounded-lg shadow p-4">
        {latestPrice && (
          <div className="mb-4 p-2 bg-gray-50 rounded">
            <h3 className="text-lg font-semibold">{token} Price: ${latestPrice.toFixed(2)}</h3>
          </div>
        )}
        <div ref={chartContainer} className="w-full" style={{ minHeight: '600px' }}></div>
      </div>
    </div>
  );
};

export default CandlestickChart;