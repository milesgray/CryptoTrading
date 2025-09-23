import React, { useEffect, useState, useRef, useCallback } from 'react';
import { getCandlestickData, webSocketService } from '../services/api';
import { format, subDays } from 'date-fns';
import _ from 'lodash';
import anychart from 'anychart';

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
  const [latestTimestamp, setLatestTimestamp] = useState(null);
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

  // Initialize the data table
  const initDataTable = useCallback(() => {
    if (!chartContainer.current) return null;
    
    try {
      // Create a data table
      const table = anychart.data.table();
      
      // Add column names first
      table.addData([], ['x', 'open', 'high', 'low', 'close', 'volume']);
      
      // Create mapping for the data
      const mapping = table.mapAs({
        x: 0,
        open: 1,
        high: 2,
        low: 3,
        close: 4,
        value: 4, // For tooltips
        volume: 5
      });
      
      console.log('Data table initialized successfully');
      return { table, mapping };
    } catch (error) {
      console.error('Error initializing data table:', error);
      return null;
    }
  }, []);
  
  // Add sample data for testing
  const addSampleData = useCallback(() => {
    if (!dataTable.current?.table) return;
    
    console.log('Adding sample data...');
    const now = Date.now();
    const day = 24 * 60 * 60 * 1000; // ms in a day
    
    // Add 30 days of sample data
    for (let i = 30; i >= 0; i--) {
      const date = new Date(now - (i * day));
      const basePrice = 100000 + (Math.random() * 20000);
      const open = basePrice;
      const close = basePrice * (0.99 + (Math.random() * 0.02));
      const high = Math.max(open, close) * (1 + Math.random() * 0.01);
      const low = Math.min(open, close) * (0.99 - Math.random() * 0.01);
      const volume = 100 + (Math.random() * 100);
      
      dataTable.current.table.addData([
        date,
        open,
        high,
        low,
        close,
        volume
      ]);
    }
    console.log('Sample data added');
  }, []);
  
  // Initialize the chart
  const initChart = useCallback(() => {
    console.log('Initializing chart...');
    
    if (!chartContainer.current) {
      console.error('Chart container not available');
      return;
    }
    
    try {
      console.log('Creating stock chart...');
      // Create a stock chart
      chart.current = anychart.stock();
      
      // Create a plot
      const plot = chart.current.plot(0);
      
      console.log('Creating OHLC series with mapping:', dataTable.current.mapping);
      // Create an OHLC series
      const series = plot.ohlc(dataTable.current.mapping)
        .name('Price')
        .risingStroke('#3ba158')
        .fallingStroke('#fa1c26');
      
      // Enable grid and axis
      plot.yGrid(true).xGrid(true);
      plot.yAxis().title('Price');
      
      console.log('Setting up scroller...');
      // Create scroller
      chart.current.scroller().ohlc(dataTable.current.mapping);
      
      // Add sample data for testing
      addSampleData();
      
      console.log('Setting container and drawing chart...');
      // Set container and draw
      chart.current.container(chartContainer.current);
      chart.current.draw();
      
      console.log('Chart initialization complete');
      
      // Force a resize to ensure the chart renders properly
      setTimeout(() => {
        if (chart.current) {
          console.log('Triggering chart resize...');
          chart.current.fitAll();
          
          // Force redraw after a short delay
          setTimeout(() => {
            if (chart.current) {
              console.log('Forcing chart redraw...');
              chart.current.draw();
            }
          }, 500);
        }
      }, 100);
      
    } catch (error) {
      console.error('Error initializing chart:', error);
    }
  }, [addSampleData]);
  
  // Initialize data table and chart on mount
  useEffect(() => {
    console.log('Component mounted, initializing data table...');
    
    if (!dataTable.current) {
      console.log('No data table found, creating new one...');
      const result = initDataTable();
      if (result?.table && result.mapping) {
        console.log('Data table created successfully, initializing chart...');
        dataTable.current = result;
        initChart();
      } else {
        console.error('Failed to create data table:', result);
      }
    } else {
      console.log('Data table already exists, skipping initialization');
    }
    
    return () => {
      console.log('Cleaning up chart...');
      if (chart.current) {
        try {
          chart.current.dispose();
        } catch (e) {
          console.error('Error disposing chart:', e);
        }
        chart.current = null;
      }
    };
  }, [initDataTable, initChart]);
  
  // Handle live price updates from WebSocket
  useEffect(() => {
    if (!isLiveUpdating || !token) return;

    const handlePriceUpdate = (priceData) => {
      if (!isMounted.current || !priceData || priceData.price === undefined) return;
      
      console.log('Received price update:', priceData);
      const now = new Date(priceData.timestamp || Date.now());
      setLatestPrice(priceData.price);
      
      // If we don't have a data table yet, initialize it
      if (!dataTable.current) {
        console.log('Initializing data table...');
        const result = initDataTable();
        if (result?.table && result.mapping) {
          dataTable.current = result;
          console.log('Data table initialized, initializing chart...');
          initChart();
          console.log('Chart initialization complete');
        } else {
          console.error('Failed to initialize data table');
          return; // Can't proceed without data table
        }
      }

      try {
        // If we have existing data, update the last point or create a new one
        if (chartData.current.length > 0) {
          const lastPoint = chartData.current[chartData.current.length - 1];
          
          if (!lastPoint) {
            console.error('Last point is null or undefined');
            return;
          }
          
          // Check if we're still in the same time window
          const lastPointTime = lastPoint.x instanceof Date ? lastPoint.x : new Date(lastPoint.x);
          const timeDiff = now - lastPointTime;
          const isSameWindow = timeDiff < (granularity * 1000);
          
          if (isSameWindow) {
            // Update the last point
            const updatedPoint = {
              ...lastPoint,
              high: Math.max(lastPoint.high, priceData.price),
              low: Math.min(lastPoint.low, priceData.price),
              close: priceData.price,
              volume: priceData.volume + lastPoint.volume,
              x: now // Update the timestamp to now
            };
            
            // Validate updatedPoint before using it
            if (!updatedPoint || typeof updatedPoint !== 'object') {
              console.error('Invalid updatedPoint:', updatedPoint);
              return;
            }
            
            // Update in memory
            const lastIndex = chartData.current.length - 1;
            if (lastIndex >= 0) {
              chartData.current[lastIndex] = updatedPoint;
            } else {
              console.error('No existing data point to update');
              return;
            }
            
            // Update the data table
            if (dataTable.current?.table) {
              try {
                // Ensure all values are numbers with fallbacks
                const rowData = [
                  now,
                  Number(updatedPoint.open || 0),
                  Number(updatedPoint.high || 0),
                  Number(updatedPoint.low || 0),
                  Number(updatedPoint.close || 0),
                  Number(updatedPoint.volume || 0)
                ];
                
                console.log('Adding data to table:', rowData);
                
                // Add data in the format AnyChart expects
                dataTable.current.table.addData(rowData);
              } catch (error) {
                console.error('Error adding data to table:', error, {
                  updatedPoint,
                  now: now
                });
              }
            }
          } else {
            // Create a new candle
            const newPoint = {
              x: now,
              open: lastPoint.close ?? priceData.price, // Fallback to current price if no last close
              high: priceData.price,
              low: priceData.price,
              close: priceData.price,
              volume: priceData.volume
            };
            
            // Add to our data
            chartData.current.push(newPoint);
            
            // Update the data table
            if (dataTable.current?.table) {
              try {
                // Ensure all values are numbers
                const rowData = [
                  now,
                  Number(newPoint.open),
                  Number(newPoint.high),
                  Number(newPoint.low),
                  Number(newPoint.close),
                  Number(newPoint.volume || 0)
                ];
                
                console.log('Adding new candle to table:', rowData);
                
                // Add data in the format AnyChart expects
                dataTable.current.table.addData(rowData);
              } catch (error) {
                console.error('Error adding new candle to table:', error, {
                  newPoint,
                  now: now.toISOString()
                });
              }
            }
          }
        } else {
          // If we don't have chart data yet, initialize it with the current price
          console.log('Initializing chart with first price point');
          const newPoint = {
            x: now,
            open: priceData.price,
            high: priceData.price,
            low: priceData.price,
            close: priceData.price,
            volume: priceData.volume
          };
          
          chartData.current = [newPoint];
          
          // Add initial data point
          if (dataTable.current?.table) {
            try {
              // Ensure all values are numbers
              const rowData = [
                now,
                Number(newPoint.open),
                Number(newPoint.high),
                Number(newPoint.low),
                Number(newPoint.close),
                Number(newPoint.volume || 0)
              ];
              
              console.log('Adding initial data point:', rowData);
              
              // Add data in the format AnyChart expects
              dataTable.current.table.addData(rowData);
            } catch (error) {
              console.error('Error adding initial data point:', error, {
                newPoint,
                timestamp: now
              });
            }
          }
          
          // Initialize the chart if not already done
          if (!chart.current) {
            initChart();
          }
        }
      } catch (error) {
        console.error('Error in handlePriceUpdate:', error, { priceData });
      }
    };

    // Connect to WebSocket and listen for updates
    const connectWebSocket = async () => {
      try {
        console.log('Connecting to WebSocket for token:', token);
        await webSocketService.connect(token);
        console.log('WebSocket connected successfully');
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
      }
    };

    // Only set up the WebSocket connection after the component is mounted
    // and we have a valid token
    if (token) {
      connectWebSocket();
      
      // Set up the price update handler
      const unsubscribe = webSocketService.onPriceUpdate(handlePriceUpdate);

      // Clean up on unmount or when dependencies change
      return () => {
        console.log('Cleaning up WebSocket connection');
        unsubscribe();
        webSocketService.disconnect();
      };
    }
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
  // Initialize the chart with WebSocket data
  const initializeChart = useCallback(() => {
    if (!isLiveUpdating || !token || !dataTable.current || !chart.current) return;
    
    // Set the chart title to indicate live status
    chart.current.title(`${token} Price Chart (Live)`);
  }, [token, isLiveUpdating]);

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

  // Handle live updates toggle
  useEffect(() => {
    if (!isMounted.current) return;
    
    if (isLiveUpdating && token) {
      // Initialize the chart for live updates
      initializeChart();
      
      // Connect to WebSocket if not already connected
      if (!webSocketService.socket || webSocketService.socket.readyState !== WebSocket.OPEN) {
        webSocketService.connect(token).catch(error => {
          console.error('Error connecting to WebSocket:', error);
          setError('Failed to connect to live price feed');
        });
      }
    } else {
      // Update the chart title to remove live indication
      if (chart.current) {
        try {
          chart.current.title(`${token} Price Chart`);
        } catch (e) {
          console.warn('Error updating chart title:', e);
        }
      }
    }
    
    // Cleanup function
    return () => {
      // We don't disconnect the WebSocket here as it might be used by other components
      // The WebSocketService will handle reconnection if needed
    };
  }, [isLiveUpdating, token, initializeChart]);

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
        <div 
          ref={chartContainer} 
          className="w-full bg-gray-100 border border-gray-300 rounded p-2" 
          style={{ 
            minHeight: '600px',
            height: '600px',
            position: 'relative'
          }}
        >
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-gray-500">Loading chart data...</div>
              <div className="mt-2 text-sm text-gray-400">
                {dataTable.current ? 'Rendering chart...' : 'Initializing data...'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CandlestickChart;