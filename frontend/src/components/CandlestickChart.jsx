import React, { useEffect, useState, useRef, useCallback } from 'react';
import { getCandlestickData, webSocketService } from '../services/api';
import { format, subDays } from 'date-fns';
import _ from 'lodash';
import anychart from 'anychart';

const ChartLoading = ({ token }) => {
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
  const [isLiveUpdating, setIsLiveUpdating] = useState(false); // Start with live updates disabled
  const [latestPrice, setLatestPrice] = useState(null);
  const [latestTimestamp, setLatestTimestamp] = useState(null);
  const [historicalDataLoaded, setHistoricalDataLoaded] = useState(false); // Track if historical data is loaded
  const chartContainer = useRef(null);
  const chart = useRef(null);
  const dataTable = useRef(null);
  const updateInterval = useRef(null);
  const chartData = useRef([]);
  const chartInitializing = useRef(false);

  // Update frequency in milliseconds (5 seconds by default)
  const UPDATE_FREQUENCY = 5000;

  // Store the latest data point for live updates
  const latestDataPoint = useRef(null);
  const isMounted = useRef(true);

  // Initialize the data table


  // Handle live price updates from WebSocket
  useEffect(() => {
    if (!isLiveUpdating || !token) return;

    const handlePriceUpdate = (priceData) => {
      if (!isMounted.current || !priceData || priceData.price === undefined) return;

      // Only process updates if historical data has been loaded
      if (!historicalDataLoaded) {
        console.log('Historical data not loaded yet, ignoring price update');
        return;
      }

      console.log('Received price update:', priceData);
      const now = new Date(priceData.timestamp || Date.now());
      setLatestPrice(priceData.price);

      // If we don't have a data table yet, initialize it with minimal setup
      if (!dataTable.current) {
        console.log('No data table, creating minimal one for live updates...');
        try {
          const table = anychart.data.table('x');
          dataTable.current = {
            table: table,
            mapping: table.mapAs({
              x: 'x',
              open: 'open',
              high: 'high',
              low: 'low',
              close: 'close',
              value: 'close'
            })
          };

          if (!chart.current && chartContainer.current && !chartInitializing.current) {
            if (chartContainer.current.offsetWidth === 0 || chartContainer.current.offsetHeight === 0) {
              return;
            }
            chartInitializing.current = true;
            import('anychart').then((anychart) => {
              try {
                const stockChart = anychart.stock();
                const table = dataTable.current.table;
                const mapping = dataTable.current.mapping;
                const plot = stockChart.plot(0);
                plot.yGrid(true).xGrid(true);
                plot.yMinorGrid(true).xMinorGrid(true);

                const candlestickSeries = plot.candlestick(mapping);
                candlestickSeries.name(token + ' Price');
                candlestickSeries.risingStroke('#0f9d58');
                candlestickSeries.risingFill('#0f9d58');
                candlestickSeries.fallingStroke('#db4437');
                candlestickSeries.fallingFill('#db4437');
                candlestickSeries.pointWidth('90%');

                stockChart.title(`${token} Price Chart (Live)`);
                stockChart.container(chartContainer.current);
                stockChart.draw();

                chart.current = stockChart;
                chartInitializing.current = false;
              } catch (err) {
                console.error(err);
                chartInitializing.current = false;
              }
            });
          }
        } catch (error) {
          console.error(error);
          return;
        }
      }

      try {
        // Aggregate ticks into the current granularity time bucket
        const bucketSizeMs = (granularity || 3600) * 1000;
        const bucketStartMs = Math.floor(now.getTime() / bucketSizeMs) * bucketSizeMs;

        let activePoint = null;
        const lastIdx = chartData.current.length - 1;

        if (lastIdx >= 0 && chartData.current[lastIdx].x === bucketStartMs) {
          // Update the current active candle
          const lastPoint = chartData.current[lastIdx];
          lastPoint.close = Number(priceData.price);
          lastPoint.high = Math.max(Number(lastPoint.high), Number(priceData.price));
          lastPoint.low = Math.min(Number(lastPoint.low), Number(priceData.price));
          lastPoint.volume = Number(lastPoint.volume) + Number(priceData.volume || 0);
          activePoint = { ...lastPoint };
        } else {
          // Create a new candle starting this period
          const prevClose = lastIdx >= 0 ? chartData.current[lastIdx].close : Number(priceData.price);
          activePoint = {
            x: bucketStartMs,
            open: prevClose,
            high: Math.max(prevClose, Number(priceData.price)),
            low: Math.min(prevClose, Number(priceData.price)),
            close: Number(priceData.price),
            volume: Number(priceData.volume || 0)
          };
          chartData.current.push(activePoint);
        }

        // Limit the size of in-memory dataset to prevent memory bloat over time
        if (chartData.current.length > 1000) {
          chartData.current = chartData.current.slice(-1000);
        }

        // Update the AnyChart data table with the aggregated candle
        if (dataTable.current?.table) {
          const table = dataTable.current.table;
          const objectData = {
            x: activePoint.x,
            open: Number(activePoint.open),
            high: Number(activePoint.high),
            low: Number(activePoint.low),
            close: Number(activePoint.close),
            volume: Number(activePoint.volume || 0)
          };

          table.addData([objectData]);

          // Redraw the chart to show the live candle growing/shrinking
          if (chart.current && typeof chart.current.draw === 'function') {
            chart.current.draw();
          }
        }
      } catch (error) {
        console.error('Error updating live price table:', error);
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
      };
    }
  }, [token, isLiveUpdating, granularity, historicalDataLoaded]);

  // Clean up on unmount
  useEffect(() => {
    isMounted.current = true;

    return () => {
      isMounted.current = false;
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

  const fetchData = useCallback(async () => {
    console.log('fetchData called');
    if (!isMounted.current) {
      console.log('fetchData: Component not mounted, returning');
      return;
    }

    // Skip if no token
    if (!token) {
      console.log('fetchData: No token, skipping');
      return;
    }

    console.log('Starting data fetch...');
    setLoading(true);
    setError(null);

    try {
      console.log('Fetching candlestick data...', { token, startDate, endDate, granularity });
      const data = await getCandlestickData(token, startDate, endDate, granularity);
      console.log('Received data from API:', data ? `Array(${data.length})` : 'null');

      if (!data || !Array.isArray(data) || data.length === 0) {
        console.warn('No data received or empty array from API');
        const errorMsg = "No historical data available. Chart will show live updates only.";
        setError(errorMsg);
        setLoading(false);
        setHistoricalDataLoaded(true);
        setIsLiveUpdating(true);
        return;
      }

      console.log(`Received ${data.length} data points`);

      // Format data for the chart
      const mappedData = data.map(item => ({
        x: new Date(item.timestamp).getTime(),
        open: parseFloat(item.open),
        high: parseFloat(item.high),
        low: parseFloat(item.low),
        close: parseFloat(item.close),
        volume: parseFloat(item.volume) || 0
      }));

      // Store the data for later use
      chartData.current = mappedData;

      // Mark historical data as loaded and enable live updates
      setHistoricalDataLoaded(true);
      setIsLiveUpdating(true);
      console.log('Historical data loaded successfully, live updates enabled');
    } catch (err) {
      console.error('Error in fetchData:', err);
      setError(err.message || "An error occurred while fetching historical data. Live updates will be attempted.");
      setHistoricalDataLoaded(true);
      setIsLiveUpdating(true);
    } finally {
      console.log('Fetch completed, setting loading to false');
      setLoading(false);
    }
  }, [token, startDate, endDate, granularity]);



  const renderChart = (data) => {
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
        // Create stock chart
        const stockChart = anychart.stock();

        // Create data table
        const table = anychart.data.table('x');

        // Ensure data is in the correct format
        const formattedData = data && data.length > 0 ? data.map(item => ({
          x: item.x, // Data is already in milliseconds from fetchData
          open: Number(item.open) || 0,
          high: Number(item.high) || 0,
          low: Number(item.low) || 0,
          close: Number(item.close) || 0,
          volume: Number(item.volume) || 0
        })) : [];

        console.log('Adding data to table:', formattedData.length, 'points');
        if (formattedData.length > 0) {
          table.addData(formattedData);
        }

        // Mapping for candlestick series
        const mapping = table.mapAs({
          x: 'x',
          open: 'open',
          high: 'high',
          low: 'low',
          close: 'close',
          value: 'close'
        });

        // Store both table and mapping for consistency
        dataTable.current = {
          table: table,
          mapping: mapping
        };

        // Create first plot with candlestick series
        const plot = stockChart.plot(0);
        plot.yGrid(true).xGrid(true);
        plot.yMinorGrid(true).xMinorGrid(true);

        // Create candlestick series
        const candlestickSeries = plot.candlestick(mapping);
        candlestickSeries.name(token + ' Price');

        // Customize the appearance of candlesticks
        candlestickSeries.risingStroke('#0f9d58');
        candlestickSeries.risingFill('#0f9d58');
        candlestickSeries.fallingStroke('#db4437');
        candlestickSeries.fallingFill('#db4437');

        // Make candlesticks touch each other by setting point width
        candlestickSeries.pointWidth('90%'); // Use 90% of available space



        // Set chart title
        stockChart.title(`${token} Price Chart${isLiveUpdating ? ' (Live)' : ''}`);

        // Create second plot for volume if available and we have data
        if (formattedData.length > 0 && formattedData.some(item => item.volume > 0)) {
          const volumeMapping = table.mapAs({
            x: 'x',
            value: 'volume'
          });

          const volumePlot = stockChart.plot(1);
          volumePlot.height('30%');
          volumePlot.yAxis().title('Volume');
          // Use a callback function instead of format string to avoid any split issues
          volumePlot.yAxis().labels().format(function () {
            const val = this.value;
            if (val >= 1e9) return (val / 1e9).toFixed(1) + 'B';
            if (val >= 1e6) return (val / 1e6).toFixed(1) + 'M';
            if (val >= 1e3) return (val / 1e3).toFixed(1) + 'K';
            return val;
          });

          const volumeSeries = volumePlot.column(volumeMapping);
          volumeSeries.name('Volume');
          volumeSeries.zIndex(100);

          // Use solid colors instead of ordinalColor scale if it crashed
          volumeSeries.risingFill('#0f9d58');
          volumeSeries.risingStroke('#0f9d58');
          volumeSeries.fallingFill('#db4437');
          volumeSeries.fallingStroke('#db4437');

          // Add scroller
          const scrollerSeries = stockChart.scroller().candlestick(mapping);
          scrollerSeries.pointWidth('90%');
        }

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
    if (token) {
      fetchData();
    }
  }, [token, startDate, endDate, granularity, fetchData]);

  // Render chart once data is loaded and container is mounted
  useEffect(() => {
    if (!loading && chartContainer.current && token && chartData.current.length > 0) {
      console.log('Rendering chart from useEffect...');
      renderChart(chartData.current);
    }
  }, [loading, token]);

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
    }
  };

  if (!token) {
    return <div>Please select a token.</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }


  return (
    <div className="container mx-auto p-4">
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
              className={`px-4 py-2 rounded-md text-white font-medium ${isLiveUpdating ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
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