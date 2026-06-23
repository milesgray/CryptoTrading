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
  const initDataTable = useCallback(() => {
    if (!chartContainer.current) return null;
    
    try {
      // Create a data table with x as the key field
      const table = anychart.data.table('x');
      
      console.log('Data table initialized successfully');
      return { table };
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
    
    const sampleData = [];
    
    // Add 30 days of sample data
    for (let i = 30; i >= 0; i--) {
      const timestamp = now - (i * day);
      const basePrice = 100000 + (Math.random() * 20000);
      const open = basePrice;
      const close = basePrice * (0.99 + (Math.random() * 0.02));
      const high = Math.max(open, close) * (1 + Math.random() * 0.01);
      const low = Math.min(open, close) * (0.99 - Math.random() * 0.01);
      const volume = 100 + (Math.random() * 100);
      
      sampleData.push({
        x: timestamp, // Already in milliseconds
        open: open,
        high: high,
        low: low,
        close: close,
        volume: volume
      });
    }
    
    try {
      dataTable.current.table.addData(sampleData);
      console.log('Sample data added successfully');
    } catch (error) {
      console.error('Error adding sample data:', error);
    }
  }, []);
  
  // Initialize the chart
  const initChart = useCallback(() => {
    console.log('Initializing chart...');
    
    if (!chartContainer.current || !dataTable.current?.table) {
      console.error('Chart container or data table not available');
      return;
    }
    
    try {
      console.log('Creating stock chart...');
      // Create a stock chart
      chart.current = anychart.stock();
      
      // Create mapping for the data
      const mapping = dataTable.current.table.mapAs({
        x: 'x',
        open: 'open',
        high: 'high',
        low: 'low',
        close: 'close',
        value: 'close'
      });
      
      // Store mapping for later use
      dataTable.current.mapping = mapping;
      
      // Create a plot
      const plot = chart.current.plot(0);
      
      console.log('Creating OHLC series with mapping');
      // Create an OHLC series
      const series = plot.ohlc(mapping)
        .name('Price')
        .risingStroke('#3ba158')
        .fallingStroke('#fa1c26');
      
      // Make candlesticks touch each other
      series.pointWidth('90%');
      
      // Enable grid and axis
      plot.yGrid(true).xGrid(true);
      plot.yAxis().title('Price');
      
      console.log('Setting up scroller...');
      // Create scroller
      const scrollerSeries = chart.current.scroller().ohlc(mapping);
      scrollerSeries.pointWidth('90%');
      
      // Add sample data for testing
      addSampleData();
      
      console.log('Setting container and drawing chart...');
      // Set container and draw
      chart.current.container(chartContainer.current);
      chart.current.draw();
      
      console.log('Chart initialization complete');
      
    } catch (error) {
      console.error('Error initializing chart:', error);
    }
  }, [addSampleData]);
  
  // Initialize data table and chart on mount
  useEffect(() => {
    console.log('Component mounted, checking initialization...');
    
    if (!dataTable.current && chartContainer.current) {
      console.log('No data table found, creating new one...');
      const result = initDataTable();
      if (result?.table) {
        console.log('Data table created successfully');
        dataTable.current = result;
      } else {
        console.error('Failed to create data table:', result);
      }
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
  }, [initDataTable]);
  
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
          // Create a simple table for live updates
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
          console.log('Minimal data table created for live updates');
          
          // Initialize chart if not already done
          console.log('Chart initialization check:', {
            hasChart: !!chart.current,
            hasContainer: !!chartContainer.current,
            isInitializing: chartInitializing.current,
            containerDimensions: chartContainer.current ? {
              width: chartContainer.current.offsetWidth,
              height: chartContainer.current.offsetHeight
            } : null,
            shouldInitialize: !chart.current && chartContainer.current && !chartInitializing.current
          });
          
          if (!chart.current && chartContainer.current && !chartInitializing.current) {
            // Check if container has dimensions
            if (chartContainer.current.offsetWidth === 0 || chartContainer.current.offsetHeight === 0) {
              console.log('Container has no dimensions, waiting for next update...');
              return;
            }
            
            console.log('Initializing chart for live updates...');
            chartInitializing.current = true;
            
            try {
              // Create a minimal chart structure using the existing table
              import('anychart').then((anychart) => {
                console.log('AnyChart loaded for live updates...');
                
                try {
                  // Create stock chart
                  const stockChart = anychart.stock();
                  
                  // Use the existing table and mapping
                  const table = dataTable.current.table;
                  const mapping = dataTable.current.mapping;
                  
                  // Create first plot with candlestick series
                  const plot = stockChart.plot(0);
                  plot.yGrid(true).xGrid(true);
                  plot.yMinorGrid(true).xMinorGrid(true);
                  
                  // Create candlestick series (empty for now)
                  const candlestickSeries = plot.candlestick(mapping);
                  candlestickSeries.name(token + ' Price');
                  
                  // Customize the appearance of candlesticks
                  candlestickSeries.risingStroke('#0f9d58');
                  candlestickSeries.risingFill('#0f9d58');
                  candlestickSeries.fallingStroke('#db4437');
                  candlestickSeries.fallingFill('#db4437');
                  
                  // Make candlesticks touch each other
                  candlestickSeries.pointWidth('90%');
                  
                  // Set chart title
                  stockChart.title(`${token} Price Chart (Live)`);
                  
                  // Draw the chart
                  console.log('Drawing minimal chart for live updates...');
                  stockChart.container(chartContainer.current);
                  stockChart.draw();
                  
                  // Save chart reference
                  chart.current = stockChart;
                  chartInitializing.current = false;
                  console.log('Minimal chart created successfully for live updates');
                  
                  // Now that the chart is ready, try to redraw any pending data
                  setTimeout(() => {
                    if (chart.current && typeof chart.current.draw === 'function') {
                      try {
                        chart.current.draw();
                        console.log('Chart redrawn after initialization');
                      } catch (drawError) {
                        console.warn('Error redrawing chart after init:', drawError);
                      }
                    }
                  }, 100);
                  
                } catch (chartError) {
                  console.error('Error creating minimal chart:', chartError);
                  chartInitializing.current = false;
                }
              }).catch(err => {
                console.error('Failed to load AnyChart for live updates:', err);
                chartInitializing.current = false;
              });
            } catch (error) {
              console.error('Error initializing chart for live updates:', error);
              chartInitializing.current = false;
            }
          }
        } catch (error) {
          console.error('Failed to create minimal data table:', error);
          return;
        }
      }

      try {
        // Always add new data points for live updates (don't try to update existing ones)
        const newPoint = {
          x: now.getTime(), // Ensure consistent millisecond timestamp
          open: priceData.price,
          high: priceData.price,
          low: priceData.price,
          close: priceData.price,
          volume: priceData.volume || 0
        };
        
        // Add to our data array
        chartData.current.push(newPoint);
        
        // Keep only last 100 points to prevent memory issues
        if (chartData.current.length > 100) {
          chartData.current = chartData.current.slice(-100);
        }
        
        // Update the data table
        if (dataTable.current?.table) {
          try {
            const table = dataTable.current.table;
            
            console.log('Table type:', typeof table, 'Table exists:', !!table);
            
            // Try object format first (for tables created by renderChart)
            const objectData = {
              x: now.getTime(), // Use consistent millisecond timestamp
              open: Number(newPoint.open),
              high: Number(newPoint.high),
              low: Number(newPoint.low),
              close: Number(newPoint.close),
              volume: Number(newPoint.volume || 0)
            };
            
            console.log('Adding live data with object format:', objectData);
            
            try {
              table.addData([objectData]);
              console.log('Live data added successfully');
              
              // Trigger chart redraw to show new data
              if (chart.current && typeof chart.current.draw === 'function') {
                try {
                  console.log('Attempting to redraw chart...');
                  chart.current.draw();
                  console.log('Chart redrawn with new data');
                } catch (drawError) {
                  console.warn('Error redrawing chart:', drawError);
                  // Try to fit all and redraw again
                  try {
                    chart.current.fitAll();
                    setTimeout(() => {
                      if (chart.current) {
                        chart.current.draw();
                      }
                    }, 100);
                  } catch (fitError) {
                    console.warn('Error with fitAll and redraw:', fitError);
                  }
                }
              } else if (chartInitializing.current) {
                console.log('Chart is still initializing, will redraw when ready...');
                // Chart is initializing, it will be redrawn automatically after initialization
              } else if (!chartContainer.current) {
                console.log('Chart container not available, waiting for mount...');
              } else if (chartContainer.current.offsetWidth === 0 || chartContainer.current.offsetHeight === 0) {
                console.log('Chart container has no dimensions, waiting for layout...');
              } else {
                console.warn('Chart not available for redraw, chart.current:', chart.current, 'chartInitializing:', chartInitializing.current);
                // Try to initialize the chart if it seems like it should exist but doesn't
                console.log('Attempting fallback chart initialization...');
                setTimeout(() => {
                  if (!chart.current && chartContainer.current && !chartInitializing.current) {
                    console.log('Retrying chart initialization...');
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
                        
                        // Make candlesticks touch each other
                        candlestickSeries.pointWidth('90%');
                        
                        stockChart.title(`${token} Price Chart (Live)`);
                        stockChart.container(chartContainer.current);
                        stockChart.draw();
                        
                        chart.current = stockChart;
                        chartInitializing.current = false;
                        console.log('Fallback chart initialization successful');
                        
                        // Redraw to show accumulated data
                        setTimeout(() => {
                          if (chart.current) {
                            chart.current.draw();
                          }
                        }, 100);
                      } catch (error) {
                        console.error('Fallback chart initialization failed:', error);
                        chartInitializing.current = false;
                      }
                    }).catch(err => {
                      console.error('Fallback AnyChart loading failed:', err);
                      chartInitializing.current = false;
                    });
                  }
                }, 500);
              }
            } catch (tableError) {
              console.log('Object format failed for live data, trying array format...', tableError);
              
              // Try array format as fallback
              const rowData = [
                now.getTime(), // Use consistent millisecond timestamp
                Number(newPoint.open),
                Number(newPoint.high),
                Number(newPoint.low),
                Number(newPoint.close),
                Number(newPoint.volume || 0)
              ];
              
              console.log('Trying array format with data:', rowData);
              table.addData(rowData);
              console.log('Live data added with array format');
              
              // Trigger chart redraw to show new data
              if (chart.current && typeof chart.current.draw === 'function') {
                try {
                  console.log('Attempting to redraw chart (array format)...');
                  chart.current.draw();
                  console.log('Chart redrawn with new data (array format)');
                } catch (drawError) {
                  console.warn('Error redrawing chart (array format):', drawError);
                  // Try to fit all and redraw again
                  try {
                    chart.current.fitAll();
                    setTimeout(() => {
                      if (chart.current) {
                        chart.current.draw();
                      }
                    }, 100);
                  } catch (fitError) {
                    console.warn('Error with fitAll and redraw (array format):', fitError);
                  }
                }
              } else if (chartInitializing.current) {
                console.log('Chart is still initializing (array format), will redraw when ready...');
                // Chart is initializing, it will be redrawn automatically after initialization
              } else if (!chartContainer.current) {
                console.log('Chart container not available (array format), waiting for mount...');
              } else if (chartContainer.current.offsetWidth === 0 || chartContainer.current.offsetHeight === 0) {
                console.log('Chart container has no dimensions (array format), waiting for layout...');
              } else {
                console.warn('Chart not available for redraw (array format), chart.current:', chart.current, 'chartInitializing:', chartInitializing.current);
                // Note: Fallback initialization is only needed in the first (object format) section
                // to avoid duplicate attempts
              }
            }
          } catch (error) {
            console.error('Error adding live data to table:', error, {
              newPoint,
              now: now,
              dataTable: dataTable.current
            });
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
  }, [token, isLiveUpdating, granularity, historicalDataLoaded]);

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
        const errorMsg = "No historical data available. Chart will show live updates only.";
        console.warn(errorMsg);
        setError(errorMsg);
        setLoading(false);
        
        // Still enable live updates even without historical data
        setHistoricalDataLoaded(true);
        setIsLiveUpdating(true);
        console.log('No historical data, but enabling live updates');
        return;
      }
      
      console.log(`Received ${data.length} data points`);
      
      // Format data for the chart
      const mappedData = data.map(item => ({
        x: new Date(item.timestamp).getTime(), // Convert timestamp to milliseconds for consistent chart display
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
      
      // Mark historical data as loaded and enable live updates
      setHistoricalDataLoaded(true);
      setIsLiveUpdating(true);
      console.log('Historical data loaded successfully, live updates enabled');
    } catch (err) {
      console.error('Error in fetchData:', err);
      setError(err.message || "An error occurred while fetching historical data. Live updates will be attempted.");
      
      // Enable live updates even if historical data fetch failed
      setHistoricalDataLoaded(true);
      setIsLiveUpdating(true);
      console.log('Historical data fetch failed, but enabling live updates as fallback');
    } finally {
      console.log('Fetch completed, setting loading to false');
      setLoading(false);
    }
  };

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

  // Set up the initial data load and chart initialization
  useEffect(() => {
    // Skip if component is not mounted or no token or data table not ready
    if (!isMounted.current || !token || !dataTable.current?.table || !chartContainer.current) {
      console.log('Initialization conditions not met:', {
        isMounted: isMounted.current,
        hasToken: !!token,
        hasDataTable: !!dataTable.current?.table,
        hasContainer: !!chartContainer.current
      });
      return;
    }
    
    // Initialize chart if not already done
    if (!chart.current) {
      console.log('Initializing chart...');
      initChart();
    }
    
    // Load data
    const loadData = async () => {
      try {
        await fetchData();
      } catch (error) {
        console.error('Error loading chart data:', error);
        setError(`Failed to load chart data: ${error.message}`);
      }
    };
    
    loadData();
    
  }, [token, initChart, fetchData]);

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