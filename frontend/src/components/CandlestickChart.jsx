import React, { useEffect, useState, useRef, useCallback } from 'react';
import { getCandlestickData, getLatestPrice } from '../services/api';
import { format, subDays } from 'date-fns';
import _ from 'lodash';

const CandlestickChart = ({ token }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [startDate, setStartDate] = useState(subDays(new Date(), 7)); // Default: 7 days ago
  const [endDate, setEndDate] = useState(new Date());
  const [granularity, setGranularity] = useState(3600); // Default: 1 hour
  const [isLiveUpdating, setIsLiveUpdating] = useState(true); // Default: live updates enabled
  const chartContainer = useRef(null);
  const chart = useRef(null);
  const dataTable = useRef(null);
  const updateInterval = useRef(null);
  const chartData = useRef([]);
  
  // Update frequency in milliseconds (5 seconds by default)
  const UPDATE_FREQUENCY = 5000;

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await getCandlestickData(token, startDate, endDate, granularity);
      
      if (data.length === 0) {
        setError("No data available for the selected time range.");
        setLoading(false);
        return;
      }
      
      // Format data for AnyStock
      const mappedData = data.map(item => ({
        x: item.timestamp, // AnyStock can parse this timestamp
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
        volume: item.volume || 0
      }));

      // Store the data for later use
      chartData.current = mappedData;
      
      renderChart(mappedData);
    } catch (err) {
      setError(err.message || "An error occurred while fetching data.");
    } finally {
      setLoading(false);
    }
  };

  const renderChart = (data) => {
    // Load the required modules before creating the chart
    if (chartContainer.current) {
      // Clean up previous chart if it exists
      if (chart.current) {
        chart.current.dispose();
        chart.current = null;
      }

      // Load AnyStock dynamically
      import('anychart').then((anychart) => {
        // Set theme
        anychart.theme('lightBlue');
        
        // Create stock chart
        const stockChart = anychart.stock();
        
        // Create data table
        const table = anychart.data.table('x');
        table.addData(data);
        dataTable.current = table;
        
        // Mapping for candlestick series
        const mapping = table.mapAs({
          open: 'open',
          high: 'high',
          low: 'low',
          close: 'close'
        });
        
        // Mapping for volume series (if available)
        const volumeMapping = table.mapAs({
          value: 'volume'
        });
        
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
        
        // Set chart title
        stockChart.title(`${token} Price Chart${isLiveUpdating ? ' (Live)' : ''}`);
        
        // Create second plot for volume if available
        if (data[0].volume) {
          const volumePlot = stockChart.plot(1);
          volumePlot.height('30%');
          volumePlot.yAxis().labels().format('{%Value}{scale:(1)(K)(M)(B)}');
          
          const volumeSeries = volumePlot.column(volumeMapping);
          volumeSeries.name('Volume');
          volumeSeries.zIndex(100);
          
          // Color volume based on price change
          volumeSeries.colorScale(anychart.scales.ordinalColor([
            { less: 0, color: '#db4437' },
            { from: 0, color: '#0f9d58' }
          ]));
          volumeSeries.colorScale().ranges([
            { from: 0, to: 0, name: 'Up', color: '#0f9d58' },
            { from: 0, to: 0, name: 'Down', color: '#db4437' }
          ]);
        }
        
        // Add scroller
        stockChart.scroller().candlestick(mapping);
        
        // Set container and draw the chart
        stockChart.container(chartContainer.current);
        stockChart.draw();
        
        // Save chart reference for cleanup
        chart.current = stockChart;
      }).catch(err => {
        setError(`Failed to load AnyStock: ${err.message}`);
      });
    }
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
    if (token) {
      fetchData();
    }
    
    // Cleanup
    return () => {
      if (chart.current) {
        chart.current.dispose();
      }
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
      }
    };
  }, [token, startDate, endDate, granularity]);

  // Set up the live update interval
  useEffect(() => {
    if (isLiveUpdating && token) {
      // Clear any existing interval
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
      }
      
      // Create a new interval
      updateInterval.current = setInterval(updateLatestPrice, UPDATE_FREQUENCY);
      
      // Start with an immediate update
      updateLatestPrice();
    } else if (updateInterval.current) {
      clearInterval(updateInterval.current);
      updateInterval.current = null;
      
      // Update the chart title to remove live indication
      if (chart.current) {
        chart.current.title(`${token} Price Chart`);
      }
    }
    
    return () => {
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
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
    setIsLiveUpdating(prev => !prev);
  };

  if (!token) {
    return <div>Please select a token.</div>;
  }

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="container mx-auto p-4">
      <h2 className="text-2xl font-bold mb-4">Candlestick Chart for {token}</h2>
      <div className="flex flex-wrap space-x-2 space-y-2 md:space-y-0 mb-4">
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
      <div ref={chartContainer} className="w-full h-96"></div>
    </div>
  );
};

export default CandlestickChart;