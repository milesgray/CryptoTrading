import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

const RetrievalVisualizer = () => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [retrieved, setRetrieved] = React.useState([]);

  useEffect(() => {
    // Initialize chart
    chartInstance.current = echarts.init(chartRef.current);
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    const response = await fetch('/api/retrieval/forecast?symbol=BTC&k=3');
    const data = await response.json();
    setRetrieved(data.retrieved.map(seg => ({
      ...seg,
      similarity: calculateSimilarity(seg)
    })));
    updateChart(data);
  };

  const updateChart = (data) => {
    const option = {
      title: { text: 'BTC Price + Retrieved Segments' },
      xAxis: { type: 'category', data: Array(60).fill(0).map((_, i) => i) },
      yAxis: { type: 'value' },
      series: [
        {
          name: 'Current Price',
          data: Array(60).fill(0).map((_, i) => [i, Math.random() * 100]), // TODO: Replace with real data
          type: 'line',
          smooth: true
        },
        ...data.retrieved.map(seg => ({
          name: `Segment ${seg.id}`,
          data: Array(60).fill(0).map((_, i) => [i, Math.random() * 100]), // TODO: Replace with real data
          type: 'line',
          lineStyle: { opacity: 0.5 }
        }))
      ]
    };
    chartInstance.current.setOption(option);
  };

  const calculateSimilarity = (segment) => {
    // TODO: Implement similarity metric
    return Math.random();
  };

  return (
    <div>
      <h2>Retrieval-Augmented Forecasting</h2>
      <div ref={chartRef} style={{ width: '100%', height: '500px' }}></div>
      <div>
        <h3>Retrieved Segments</h3>
        <ul>
          {retrieved.map((segment, idx) => (
            <li key={idx}>
              Segment {segment.id} (Similarity: {segment.similarity?.toFixed(2) || 'N/A'})
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default RetrievalVisualizer;