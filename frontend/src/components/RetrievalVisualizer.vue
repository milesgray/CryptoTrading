<template>
  <div>
    <h2>Retrieval-Augmented Forecasting</h2>
    <div id="chart-container" style="width: 100%; height: 500px;"></div>
    <div>
      <h3>Retrieved Segments</h3>
      <ul>
        <li v-for="(segment, idx) in retrieved" :key="idx">
          Segment {{ segment.id }} (Similarity: {{ segment.similarity?.toFixed(2) || 'N/A' }})
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
import * as echarts from 'echarts';

export default {
  data() {
    return {
      retrieved: [],
      chart: null
    };
  },
  mounted() {
    this.chart = echarts.init(document.getElementById('chart-container'));
    this.fetchData();
    setInterval(this.fetchData, 60000);
  },
  methods: {
    async fetchData() {
      const response = await fetch('/api/retrieval/forecast?symbol=BTC&k=3');
      const data = await response.json();
      this.retrieved = data.retrieved.map(seg => ({
        ...seg,
        similarity: this.calculateSimilarity(seg)
      }));
      this.updateChart(data);
    },
    updateChart(data) {
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
      this.chart.setOption(option);
    },
    calculateSimilarity(segment) {
      // TODO: Implement similarity metric
      return Math.random();
    }
  }
};
</script>

<style scoped>
ul {
  list-style-type: none;
  padding: 0;
}
li {
  margin: 5px 0;
}
</style>