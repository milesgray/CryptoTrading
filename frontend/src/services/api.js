import axios from 'axios';
import { formatISO } from 'date-fns';

const api = axios.create({
  baseURL: '/api', // Use the proxy we configured in vite.config.js
});

export const getCandlestickData = async (token, start, end, granularity) => {
  try {
    const formattedStart = formatISO(start);  // Format dates for the API
    const formattedEnd = formatISO(end);
    const response = await api.get(`/candlestick/${token}`, {
      params: {
        start: formattedStart,
        end: formattedEnd,
        granularity,
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching candlestick data:", error); // Log the error
    throw error;  // Re-throw so the component can handle it
  }
};

export const getLatestPrice = async (token) => {
  try {
      const response = await api.get(`/latest_price/${token}`);
      return response.data;
  } catch(error) {
      console.error("Error fetching latest price:", error);
      throw error;
  }
}