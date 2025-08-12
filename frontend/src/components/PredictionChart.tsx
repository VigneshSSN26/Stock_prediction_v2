import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface PredictionChartProps {
  historicalData: {
    dates: string[];
    close: number[];
  };
  predictions: {
    dates: string[];
    close: number[];
  };
  symbol: string;
}

const PredictionChart: React.FC<PredictionChartProps> = ({
  historicalData,
  predictions,
  symbol
}) => {
  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
                 labels: {
           usePointStyle: true,
           padding: 20,
           font: {
             family: 'Inter',
             size: 12,
             weight: 600
           }
         }
      },
               title: {
           display: true,
           text: `📈 ${symbol} Stock Price Prediction`,
           font: {
             family: 'Inter',
             size: 18,
             weight: 700
           },
           color: '#2d3748'
         },
    },
    scales: {
      x: {
                 title: {
           display: true,
           text: 'Date',
           font: {
             family: 'Inter',
             size: 12,
             weight: 600
           },
           color: '#4a5568'
         },
        grid: {
          color: 'rgba(226, 232, 240, 0.5)',
          drawBorder: false
        },
        ticks: {
          font: {
            family: 'Inter',
            size: 10
          },
          color: '#718096'
        }
      },
      y: {
                 title: {
           display: true,
           text: 'Price ($)',
           font: {
             family: 'Inter',
             size: 12,
             weight: 600
           },
           color: '#4a5568'
         },
        grid: {
          color: 'rgba(226, 232, 240, 0.5)',
          drawBorder: false
        },
        ticks: {
          font: {
            family: 'Inter',
            size: 10
          },
          color: '#718096'
        }
      },
    },
  };

  const data = {
    labels: [...historicalData.dates, ...predictions.dates],
    datasets: [
      {
        label: '📊 Historical Prices',
        data: historicalData.close,
        borderColor: '#4ecdc4',
        backgroundColor: 'rgba(78, 205, 196, 0.1)',
        borderWidth: 3,
        pointRadius: 0,
        pointHoverRadius: 6,
        pointHoverBackgroundColor: '#4ecdc4',
        pointHoverBorderColor: '#ffffff',
        pointHoverBorderWidth: 2,
        tension: 0.4,
        fill: true
      },
      {
        label: '🔮 Predicted Prices',
        data: [...Array(historicalData.close.length).fill(null), ...predictions.close],
        borderColor: '#667eea',
        backgroundColor: 'rgba(102, 126, 234, 0.1)',
        borderWidth: 3,
        borderDash: [8, 4],
        pointRadius: 0,
        pointHoverRadius: 6,
        pointHoverBackgroundColor: '#667eea',
        pointHoverBorderColor: '#ffffff',
        pointHoverBorderWidth: 2,
        tension: 0.4,
        fill: true
      },
    ],
  };

  return (
    <div className="prediction-chart">
      <h3>📈 Price Prediction Chart</h3>
      <div className="chart-container">
        <Line options={options} data={data} />
      </div>
    </div>
  );
};

export default PredictionChart;
