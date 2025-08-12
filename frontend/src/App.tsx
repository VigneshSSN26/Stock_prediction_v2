import React, { useState, useEffect } from 'react';
import axios from 'axios';
import StockSelector from './components/StockSelector';
import TrainingControls, { TrainingParams } from './components/TrainingControls';
import PredictionChart from './components/PredictionChart';
import MetricsDisplay from './components/MetricsDisplay';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

interface HistoricalData {
  dates: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

interface PredictionData {
  symbol: string;
  predictions: {
    [key: string]: number[];
  };
  dates: string[];
  last_actual_date: string;
}

interface MetricsData {
  symbol: string;
  rmse: number;
  mae: number;
  accuracy: number;
  training_loss: number;
  validation_loss: number;
}

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [historicalData, setHistoricalData] = useState<HistoricalData | null>(null);
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch historical data when symbol changes
  useEffect(() => {
    if (selectedSymbol) {
      fetchHistoricalData();
    }
  }, [selectedSymbol]);

  const fetchHistoricalData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await axios.get(`${API_BASE_URL}/api/history?symbol=${selectedSymbol}`);
      setHistoricalData(response.data);
    } catch (err) {
      setError('Failed to fetch historical data');
      console.error('Error fetching historical data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrain = async (params: TrainingParams) => {
    try {
      setIsTraining(true);
      setError(null);
      
      const response = await axios.post(`${API_BASE_URL}/api/train`, params);
      console.log('Training completed:', response.data);
      
      // Fetch metrics after training
      await fetchMetrics(params.symbol);
      
      // Fetch predictions after training
      await fetchPredictions(params.symbol);
      
    } catch (err) {
      setError('Failed to train model');
      console.error('Error training model:', err);
    } finally {
      setIsTraining(false);
    }
  };

  const fetchPredictions = async (symbol: string) => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await axios.post(`${API_BASE_URL}/api/predict`, {
        symbol: symbol,
        days: 30
      });
      setPredictionData(response.data);
    } catch (err) {
      setError('Failed to fetch predictions');
      console.error('Error fetching predictions:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchMetrics = async (symbol: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/metrics?symbol=${symbol}`);
      setMetricsData(response.data);
    } catch (err) {
      console.error('Error fetching metrics:', err);
    }
  };

  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol);
    setPredictionData(null);
    setMetricsData(null);
  };

  return (
    <div className="App">
      <div className="container">
        <header>
          <h1>Stock Price Prediction</h1>
          <p>AI-powered stock price forecasting using LSTM models</p>
        </header>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        <div className="grid">
          <div>
            <StockSelector
              onSymbolChange={handleSymbolChange}
              selectedSymbol={selectedSymbol}
            />
          </div>
          
          <div>
            <TrainingControls
              onTrain={handleTrain}
              isTraining={isTraining}
            />
          </div>
        </div>

        {isLoading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Loading...</p>
          </div>
        )}

        {historicalData && predictionData && (
          <div>
            <PredictionChart
              historicalData={{
                dates: historicalData.dates,
                close: historicalData.close
              }}
              predictions={{
                dates: predictionData.dates,
                close: predictionData.predictions.Close || []
              }}
              symbol={selectedSymbol}
            />
          </div>
        )}

        {metricsData && (
          <div>
            <MetricsDisplay
              metrics={metricsData}
              symbol={selectedSymbol}
            />
          </div>
        )}

        {historicalData && !predictionData && (
          <div className="stock-selector">
            <p>
              Train a model for {selectedSymbol} to see predictions
            </p>
            <button onClick={() => fetchPredictions(selectedSymbol)}>
              Get Predictions
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
