import React, { useState, useEffect } from 'react';
import axios from 'axios';
import StockSelector from './components/StockSelector';
import TrainingControls, { TrainingParams } from './components/TrainingControls';
import PredictionChart from './components/PredictionChart';
import MetricsDisplay from './components/MetricsDisplay';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:7860';

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
}

interface MetricsData {
  message: string;
}

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('RELIANCE.NS');
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
    } catch (err: any) {
      setError(`Failed to fetch historical data: ${err.response?.data?.error || err.message}`);
      console.error('Error fetching historical data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrain = async (params: TrainingParams) => {
    try {
      setIsTraining(true);
      setError(null);
      
      const response = await axios.post(`${API_BASE_URL}/api/train`, {
        symbol: params.symbol,
        epochs: params.epochs,
        start_date: params.startDate,
        end_date: params.endDate
      });
      console.log('Training completed:', response.data);
      
      // Fetch metrics after training
      await fetchMetrics(params.symbol);
      
      // Fetch predictions after training
      await fetchPredictions(params.symbol);
      
    } catch (err: any) {
      setError(`Failed to train model: ${err.response?.data?.error || err.message}`);
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
    } catch (err: any) {
      setError(`Failed to fetch predictions: ${err.response?.data?.error || err.message}`);
      console.error('Error fetching predictions:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchMetrics = async (symbol: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/metrics?symbol=${symbol}`);
      setMetricsData(response.data);
    } catch (err: any) {
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
                close: predictionData.predictions.final || predictionData.predictions.Close || []
              }}
              symbol={selectedSymbol}
            />
          </div>
        )}

        {metricsData && (
          <div>
            <MetricsDisplay
              metrics={{
                rmse: 0.0,
                mae: 0.0,
                accuracy: 0.0,
                training_loss: 0.0,
                validation_loss: 0.0
              }}
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

        {predictionData && (
          <div className="predictions-summary">
            <h3>Prediction Summary for {selectedSymbol}</h3>
            <div className="prediction-features">
              {Object.keys(predictionData.predictions).map(feature => (
                <div key={feature} className="feature-prediction">
                  <h4>{feature}</h4>
                  <p>Next 30 days: {predictionData.predictions[feature].slice(0, 5).map(v => v.toFixed(2)).join(', ')}...</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
