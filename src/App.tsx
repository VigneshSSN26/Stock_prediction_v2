import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import AuthWrapper from './components/Auth/AuthWrapper';
import Header from './components/Dashboard/Header';
import StockSelector from './components/StockSelector';
import TrainingControls, { TrainingParams } from './components/TrainingControls';
import PredictionChart from './components/PredictionChart';
import MetricsDisplay from './components/MetricsDisplay';
import Portfolio from './components/Dashboard/Portfolio';
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

interface JobStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  symbol: string;
  epochs: number;
  created_at: string;
  result?: any;
  error?: string;
}

const Dashboard: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('RELIANCE.NS');
  const [historicalData, setHistoricalData] = useState<HistoricalData | null>(null);
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch historical data when symbol changes
  useEffect(() => {
    if (selectedSymbol) {
      fetchHistoricalData();
    }
  }, [selectedSymbol]);

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

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

  const pollJobStatus = async (jobId: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/job-status/${jobId}`);
      const jobStatus: JobStatus = response.data;
      setCurrentJob(jobStatus);

      if (jobStatus.status === 'completed') {
        // Job completed successfully
        setIsTraining(false);
        setError(null);
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
        
        // Fetch predictions and metrics after successful training
        await fetchPredictions(jobStatus.symbol);
        await fetchMetrics(jobStatus.symbol);
        
        console.log('Training completed:', jobStatus.result);
        
      } else if (jobStatus.status === 'failed') {
        // Job failed
        setIsTraining(false);
        setError(`Training failed: ${jobStatus.error || 'Unknown error'}`);
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
        console.error('Training failed:', jobStatus.error);
      }
      // If status is 'pending' or 'running', continue polling
      
    } catch (err: any) {
      console.error('Error polling job status:', err);
      // Don't stop polling on network errors, just log them
    }
  };

  const handleTrain = async (params: TrainingParams) => {
    try {
      setIsTraining(true);
      setError(null);
      setCurrentJob(null);
      
      // Start the training job
      const response = await axios.post(`${API_BASE_URL}/api/train`, {
        symbol: params.symbol,
        epochs: params.epochs,
        start_date: params.startDate,
        end_date: params.endDate
      });
      
      const { job_id } = response.data;
      console.log('Training job started:', job_id);
      
      // Start polling for job status
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
      
      // Poll every 2 seconds
      pollingIntervalRef.current = setInterval(() => {
        pollJobStatus(job_id);
      }, 2000);
      
      // Initial status check
      await pollJobStatus(job_id);
      
    } catch (err: any) {
      setIsTraining(false);
      setError(`Failed to start training: ${err.response?.data?.error || err.message}`);
      console.error('Error starting training:', err);
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
    <div className="dashboard">
      <Header 
        title="StockAI Predictor" 
        subtitle="Advanced AI-powered stock price forecasting platform"
      />

      <main className="dashboard-main">
        <div className="container">
          {error && (
            <div className="error-banner">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          <div className="dashboard-grid">
            <div className="dashboard-card">
              <StockSelector
                onSymbolChange={handleSymbolChange}
                selectedSymbol={selectedSymbol}
              />
            </div>
            
            <div className="dashboard-card">
              <TrainingControls
                onTrain={handleTrain}
                isTraining={isTraining}
              />
            </div>
          </div>

          {/* Training Progress Display */}
          {isTraining && currentJob && (
            <div className="dashboard-card training-progress">
              <h3>Training Progress</h3>
              <div className="progress-container">
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${currentJob.progress}%` }}
                  ></div>
                </div>
                <p className="progress-text">{currentJob.message}</p>
                <p className="progress-details">
                  Status: {currentJob.status} | Progress: {currentJob.progress}%
                </p>
              </div>
            </div>
          )}

          {isLoading && (
            <div className="loading-overlay">
              <div className="loading-spinner"></div>
              <p>Loading...</p>
            </div>
          )}

          {/* Portfolio Section */}
          <div className="dashboard-card">
            <Portfolio />
          </div>

          {historicalData && predictionData && (
            <div className="dashboard-card">
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
            <div className="dashboard-card">
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
            <div className="dashboard-card">
              <div className="empty-state">
                <div className="empty-icon">📊</div>
                <h3>Ready to Predict</h3>
                <p>Train a model for {selectedSymbol} to see predictions</p>
                <button 
                  className="primary-button"
                  onClick={() => fetchPredictions(selectedSymbol)}
                >
                  Get Predictions
                </button>
              </div>
            </div>
          )}

          {predictionData && (
            <div className="dashboard-card">
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
      </main>
    </div>
  );
};

const App: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  return isAuthenticated ? <Dashboard /> : <AuthWrapper />;
};

const AppWithAuth: React.FC = () => {
  return (
    <AuthProvider>
      <App />
    </AuthProvider>
  );
};

export default AppWithAuth;
