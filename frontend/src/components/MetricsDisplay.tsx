import React from 'react';

interface MetricsDisplayProps {
  metrics: {
    rmse: number;
    mae: number;
    accuracy: number;
    training_loss: number;
    validation_loss: number;
  };
  symbol: string;
}

const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ metrics, symbol }) => {
  const formatMetric = (value: number) => {
    return typeof value === 'number' ? value.toFixed(4) : 'N/A';
  };

  return (
    <div className="metrics-display">
      <h3>Model Performance Metrics - {symbol}</h3>
      <div className="metrics-grid">
        <div className="metric-card" style={{ backgroundColor: '#eff6ff' }}>
          <h4 style={{ color: '#1e40af' }}>RMSE</h4>
          <p style={{ color: '#2563eb', fontSize: '1.5rem', fontWeight: 'bold' }}>{formatMetric(metrics.rmse)}</p>
          <p className="description" style={{ color: '#2563eb' }}>Root Mean Square Error</p>
        </div>
        
        <div className="metric-card" style={{ backgroundColor: '#f0fdf4' }}>
          <h4 style={{ color: '#166534' }}>MAE</h4>
          <p style={{ color: '#16a34a', fontSize: '1.5rem', fontWeight: 'bold' }}>{formatMetric(metrics.mae)}</p>
          <p className="description" style={{ color: '#16a34a' }}>Mean Absolute Error</p>
        </div>
        
        <div className="metric-card" style={{ backgroundColor: '#faf5ff' }}>
          <h4 style={{ color: '#6b21a8' }}>Accuracy</h4>
          <p style={{ color: '#9333ea', fontSize: '1.5rem', fontWeight: 'bold' }}>{formatMetric(metrics.accuracy)}%</p>
          <p className="description" style={{ color: '#9333ea' }}>Model Accuracy</p>
        </div>
        
        <div className="metric-card" style={{ backgroundColor: '#fff7ed' }}>
          <h4 style={{ color: '#c2410c' }}>Training Loss</h4>
          <p style={{ color: '#ea580c', fontSize: '1.5rem', fontWeight: 'bold' }}>{formatMetric(metrics.training_loss)}</p>
          <p className="description" style={{ color: '#ea580c' }}>Final Training Loss</p>
        </div>
        
        <div className="metric-card" style={{ backgroundColor: '#fef2f2' }}>
          <h4 style={{ color: '#991b1b' }}>Validation Loss</h4>
          <p style={{ color: '#dc2626', fontSize: '1.5rem', fontWeight: 'bold' }}>{formatMetric(metrics.validation_loss)}</p>
          <p className="description" style={{ color: '#dc2626' }}>Final Validation Loss</p>
        </div>
      </div>
    </div>
  );
};

export default MetricsDisplay;
