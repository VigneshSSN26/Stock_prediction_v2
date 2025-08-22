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
      <h3>📊 Model Performance Metrics - {symbol}</h3>
      <div className="metrics-grid">
        <div className="metric-card">
          <h4>📏 RMSE</h4>
          <p>{formatMetric(metrics.rmse)}</p>
          <p className="description">Root Mean Square Error</p>
        </div>
        
        <div className="metric-card">
          <h4>📐 MAE</h4>
          <p>{formatMetric(metrics.mae)}</p>
          <p className="description">Mean Absolute Error</p>
        </div>
        
        <div className="metric-card">
          <h4>🎯 Accuracy</h4>
          <p>{formatMetric(metrics.accuracy)}%</p>
          <p className="description">Model Accuracy</p>
        </div>
        
        <div className="metric-card">
          <h4>🔥 Training Loss</h4>
          <p>{formatMetric(metrics.training_loss)}</p>
          <p className="description">Final Training Loss</p>
        </div>
        
        <div className="metric-card">
          <h4>⚡ Validation Loss</h4>
          <p>{formatMetric(metrics.validation_loss)}</p>
          <p className="description">Final Validation Loss</p>
        </div>
      </div>
    </div>
  );
};

export default MetricsDisplay;
