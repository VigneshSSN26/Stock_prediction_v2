import React, { useState } from 'react';

interface TrainingControlsProps {
  onTrain: (params: TrainingParams) => void;
  isTraining: boolean;
}

export interface TrainingParams {
  symbol: string;
  epochs: number;
  startDate: string;
  endDate: string;
}

const TrainingControls: React.FC<TrainingControlsProps> = ({ onTrain, isTraining }) => {
  const [params, setParams] = useState<TrainingParams>({
    symbol: 'RELIANCE.NS',
    epochs: 20,
    startDate: '',
    endDate: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onTrain(params);
  };

  const handleInputChange = (field: keyof TrainingParams, value: string | number) => {
    setParams(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <div className="training-controls">
      <h3>Model Training</h3>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="symbol">
            Stock Symbol
          </label>
          <input
            type="text"
            id="symbol"
            value={params.symbol}
            onChange={(e) => handleInputChange('symbol', e.target.value)}
            placeholder="e.g., RELIANCE.NS, AAPL, GOOGL"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="epochs">
            Training Epochs
          </label>
          <input
            type="number"
            id="epochs"
            value={params.epochs}
            onChange={(e) => handleInputChange('epochs', parseInt(e.target.value))}
            min="5"
            max="100"
            required
          />
          <small style={{ color: '#718096', fontSize: '0.875rem', display: 'block', marginTop: '0.5rem' }}>
            💡 Higher epochs = better accuracy but longer training time
          </small>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div className="form-group">
            <label htmlFor="startDate">
              Start Date (Optional)
            </label>
            <input
              type="date"
              id="startDate"
              value={params.startDate}
              onChange={(e) => handleInputChange('startDate', e.target.value)}
            />
            <small style={{ color: '#718096', fontSize: '0.875rem', display: 'block', marginTop: '0.5rem' }}>
               Leave empty for last 1 year
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="endDate">
              End Date (Optional)
            </label>
            <input
              type="date"
              id="endDate"
              value={params.endDate}
              onChange={(e) => handleInputChange('endDate', e.target.value)}
            />
            <small style={{ color: '#718096', fontSize: '0.875rem', display: 'block', marginTop: '0.5rem' }}>
               Leave empty for current date
            </small>
          </div>
        </div>

        <button
          type="submit"
          disabled={isTraining}
          style={{ width: '100%', marginTop: '1.5rem' }}
        >
          {isTraining ? ' Training Model...' : ' Start Training'}
        </button>
        
        {isTraining && (
          <div style={{ 
            marginTop: '1.5rem', 
            textAlign: 'center', 
            color: '#4a5568',
            background: 'rgba(78, 205, 196, 0.1)',
            padding: '1rem',
            borderRadius: '12px',
            border: '1px solid rgba(78, 205, 196, 0.2)'
          }}>
            <p style={{ fontWeight: '600', marginBottom: '0.5rem' }}>🚀 Training job started!</p>
            <p style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>
              Training LSTM models for each feature (Open, High, Low, Close, Volume)
            </p>
            <p style={{ fontSize: '0.75rem', color: '#718096' }}>
              The training runs in the background. You can continue using the app while it trains.
            </p>
          </div>
        )}
      </form>
    </div>
  );
};

export default TrainingControls;
