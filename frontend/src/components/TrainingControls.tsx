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
      <h3>Training Parameters</h3>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="symbol">
            Stock Symbol
          </label>
          <input
            type="text"
            id="symbol"
            value={params.symbol}
            onChange={(e) => handleInputChange('symbol', e.target.value)}
            placeholder="e.g., RELIANCE.NS, AAPL"
            required
          />
        </div>

        <div>
          <label htmlFor="epochs">
            Epochs (Training Iterations)
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
          <small style={{ color: '#6b7280', fontSize: '0.875rem' }}>
            Higher epochs = better accuracy but longer training time
          </small>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <label htmlFor="startDate">
              Start Date (Optional)
            </label>
            <input
              type="date"
              id="startDate"
              value={params.startDate}
              onChange={(e) => handleInputChange('startDate', e.target.value)}
            />
            <small style={{ color: '#6b7280', fontSize: '0.875rem' }}>
              Leave empty for last 1 year
            </small>
          </div>

          <div>
            <label htmlFor="endDate">
              End Date (Optional)
            </label>
            <input
              type="date"
              id="endDate"
              value={params.endDate}
              onChange={(e) => handleInputChange('endDate', e.target.value)}
            />
            <small style={{ color: '#6b7280', fontSize: '0.875rem' }}>
              Leave empty for current date
            </small>
          </div>
        </div>

        <button
          type="submit"
          disabled={isTraining}
          style={{ width: '100%', marginTop: '1rem' }}
        >
          {isTraining ? 'Training Model...' : 'Train Model'}
        </button>
        
        {isTraining && (
          <div style={{ marginTop: '1rem', textAlign: 'center', color: '#6b7280' }}>
            <p>⏳ Training in progress... This may take a few minutes.</p>
            <p>Training LSTM models for each feature (Open, High, Low, Close, Volume)</p>
          </div>
        )}
      </form>
    </div>
  );
};

export default TrainingControls;
