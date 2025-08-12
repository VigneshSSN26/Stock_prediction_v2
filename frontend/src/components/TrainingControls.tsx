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
    symbol: 'AAPL',
    epochs: 50,
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
            placeholder="e.g., AAPL"
            required
          />
        </div>

        <div>
          <label htmlFor="epochs">
            Epochs
          </label>
          <input
            type="number"
            id="epochs"
            value={params.epochs}
            onChange={(e) => handleInputChange('epochs', parseInt(e.target.value))}
            min="1"
            max="200"
            required
          />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <label htmlFor="startDate">
              Start Date
            </label>
            <input
              type="date"
              id="startDate"
              value={params.startDate}
              onChange={(e) => handleInputChange('startDate', e.target.value)}
            />
          </div>

          <div>
            <label htmlFor="endDate">
              End Date
            </label>
            <input
              type="date"
              id="endDate"
              value={params.endDate}
              onChange={(e) => handleInputChange('endDate', e.target.value)}
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={isTraining}
          style={{ width: '100%', marginTop: '1rem' }}
        >
          {isTraining ? 'Training...' : 'Train Model'}
        </button>
      </form>
    </div>
  );
};

export default TrainingControls;
