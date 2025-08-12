from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__)
CORS(app)

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get historical stock price data"""
    try:
        symbol = request.args.get('symbol', 'AAPL')
        period = request.args.get('period', '1y')
        
        # Add .NS suffix for Indian stocks if not present
        if not symbol.endswith('.NS') and not '.' in symbol:
            symbol = symbol + '.NS'
        
        data = get_stock_data(symbol, period)
        
        # Convert to JSON-serializable format
        history_data = {
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'open': data['Open'].tolist(),
            'high': data['High'].tolist(),
            'low': data['Low'].tolist(),
            'close': data['Close'].tolist(),
            'volume': data['Volume'].tolist()
        }
        
        return jsonify(history_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the LSTM model based on given parameters"""
    try:
        request_data = request.json
        symbol = request_data.get('symbol', 'AAPL')
        epochs = request_data.get('epochs', 50)
        start_date = request_data.get('start_date')
        end_date = request_data.get('end_date')
        
        # Add .NS suffix for Indian stocks if not present
        if not symbol.endswith('.NS') and not '.' in symbol:
            symbol = symbol + '.NS'
        
        # Get data based on date range or period
        if start_date and end_date:
            data = yf.download(tickers=symbol, start=start_date, end=end_date, interval='1d')
        else:
            data = get_stock_data(symbol, '1y')
        
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        data['Volume'] = np.log1p(data['Volume'])
        
        # Train models for each feature
        models = {}
        scalers = {}
        features = data.columns.tolist()
        
        for feature in features:
            X, y, scaler = prepare_data(data[[feature]], prediction_horizon=30)
            model = train_lstm_model(X, y, output_size=30, epochs=epochs)
            models[feature] = model
            scalers[feature] = scaler
        
        # Save models and scalers
        model_filename = f'model/{symbol.replace(".", "_")}_models.pkl'
        scaler_filename = f'model/{symbol.replace(".", "_")}_scalers.pkl'
        
        with open(model_filename, 'wb') as f:
            pickle.dump(models, f)
        
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scalers, f)
        
        # Train the lightweight NeuralNetwork aggregator to predict next-day Close
        try:
            nn_model = NeuralNetwork(input_size=len(features), hidden_size1=64, hidden_size2=32, output_size=1)

            # Build NN training dataset: inputs are scaled feature values at time t, target is scaled Close at time t+1
            close_scaler = scalers['Close']
            # Scale each feature column using its scaler
            scaled_columns = {}
            for feat in features:
                values = data[[feat]].values
                # Note: 'Volume' is already log1p-transformed above; scalers were fit on these values
                scaled_columns[feat] = scalers[feat].fit_transform(values).flatten()

            # Construct X (t) and y (t+1) in scaled space
            min_len = min(len(scaled_columns[f]) for f in features)
            X_nn = []
            y_nn = []
            for t in range(min_len - 1):
                X_nn.append([scaled_columns[f][t] for f in features])
                y_nn.append(close_scaler.transform([[data['Close'].values[t + 1]]])[0, 0])

            if len(X_nn) >= 16:  # require some minimal samples
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                X_tensor = torch.tensor(np.array(X_nn), dtype=torch.float32).to(device)
                y_tensor = torch.tensor(np.array(y_nn), dtype=torch.float32).unsqueeze(1).to(device)

                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
                nn_model.to(device)
                nn_epochs = max(10, int(epochs))  # tie to epochs param
                nn_model.train()
                for _ in range(nn_epochs):
                    optimizer.zero_grad()
                    out = nn_model(X_tensor)
                    loss = criterion(out, y_tensor)
                    loss.backward()
                    optimizer.step()

                # Save the trained NN model
                nn_path = f"model/{symbol.replace('.', '_')}_nn.pt"
                torch.save(nn_model.state_dict(), nn_path)
            else:
                # Not enough samples; skip training NN
                pass
        except Exception:
            # NN training is optional; ignore errors to not block LSTM training API
            pass

        return jsonify({
            'message': f'Models trained successfully for {symbol}',
            'symbol': symbol,
            'epochs': epochs,
            'features_trained': features
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Get predictions for the given stock symbol"""
    try:
        request_data = request.json
        symbol = request_data.get('symbol', 'AAPL')
        days = request_data.get('days', 30)
        
        # Add .NS suffix for Indian stocks if not present
        if not symbol.endswith('.NS') and not '.' in symbol:
            symbol = symbol + '.NS'
        
        # Check if models exist
        model_filename = f'model/{symbol.replace(".", "_")}_models.pkl'
        scaler_filename = f'model/{symbol.replace(".", "_")}_scalers.pkl'
        
        if not os.path.exists(model_filename):
            return jsonify({'error': f'No trained models found for {symbol}. Please train the model first.'}), 404
        
        # Load models and scalers
        with open(model_filename, 'rb') as f:
            models = pickle.load(f)
        
        with open(scaler_filename, 'rb') as f:
            scalers = pickle.load(f)
        
        # Get recent data for prediction
        data = get_stock_data(symbol, '3mo')
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        data['Volume'] = np.log1p(data['Volume'])
        
        # Make predictions for each feature
        predictions = {}
        features = data.columns.tolist()
        
        for feature in features:
            if feature in models and feature in scalers:
                X, _, _ = prepare_data(data[[feature]], prediction_horizon=days)
                feature_predictions = sliding_window_prediction(
                    models[feature], X[-1:], days, scalers[feature]
                )
                predictions[feature] = feature_predictions
        
        # Optionally produce consolidated 'final' Close prediction using trained NeuralNetwork
        try:
            nn_path = f"model/{symbol.replace('.', '_')}_nn.pt"
            if os.path.exists(nn_path):
                # Prepare NN inputs per day using scaled predicted features
                nn_model = NeuralNetwork(input_size=len(features), hidden_size1=64, hidden_size2=32, output_size=1)
                nn_model.load_state_dict(torch.load(nn_path, map_location='cpu'))
                nn_model.eval()

                # Build matrix of size (days, features) in scaled space
                X_days = []
                for i in range(days):
                    feature_vector = []
                    for feat in features:
                        if feat in predictions:
                            val = predictions[feat][i]
                        else:
                            # If a feature prediction is missing, backfill with last known value
                            val = float(data[feat].iloc[-1])
                        # Scale using the corresponding scaler; ensure 2D input
                        scaler = scalers[feat]
                        scaled_val = scaler.transform(np.array(val, ndmin=2).reshape(-1, 1))[0, 0]
                        feature_vector.append(scaled_val)
                    X_days.append(feature_vector)

                X_tensor = torch.tensor(np.array(X_days), dtype=torch.float32)
                with torch.no_grad():
                    y_scaled = nn_model(X_tensor).numpy().reshape(-1, 1)

                # Inverse scale using Close scaler
                close_scaler = scalers['Close']
                final_close = close_scaler.inverse_transform(y_scaled).flatten().tolist()
                predictions['final'] = final_close
        except Exception:
            # If NN inference fails, fall back silently
            pass

        # Generate future dates
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        future_dates = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        return jsonify({
            'symbol': symbol,
            'predictions': predictions,
            'dates': future_dates,
            'last_actual_date': last_date.strftime('%Y-%m-%d')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics"""
    try:
        symbol = request.args.get('symbol', 'AAPL')
        
        # Add .NS suffix for Indian stocks if not present
        if not symbol.endswith('.NS') and not '.' in symbol:
            symbol = symbol + '.NS'
        
        # This would typically load saved metrics from training
        # For now, return placeholder metrics
        metrics = {
            'symbol': symbol,
            'rmse': 0.0,
            'mae': 0.0,
            'accuracy': 0.0,
            'training_loss': 0.0,
            'validation_loss': 0.0
        }
        
        return jsonify(metrics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper functions
def get_stock_data(ticker, period='max'):
    """Fetch stock data from Yahoo Finance"""
    data = yf.download(tickers=ticker, period=period, interval='1d')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    if data.empty:
        raise ValueError("No data found for the specified ticker.")
    return data

def prepare_data(data, prediction_horizon, sequence_length=75):
    """Prepare data for LSTM training"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - prediction_horizon + 1):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i:i+prediction_horizon, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.expand_dims(X, axis=2)
    return X, y, scaler

def train_lstm_model(X, y, input_size=1, hidden_size=100, output_size=1, epochs=50, lr=0.01, batch_size=32, bidirectional=True):
    """Train LSTM model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, bidirectional=bidirectional).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return model

def sliding_window_prediction(model, X_input, days, scaler):
    """Make iterative predictions using sliding window"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_input = torch.tensor(X_input, dtype=torch.float32).to(device)
    predictions = []

    for _ in range(days):
        with torch.no_grad():
            pred = model(X_input)
            predictions.append(pred.cpu().numpy().flatten()[0])
            pred_scaled = pred.unsqueeze(-1)
            X_input = torch.cat((X_input[:, 1:, :], pred_scaled), dim=1)

    predictions = np.array(predictions)
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten().tolist()

# Model definitions
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_size1=64, hidden_size2=32, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight); nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=150, output_size=1, bidirectional=True, dropout=0.1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
