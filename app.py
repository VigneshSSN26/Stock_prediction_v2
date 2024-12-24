from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Only consider the last time step output
        return out

# Load and prepare the stock data
def get_stock_data(ticker, period='6mo'):
    data = yf.download(tickers=ticker, period=period, interval='1d')
    data = data[['Close']].dropna()
    return data

# Prepare data for LSTM
def prepare_data(data, sequence_length=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to 3D: (samples, sequence_length, input_size)
    X = np.expand_dims(X, axis=2)  # Adding input_size dimension (1)
    return X, y, scaler

# Train the LSTM model
def train_model(X_train, y_train, input_size=1, hidden_size=50, epochs=50, lr=0.001):
    model = LSTM(input_size=input_size, hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)  # (batch_size, seq_length, input_size)
    y_train = torch.tensor(y_train, dtype=torch.float32)  # (batch_size,)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
    
    return model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse request data
        request_data = request.json
        ticker = request_data['ticker'].upper() + ".NS"  # Add ".NS" to ensure correct symbol
        days = int(request_data['days'])
        period = request_data.get('period', '6mo')
        
        # Get and preprocess stock data
        data = get_stock_data(ticker, period)
        X, y, scaler = prepare_data(data)
        
        # Split into train and test
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test = X[train_size:]
        
        # Train the LSTM model
        model = train_model(X_train, y_train)
        model.eval()
        
        # Predict the future days
        last_sequence = torch.tensor(X_test[-1:], dtype=torch.float32)  # (1, seq_length, input_size)
        predictions = []
        for _ in range(days):
            with torch.no_grad():
                next_pred = model(last_sequence)  # Predict next step
                predictions.append(next_pred.item())
                
                # Update the sequence for the next prediction
                next_pred_scaled = next_pred.unsqueeze(-1)  # Ensure next_pred is shaped (1, 1, 1)
                # Remove the first element of last_sequence and add next_pred
                last_sequence = torch.cat((last_sequence[:, 1:, :], next_pred_scaled), dim=1)
        
        # Transform predictions back to original scale
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return jsonify({'predictions': predictions.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
