from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests

app = Flask(__name__)
CORS(app)

ALPHA_VANTAGE_API_KEY = "7ZWGJ9RC72KQNHH6"

@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    try:
        # Parse request data
        request_data = request.json
        ticker = request_data['ticker'].upper() + ".NS"
        days = int(request_data['days'])
        period = request_data.get('period', 'max')
        
        # Get and preprocess stock data
        data = get_stock_data(ticker, period)
        X, y, scaler = prepare_data(data)
        
        # Split into train, validation, and test
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.2)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        # Train the LSTM model
        model = train_model(X_train, y_train, bidirectional=True)
        model.eval()
        
        # Predict the future days
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        last_sequence = torch.tensor(X_test[-1:], dtype=torch.float32).to(device)
        predictions = []
        for _ in range(days):
            with torch.no_grad():
                next_pred = model(last_sequence)
                predictions.append(next_pred.item())
                
                # Update the sequence for the next prediction
                next_pred_scaled = next_pred.unsqueeze(-1)
                last_sequence = torch.cat((last_sequence[:, 1:, :], next_pred_scaled), dim=1)
        
        # Transform predictions back to original scale
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return jsonify({'predictions': predictions.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Fetch and prepare stock data for LSTM prediction
def get_stock_data(ticker, period='max'):
    try:
        data = yf.download(tickers=ticker, period=period, interval='1d')
        data = data[['Close']].dropna()
        if data.empty:
            raise ValueError("No data found for the specified ticker.")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching stock data: {e}")

# Prepare data for LSTM model
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
    X = np.expand_dims(X, axis=2)
    return X, y, scaler

# Train the LSTM model
def train_model(X_train, y_train, input_size=1, hidden_size=50, epochs=50, lr=0.001, batch_size=32, bidirectional=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    return model

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.num_directions, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_directions, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(self.dropout(out[:, -1, :]))  # Only consider the last time step
        return out

if __name__ == '__main__':
    app.run(debug=True)
