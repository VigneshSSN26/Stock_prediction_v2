from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

ALPHA_VANTAGE_API_KEY = "7ZWGJ9RC72KQNHH6"

# Route for predicting stock prices
@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    try:
        request_data = request.json
        ticker = request_data['ticker'].upper() + ".NS"
        days = int(request_data['days'])  # Prediction horizon
        period = request_data.get('period', 'max')
        
        # Fetch stock data
        data = get_stock_data(ticker, period)
        
        # Prepare datasets for all features
        X_open, _, scaler_open = prepare_multi_feature_data(data, 'Open', days)
        X_high, _, scaler_high = prepare_multi_feature_data(data, 'High', days)
        X_low, _, scaler_low = prepare_multi_feature_data(data, 'Low', days)
        X_volume, _, scaler_volume = prepare_multi_feature_data(data, 'Volume', days)
        
        # Predict using LSTM models
        pred_open = model_open(torch.tensor(X_open[-1:], dtype=torch.float32).to(device))
        pred_high = model_high(torch.tensor(X_high[-1:], dtype=torch.float32).to(device))
        pred_low = model_low(torch.tensor(X_low[-1:], dtype=torch.float32).to(device))
        pred_volume = model_volume(torch.tensor(X_volume[-1:], dtype=torch.float32).to(device))
        
        # Consolidate predictions
        pred_concat = torch.cat([pred_open, pred_high, pred_low, pred_volume], dim=1)
        final_pred = consolidation_nn(pred_concat)
        
        # Inverse transform to original scale
        predictions = scaler_open.inverse_transform(final_pred.cpu().detach().numpy())
        return jsonify({'predictions': predictions.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Fetch stock data
def get_stock_data(ticker, period='max'):
    data = yf.download(tickers=ticker, period=period, interval='1d')
    if data.empty:
        raise ValueError("No data found for the specified ticker.")
    return data

# Prepare data for each feature
def prepare_multi_feature_data(data, feature_col, prediction_horizon, sequence_length=75):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[feature_col]].values)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - prediction_horizon + 1):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i:i+prediction_horizon, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = np.expand_dims(X, axis=2)  # Reshape to 3D
    return X, y, scaler

# LSTM Model Definition
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
        out = self.fc(out[:, -1, :])  # Use the last time step
        return out

# Consolidation Neural Network Definition
class ConsolidationNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=1):
        super(ConsolidationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function for LSTM models
def train_model(X, y, input_size=1, hidden_size=150, output_size=1, epochs=75, lr=0.01, batch_size=32, bidirectional=True):
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    return model

# Train and save models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = get_stock_data('RELIANCE.NS', '1y')  # Example ticker and period

X_open, y_open, _ = prepare_multi_feature_data(data, 'Open', 10)
X_high, y_high, _ = prepare_multi_feature_data(data, 'High', 10)
X_low, y_low, _ = prepare_multi_feature_data(data, 'Low', 10)
X_volume, y_volume, _ = prepare_multi_feature_data(data, 'Volume', 10)

model_open = train_model(X_open, y_open)
model_high = train_model(X_high, y_high)
model_low = train_model(X_low, y_low)
model_volume = train_model(X_volume, y_volume)

consolidation_nn = ConsolidationNN(input_size=4, hidden_size=64, output_size=10).to(device)

if __name__ == '__main__':
    app.run(debug=True)
