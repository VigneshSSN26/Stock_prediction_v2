from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    try:
        request_data = request.json
        ticker = request_data['ticker'].upper() + ".NS"
        days = int(request_data['days'])  # Days to predict
        period = request_data.get('period', 'max')

        # Get and preprocess stock data
        data = get_stock_data(ticker, period)
        models = {}
        lstm_predictions = []
        scalers = {} #store scalers

        features = data.columns.tolist()  # Automatically get all column names

        for feature in features:
            X, y, scaler = prepare_data(data[[feature]], prediction_horizon=days)
            print(f"Training model for feature: {feature}")
            model = train_model(X, y, output_size=days, bidirectional=True)
            models[feature] = model
            
            feature_predictions = sliding_window_prediction(model, X[-1:], days, scaler)

            # Print the feature predictions to see them individually
            print(f"Predictions for {feature}: {feature_predictions}")

            lstm_predictions.append(feature_predictions[0])  # Use the first predicted value for the feature

        # Normalize LSTM predictions
        lstm_predictions = np.array(lstm_predictions)
        lstm_predictions_normalized = (lstm_predictions - lstm_predictions.mean()) / lstm_predictions.std()

        # Debugging: Print normalized inputs
        print(f"Normalized LSTM Predictions: {lstm_predictions_normalized}")

        # Neural Network Processing
        all_predictions_tensor = torch.tensor(lstm_predictions_normalized, dtype=torch.float32).unsqueeze(0)  # Batch dimension
        neural_network = NeuralNetwork(input_size=len(features), hidden_size1=64, hidden_size2=32, output_size=1)
        final_prediction = neural_network(all_predictions_tensor)

        # Denormalize final prediction
        final_prediction = final_prediction.detach().numpy().flatten()
        denormalized_prediction = (final_prediction * lstm_predictions.std()) + lstm_predictions.mean()

# Debugging: Print denormalized output
        print(f"Denormalized Final Prediction: {denormalized_prediction}")

        # Return the denormalized prediction
        return jsonify({'final_prediction': denormalized_prediction.tolist()})


    except Exception as e:
        return jsonify({'error': str(e)})

# Fetch stock data
def get_stock_data(ticker, period='max'):
    data = yf.download(tickers=ticker, period=period, interval='1d')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()  # Select only relevant columns
    if data.empty:
        raise ValueError("No data found for the specified ticker.")
    # Logarithmic transformation for volume to handle outliers
    data['Volume'] = np.log1p(data['Volume'])
    return data

# Prepare data
def prepare_data(data, prediction_horizon, sequence_length=75):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - prediction_horizon + 1):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i:i+prediction_horizon, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.expand_dims(X, axis=2)  # Reshape to 3D
    return X, y, scaler

# Train the LSTM model
def train_model(X, y, input_size=1, hidden_size=100, output_size=1, epochs=50, lr=0.01, batch_size=32, bidirectional=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, bidirectional=bidirectional).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert to tensors
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

# Iterative sliding window prediction
def sliding_window_prediction(model, X_input, days, scaler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_input = torch.tensor(X_input, dtype=torch.float32).to(device)
    predictions = []

    for _ in range(days):
        with torch.no_grad():
            pred = model(X_input)  # Predict one step ahead
            predictions.append(pred.cpu().numpy().flatten()[0])  # Save the first (and only) value

            # Print prediction at each step
            print(f"Prediction: {pred.cpu().numpy().flatten()[0]}")

            # Update the input with the new prediction
            pred_scaled = pred.unsqueeze(-1)  # Expand dimensions to match input shape
            X_input = torch.cat((X_input[:, 1:, :], pred_scaled), dim=1)  # Slide the window

    # Transform predictions back to original scale
    predictions = np.array(predictions)
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten().tolist()

# Neural Network to consolidate LSTM predictions
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_size1=64, hidden_size2=32, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        print(f"After fc2: {x}")
        x = self.fc3(x)
        print(f"After fc3 (Output): {x}")
        return x

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

if __name__ == '__main__':
    app.run(debug=True)
