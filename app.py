from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

app = Flask(__name__)
CORS(app)

# API to fetch stock data
@app.route('/stock-data', methods=['GET'])
def get_stock_data():
    try:
        ticker = request.args.get('ticker', 'RELIANCE.NS')
        data = yf.download(tickers=ticker, period='1mo', interval='1d')
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].astype(str)  # Convert to string for JSON serialization
        return jsonify(data.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)})

# API for predictions
@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route hit")
    try:
        # Parse request data
        request_data = request.json
        ticker = request_data['ticker'].upper()
        days = int(request_data['days'])
        period = request_data.get('period', '6mo')  # Default to '6mo' if not provided

        # Fetch historical data based on 'period'
        df = yf.download(tickers=f"{ticker}.NS", period=period, interval='1d')
        if df.empty:
            return jsonify({'error': 'No data found for this ticker'})

        # Check if 'Adj Close' exists, otherwise use 'Close'
        price_column = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[price_column] = scaler.fit_transform(df[[price_column]])
        data = df[price_column].values

        look_back = 60  # Number of previous days to consider for each prediction
        X = []
        for i in range(look_back, len(data)):
            X.append(data[i - look_back:i])

        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Single feature (Adj Close or Close)

        if X.shape[0] < 1:
            return jsonify({'error': 'Not enough data for prediction'})

        # Define the LSTM model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train with dummy data (replace with saved model in production)
        model.fit(X, X[:, -1], epochs=1, batch_size=32, verbose=0)

        # Predict future prices for the 'days' requested
        future_prices = []
        last_input = X[-1].reshape(1, look_back, X.shape[2])  # The last available input

        for _ in range(days):
            prediction = model.predict(last_input)  # Predict next value (shape: (1, 1))
            
            # Append the predicted value to future prices
            future_prices.append(scaler.inverse_transform(prediction).tolist()[0][0])

            # Create a new input with the same feature dimensions
            next_input = np.zeros((1, 1, last_input.shape[2]))  # Shape: (1, 1, features)
            next_input[0, 0, 0] = prediction[0][0]  # Replace index 0 with the predicted value
            
            # Update last input for the next prediction
            last_input = np.append(last_input[:, 1:, :], next_input, axis=1)

        return jsonify({'future_prices': future_prices})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
