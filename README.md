# Stock Price Prediction Project

A full-stack web application for stock price prediction using LSTM (Long Short-Term Memory) neural networks. The project consists of a Flask backend API and a React frontend.

## Project Structure

```
Stock_prediction/
├── backend/                 # Flask API backend
│   ├── app.py              # Main Flask application
│   ├── requirements.txt    # Python dependencies
│   ├── model/             # Directory for saved models
│   └── data/              # Directory for datasets
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── StockSelector.tsx
│   │   │   ├── TrainingControls.tsx
│   │   │   ├── PredictionChart.tsx
│   │   │   └── MetricsDisplay.tsx
│   │   ├── App.tsx         # Main App component
│   │   └── App.css         # Styling
│   ├── package.json        # Node.js dependencies
│   └── .env               # Environment variables
└── README.md              # This file
```

## Features

- **Stock Data Retrieval**: Fetch historical stock data from Yahoo Finance
- **LSTM Model Training**: Train custom LSTM models for stock prediction
- **Real-time Predictions**: Get future stock price predictions
- **Interactive Charts**: Visualize historical data and predictions using Chart.js
- **Performance Metrics**: Display RMSE, MAE, accuracy, and loss metrics
- **Responsive UI**: Modern, responsive web interface

## Backend API Endpoints

### GET /api/history
Retrieve historical stock price data.

**Parameters:**
- `symbol` (string): Stock symbol (e.g., "AAPL", "RELIANCE")
- `period` (string, optional): Time period (default: "1y")

**Example:**
```bash
curl "http://localhost:5000/api/history?symbol=AAPL&period=1y"
```

### POST /api/train
Train LSTM model for a specific stock.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "epochs": 50,
  "start_date": "2023-01-01",
  "end_date": "2024-01-01"
}
```

### POST /api/predict
Get predictions for a trained model.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "days": 30
}
```

### GET /api/metrics
Get model performance metrics.

**Parameters:**
- `symbol` (string): Stock symbol

## Installation and Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the Flask application:
```bash
python app.py
```

The backend will be available at `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## Usage

1. **Select a Stock**: Use the stock selector to choose a stock symbol
2. **Train Model**: Set training parameters (epochs, date range) and click "Train Model"
3. **View Predictions**: After training, view the prediction chart showing historical vs predicted prices
4. **Check Metrics**: Review model performance metrics

## Model Architecture

The LSTM model uses:
- **Input Features**: Open, High, Low, Close, Volume
- **Architecture**: Bidirectional LSTM with dropout
- **Output**: Multi-step prediction (default 30 days)
- **Preprocessing**: MinMaxScaler normalization
- **Training**: Adam optimizer with MSE loss

## Deployment

### Option 1: Single Server Deployment

1. Build the React frontend:
```bash
cd frontend
npm run build
```

2. Copy the build folder to backend/static:
```bash
cp -r build ../backend/static
```

3. Deploy the backend to your server (Render, Railway, Heroku)

### Option 2: Separate Services

1. **Backend**: Deploy to Python-friendly platforms (Render, Railway, Heroku)
2. **Frontend**: Deploy to Vercel, Netlify, or similar
3. Update the frontend `.env` file with the backend URL

## Environment Variables

### Frontend (.env)
```
REACT_APP_API_URL=http://localhost:5000
```

### Backend
Create a `.env` file in the backend directory:
```
FLASK_ENV=development
FLASK_DEBUG=1
```

## Dependencies

### Backend
- Flask==2.3.2
- Flask-Cors==3.0.10
- yfinance==0.2.26
- pandas==2.1.1
- numpy==1.26.0
- scikit-learn==1.3.0
- torch==2.0.1
- waitress==2.1.2
- gunicorn==21.2.0
- python-dotenv==1.0.0

### Frontend
- React 18
- TypeScript
- Axios
- Chart.js
- react-chartjs-2

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue on the GitHub repository.


