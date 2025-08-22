<<<<<<< HEAD
# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can’t go back!**

If you aren’t satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you’re on your own.

You don’t have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn’t feel obligated to use this feature. However we understand that this tool wouldn’t be useful if you couldn’t customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).
=======
# Stock Price Prediction System

A complete full-stack application for stock price prediction using LSTM neural networks. The system consists of a Flask backend API and a React frontend.

## Project Structure

```
Stock_prediction_v2/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── StockSelector.tsx
│   │   │   ├── TrainingControls.tsx
│   │   │   ├── PredictionChart.tsx
│   │   │   └── MetricsDisplay.tsx
│   │   ├── App.tsx          # Main App component
│   │   └── App.css          # Styling
│   ├── package.json         # Node.js dependencies
│   └── .env                 # Environment variables
└── README.md               # This file

Stock_backend/               # Flask backend API
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── run.py                 # Production server
├── Procfile               # Deployment config
├── model/                 # Saved models directory
└── data/                  # Datasets directory
```

## Features

- **Real-time Stock Data**: Fetch historical data from Alpha Vantage API
- **LSTM Model Training**: Train custom LSTM models for stock prediction
- **Multi-feature Prediction**: Predict Open, High, Low, Close, and Volume
- **Neural Network Aggregator**: Consolidated final predictions
- **Interactive Charts**: Visualize historical data and predictions using Chart.js
- **Responsive UI**: Modern, responsive web interface
- **Indian & US Stocks**: Support for both NSE and US markets

## Backend Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Navigate to the backend directory:
```bash
cd Stock_backend
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

The backend will be available at `http://localhost:7860`

### Backend API Endpoints

#### GET /api/history
Retrieve historical stock price data.

**Parameters:**
- `symbol` (string): Stock symbol (e.g., "RELIANCE.NS", "AAPL")
- `period` (string, optional): Time period (default: "1y")

**Example:**
```bash
curl "http://localhost:7860/api/history?symbol=RELIANCE.NS&period=1y"
```

#### POST /api/train
Train LSTM model for a specific stock.

**Request Body:**
```json
{
  "symbol": "RELIANCE.NS",
  "epochs": 20,
  "start_date": "2023-01-01",
  "end_date": "2024-01-01"
}
```

#### POST /api/predict
Get predictions for a trained model.

**Request Body:**
```json
{
  "symbol": "RELIANCE.NS",
  "days": 30
}
```

#### GET /api/metrics
Get model performance metrics.

**Parameters:**
- `symbol` (string): Stock symbol

## Frontend Setup

### Prerequisites
- Node.js 16+
- npm

### Installation

1. Navigate to the frontend directory:
```bash
cd Stock_prediction_v2/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file:
```bash
echo REACT_APP_API_URL=http://localhost:7860 > .env
```

4. Start the development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## Usage

### 1. Select a Stock
- Use the stock selector to choose a stock symbol
- For Indian stocks, use `.NS` suffix (e.g., `RELIANCE.NS`)
- For US stocks, use simple symbols (e.g., `AAPL`)

### 2. Train Model
- Set training parameters:
  - **Epochs**: Number of training iterations (5-100)
  - **Start Date**: Optional start date for training data
  - **End Date**: Optional end date for training data
- Click "Train Model" to start training
- Training may take several minutes depending on epochs

### 3. View Predictions
- After training, predictions will be displayed automatically
- View the prediction chart showing historical vs predicted prices
- Check the prediction summary for all features

### 4. Model Features
The system trains separate LSTM models for:
- **Open Price**
- **High Price**
- **Low Price**
- **Close Price**
- **Volume**

Plus a Neural Network aggregator for consolidated predictions.

## Supported Stock Symbols

### Indian Stocks (NSE)
- RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS
- HINDUNILVR.NS, ITC.NS, SBIN.NS, BHARTIARTL.NS, KOTAKBANK.NS
- AXISBANK.NS, ASIANPAINT.NS, MARUTI.NS, HCLTECH.NS, SUNPHARMA.NS
- WIPRO.NS, ULTRACEMCO.NS, TITAN.NS, BAJFINANCE.NS, NESTLEIND.NS

### US Stocks
- AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, NFLX
- JPM, JNJ, V, PG, UNH, HD, MA, DIS, PYPL, ADBE

## Model Architecture

### LSTM Model
- **Input**: Time series sequences (75 days)
- **Architecture**: Bidirectional LSTM with dropout
- **Output**: Multi-step prediction (30 days)
- **Features**: Open, High, Low, Close, Volume

### Neural Network Aggregator
- **Input**: Scaled feature predictions
- **Architecture**: 3-layer feedforward network
- **Output**: Consolidated Close price prediction
- **Purpose**: Combine individual feature predictions

## Deployment

### Backend Deployment (Render)

1. Push the `Stock_backend` folder to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Configure:
   - **Root Directory**: (leave empty)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run.py`
   - **Environment**: Python

### Frontend Deployment (Vercel/Netlify)

1. Push the `Stock_prediction_v2/frontend` folder to GitHub
2. Deploy to Vercel or Netlify
3. Set environment variable:
   - `REACT_APP_API_URL`: Your backend URL

## Environment Variables

### Frontend (.env)
```
REACT_APP_API_URL=http://localhost:7860
```

### Backend
The backend uses a hardcoded Alpha Vantage API key for demonstration.
For production, use environment variables:
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## Dependencies

### Backend
- Flask==2.3.2
- Flask-Cors==3.0.10
- alpha-vantage==2.3.1
- pandas==2.1.1
- numpy==1.26.0
- scikit-learn==1.3.0
- torch==2.0.1
- waitress==2.1.2

### Frontend
- React 18
- TypeScript
- Axios
- Chart.js
- react-chartjs-2

## Troubleshooting

### Common Issues

1. **API Key Limit**: Alpha Vantage has rate limits. Use sparingly.
2. **Training Time**: Higher epochs = longer training time.
3. **Memory Issues**: Large datasets may require more RAM.
4. **CORS Errors**: Ensure backend CORS is properly configured.

### Error Messages

- **"No data found"**: Check stock symbol format
- **"Training failed"**: Reduce epochs or check data availability
- **"Model not found"**: Train the model first before predicting

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


>>>>>>> 02d18e8c5ea6c9645d8f59d387b2faec25586eb0
