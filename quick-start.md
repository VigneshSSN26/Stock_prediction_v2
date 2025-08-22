# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Start Backend
```bash
cd Stock_backend
python -m venv venv
venv\Scripts\activate  # Windows
# OR source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
python app.py
```
✅ Backend running at http://localhost:7860

### 2. Start Frontend
```bash
cd Stock_prediction_v2/frontend
npm install
echo REACT_APP_API_URL=http://localhost:7860 > .env
npm start
```
✅ Frontend running at http://localhost:3000

### 3. Test the System

1. **Open** http://localhost:3000
2. **Select** a stock (e.g., RELIANCE.NS)
3. **Train** with 10 epochs (quick test)
4. **View** predictions!

## 🧪 Quick Test Commands

### Test Backend API
```bash
# Test history endpoint
curl "http://localhost:7860/api/history?symbol=RELIANCE.NS"

# Test training (small epochs for quick test)
curl -X POST "http://localhost:7860/api/train" \
  -H "Content-Type: application/json" \
  -d "{\"symbol\":\"RELIANCE.NS\",\"epochs\":5}"

# Test predictions
curl -X POST "http://localhost:7860/api/predict" \
  -H "Content-Type: application/json" \
  -d "{\"symbol\":\"RELIANCE.NS\",\"days\":10}"
```

## 📊 What You'll See

- **Stock Selector**: Choose from Indian (.NS) or US stocks
- **Training Controls**: Set epochs and date range
- **Prediction Chart**: Historical vs predicted prices
- **Feature Summary**: All predicted features (Open, High, Low, Close, Volume, Final)

## 🔧 Troubleshooting

### Backend Issues
- **Port 7860 in use**: Change port in `app.py` line 246
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **API errors**: Check Alpha Vantage rate limits

### Frontend Issues
- **CORS errors**: Backend CORS is enabled
- **API not found**: Check `.env` file has correct URL
- **Build errors**: Run `npm install` again

## 🎯 Next Steps

1. **Train more stocks**: Try different symbols
2. **Adjust epochs**: Higher = better accuracy, longer training
3. **Deploy**: Follow deployment guide in README.md
4. **Customize**: Modify models or add features

## 📞 Support

- Check the main README.md for detailed documentation
- API endpoints are documented in the backend section
- All components are fully functional and tested
