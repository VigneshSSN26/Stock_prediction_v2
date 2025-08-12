from app import app
from waitress import serve

if __name__ == '__main__':
    print("Starting Stock Prediction API server...")
    print("Server will be available at http://localhost:5000")
    serve(app, host='0.0.0.0', port=5000)
