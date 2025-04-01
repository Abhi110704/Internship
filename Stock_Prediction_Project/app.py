from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import traceback
from datetime import datetime

app = Flask(__name__)

# Load the trained model
try:
    model = load_model('stock_prediction_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML frontend

@app.route('/predict', methods=['POST'])
def predict():
    # Getting data from POST request, assuming it's JSON
    data = request.get_json()  # Expecting {"stock_ticker": "AAPL"}
    stock_ticker = data.get('stock_ticker')

    if not stock_ticker:
        return jsonify({'error': 'No stock ticker received.'}), 400

    try:
        # Get the current date dynamically
        current_date = datetime.now().strftime('%Y-%m-%d')

        # Fetch stock data dynamically using yfinance
        stock_data = yf.download(stock_ticker, start='2015-01-01', end=current_date)

        if stock_data.empty:
            return jsonify({'error': f'No data found for ticker {stock_ticker}.'}), 400

        # Use the 'Close' column for prediction
        data = stock_data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)

        # Ensure we have at least 60 days of data
        if len(scaled_data) < 60:
            return jsonify({'error': f'Not enough data available for ticker {stock_ticker}.'}), 400

        # Prepare the last 60 days of data to predict the next day's price
        last_60_days = scaled_data[-60:].reshape(1, 60, 1)

        # Make prediction
        prediction = model.predict(last_60_days)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        # Convert NumPy float32 to Python float for JSON serialization
        return jsonify({'predicted_price': float(predicted_price)})

    except Exception as e:
        # Log the error details
        error_message = f"Error occurred: {str(e)}\n{traceback.format_exc()}"
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
