import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

def fetch_data(ticker, start_date='2015-01-01', end_date='2023-01-01'):
    """Fetch stock data directly from Yahoo Finance."""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data found for ticker '{ticker}'. Please enter a valid stock symbol.")
        return stock_data
    except Exception as e:
        raise ValueError(f"Failed to fetch data for ticker '{ticker}': {e}")

def prepare_data(stock_data):
    """Prepare the data for training or prediction."""
    # Use the 'Close' column for prediction
    data = stock_data[['Close']]
    
    # Normalize the data to the range (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    
    return scaled_data, scaler, data

def create_and_train_model(ticker, start_date='2015-01-01', end_date='2023-01-01'):
    """Create and train the LSTM model."""
    # Fetch stock data
    stock_data = fetch_data(ticker, start_date, end_date)
    
    # Prepare data
    scaled_data, scaler, _ = prepare_data(stock_data)
    
    # Prepare training data (sequence_length = 60)
    sequence_length = 60
    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])  # Use past 60 days of data
        y.append(scaled_data[i, 0])  # The next day's closing price

    X, y = np.array(X), np.array(y)

    # Reshape X to match the LSTM input shape (samples, time steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Define the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)  # Output layer to predict the next day's closing price
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)  # Train for 10 epochs (adjustable)
    
    # Save the trained model
    model.save('stock_prediction_model.h5')
    
    print(f"Model training complete and saved as 'stock_prediction_model.h5'")
    return scaler  # Return the scaler for later use

def predict_next_day_price(ticker, model_path='stock_prediction_model.h5', end_date='2023-01-01'):
    """Predict the next day's stock price using the trained model."""
    # Load the trained model
    try:
        model = load_model(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load the model: {e}")
    
    # Fetch stock data for prediction
    stock_data = fetch_data(ticker, end_date=end_date)
    scaled_data, scaler, data = prepare_data(stock_data)
    
    # Ensure we have at least 60 days of data for prediction
    if len(scaled_data) < 60:
        raise ValueError(f"Not enough data available for ticker '{ticker}'. At least 60 days of data are required.")
    
    # Use the last 60 days of data for prediction
    last_60_days = scaled_data[-60:]
    last_60_days = last_60_days.reshape(1, 60, 1)  # Reshape to match LSTM input

    # Predict the next day's price
    prediction = model.predict(last_60_days)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    
    print(f"Predicted next day's price for {ticker}: {predicted_price:.2f}")
    return predicted_price

# Example Usage
try:
    # Train the model once (optional if already trained)
    ticker = 'AAPL'  # Replace with any valid stock ticker
    create_and_train_model(ticker)
    
    # Predict the next day's price
    predict_next_day_price(ticker)
    
    # Test with an invalid stock ticker
    # predict_next_day_price('INVALID_TICKER')  # Uncomment to test error handling
except ValueError as e:
    print(f"Error: {e}")
