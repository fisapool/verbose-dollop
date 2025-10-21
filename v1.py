#!/usr/bin/env python
# coding: utf-8

"""
Gold Price Forecasting Pipeline
This script provides a comprehensive pipeline for forecasting gold prices using:
1. Data loading and preprocessing (including datetime handling)
2. Exploratory data analysis
3. Multiple forecasting models (ARIMA, LSTM)
4. Model evaluation and comparison
"""

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
sns.set_style('darkgrid')
import re # Added for dynamic frequency determination
import datetime # Added for use in forecast_lstm_future fallback

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout

# For model evaluation
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# For data fetching
import yfinance as yf

# --- REPRODUCIBILITY & SETUP FIX ---
# Set Global SEED for Reproducibility (Fixes SUB-OPTIMAL - Reproducibility)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
# -----------------------------------

def get_frequency_from_filename(data_path):
    """
    Determines the Pandas frequency string from the filename (e.g., 'XAU_30m_data.csv' -> '30min').
    """
    # Default to Daily if not found (used for YFinance or unrecognized CSV names)
    default_freq = 'D'

    # Check for YFinance or if path is the default daily file
    if data_path == 'XAU_1d_data.csv' or not data_path.endswith('.csv'):
        return default_freq

    # Extract frequency part (e.g., '1d', '30m', '1h', '1Month')
    match = re.search(r'XAU_(\d+)([a-zA-Z]+)_data\.csv', data_path)
    
    if match:
        num = match.group(1)
        unit = match.group(2)
        
        # Map units to Pandas frequency strings
        if unit.lower() == 'd':
            return 'D'
        elif unit.lower() == 'h':
            return num + 'H'
        elif unit.lower() == 'm':
            return num + 'min'
        elif unit.lower() == 'w':
            return 'W'
        elif unit.lower() == 'month':
            return 'M'
            
    return default_freq


def fetch_gold_data_yfinance(ticker='GC=F', period='2y', exogenous_tickers=None):
    """
    Fetch gold price data from Yahoo Finance, optionally including exogenous variables.
    (Updated to include exogenous variables for improved trend accuracy)
    """
    print(f"Fetching gold price data from Yahoo Finance for ticker: {ticker}, period: {period}")

    if exogenous_tickers is None:
        exogenous_tickers = ['DX-Y.NYB', 'CL=F', '^GSPC', '^TNX', 'EURUSD=X', 'GBPUSD=X']  # US Dollar Index, Crude Oil, S&P 500, Treasury Yield, EUR/USD, GBP/USD

    try:
        # Fetch gold data
        gold_data = yf.download(ticker, period=period)

        # Check if data is empty
        if gold_data.empty:
            raise ValueError(f"No data returned for ticker {ticker} and period {period}")

        # Handle MultiIndex columns from yfinance
        if isinstance(gold_data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            gold_data.columns = gold_data.columns.get_level_values(0)

        # Select relevant columns for gold
        df = gold_data[['Open', 'High', 'Low', 'Close']].copy()
        df.columns = ['Gold_Open', 'Gold_High', 'Gold_Low', 'Gold_Close']

        # Fetch exogenous data
        for exo_ticker in exogenous_tickers:
            print(f"Fetching exogenous data for {exo_ticker}...")
            exo_data = yf.download(exo_ticker, period=period)

            if not exo_data.empty:
                if isinstance(exo_data.columns, pd.MultiIndex):
                    exo_data.columns = exo_data.columns.get_level_values(0)

                # Use 'Close' price for exogenous variables
                df[f'{exo_ticker}_Close'] = exo_data['Close']
            else:
                print(f"Warning: No data for {exo_ticker}, skipping.")

        # Forward fill any missing values
        df = df.fillna(method='ffill')

        print(f"Data fetched successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        raise ValueError(f"Failed to fetch data from Yahoo Finance: {str(e)}")

# FIX: Added 'freq' parameter
def load_and_preprocess_data(data_path='XAU_1d_data.csv', freq='D', use_yfinance=False, ticker='GC=F', period='2y'):
    """
    Load and preprocess the gold price data.
    Handles datetime conversion and missing dates.

    Parameters:
        data_path (str): Path to the CSV file containing gold price data
        freq (str): The time series frequency (e.g., 'D', '30min'). (FIX: Used for reindexing)
        use_yfinance (bool): Whether to fetch data from Yahoo Finance instead of CSV
        ticker (str): Yahoo Finance ticker symbol (default: 'GC=F' for gold futures)
        period (str): Period to fetch data for (default: '2y' for 2 years)

    Returns:
        pd.DataFrame: Preprocessed DataFrame with datetime index

    Raises:
        FileNotFoundError: If CSV file is not found
        ValueError: If data loading or preprocessing fails
    """
    print("Loading and preprocessing data...")

    if use_yfinance:
        df = fetch_gold_data_yfinance(ticker=ticker, period=period)
    else:
        try:
            # Load data
            df = pd.read_csv(data_path, delimiter=';')

            # Check if required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in CSV: {missing_cols}")

            # Convert date column to datetime with correct format
            df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M', errors='coerce')
            if df['Date'].isna().any():
                raise ValueError("Some dates could not be parsed with the expected format '%Y.%m.%d %H:%M'")

            df.set_index('Date', inplace=True)

        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {data_path}")
        except Exception as e:
            raise ValueError(f"Failed to load data from CSV: {str(e)}")

    # Check if DataFrame is empty after loading
    if df.empty:
        raise ValueError("Loaded data is empty")

    # Fill missing dates with forward fill
    # FIX: CRITICAL - Logic Error: Changed hardcoded freq='D' to dynamic 'freq' parameter
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_range).ffill()

    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def explore_data(df):
    """
    Perform exploratory data analysis on the dataset.
    (Function remains unchanged)
    """
    print("\nExploring data...")

    # Basic info
    print("Data types and non-null count:")
    print(df.info())

    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe().round(2))

    # Plot the close price over time
    plt.figure(figsize=(15, 6))
    plt.plot(df['Close'], label='Gold Price (Close)')
    plt.title('Daily Gold Prices')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_data_for_arima(df, target_column='Close', test_size=0.2):
    """
    Prepare data for ARIMA modeling.
    (Function remains unchanged)
    """
    # Extract target variable
    data = df[target_column].copy()

    # Split into train and test sets
    train_size = int(len(data) * (1 - test_size))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    print(f"\nData split for ARIMA:")
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    return train_data, test_data

def fit_arima_model(train_data, order=(5,1,0)):
    """
    Fit an ARIMA model to the training data.
    (Function remains unchanged)
    """
    print("\nFitting ARIMA model...")
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    return fitted_model

def evaluate_arima_model(fitted_model, test_data):
    """
    Evaluate the ARIMA model and calculate performance metrics.
    (Function remains unchanged)
    """
    # Forecast
    print("\nGenerating forecasts...")
    forecast = fitted_model.forecast(steps=len(test_data))

    # Create test DataFrame with both actual and forecast values
    test_df = pd.DataFrame({
        'Actual': test_data,
        'Forecast': forecast
    }, index=test_data.index)

    # Calculate error metrics
    mse = mean_squared_error(test_df['Actual'], test_df['Forecast'])
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(test_df['Actual'], test_df['Forecast']) * 100

    # Calculate directional accuracy (skip first row to avoid NaN from shift)
    # Predicted direction: if forecast > previous actual, 'Up', else 'Down'
    # Actual direction: if actual > previous actual, 'Up', else 'Down'
    test_df['Predicted_Direction'] = (test_df['Forecast'] > test_df['Actual'].shift(1)).map({True: 'Up', False: 'Down'})
    test_df['Actual_Direction'] = (test_df['Actual'] > test_df['Actual'].shift(1)).map({True: 'Up', False: 'Down'})
    # Drop rows with NaN directions (first row)
    valid_directions = test_df.dropna(subset=['Predicted_Direction', 'Actual_Direction'])
    directional_accuracy = (valid_directions['Predicted_Direction'] == valid_directions['Actual_Direction']).mean() * 100

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }

    print("\nARIMA Model Performance:")
    for metric, value in metrics.items():
        if metric == 'Directional_Accuracy':
            print(f"{metric}: {value:.2f}%")
        else:
            print(f"{metric}: {value:.2f}")

    return test_df, metrics

def plot_arima_results(test_df):
    """
    Plot the ARIMA forecasting results.
    (Function remains unchanged)
    """
    # Plot results
    plt.figure(figsize=(15, 6))
    plt.plot(test_df.index, test_df['Actual'], label='Actual Price', color='orange')
    plt.plot(test_df.index, test_df['Forecast'], label='Predicted Price (ARIMA)', linestyle='--', color='green')
    plt.title('Gold Price Forecasting with ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.grid(True)
    plt.show()

# FIX: CRITICAL - Data Leakage
def prepare_data_for_lstm(df, target_column='Gold_Close', window_size=60, test_size=0.2):
    """
    Prepare data for LSTM modeling with exogenous variables.

    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame with exogenous columns
        target_column (str): Column name to forecast (default: 'Gold_Close')
        window_size (int): Number of time steps in the input sequences (default: 60)
        test_size (float): Proportion of data to use for testing (default: 0.2)

    Returns:
        np.ndarray: Training input sequences
        np.ndarray: Training target values
        np.ndarray: Test input sequences
        np.ndarray: Test target values
        MinMaxScaler: Fitted scaler object

    Raises:
        ValueError: If insufficient data for the given window_size
    """
    print("\nPreparing data for LSTM...")

    # Extract all features (exogenous + target)
    feature_columns = [col for col in df.columns if col != target_column]
    feature_columns.append(target_column)  # Add target as last feature

    data = df[feature_columns].copy().values

    # Split into train and test sets (UNSCALED data)
    train_size = int(len(data) * (1 - test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # FIX: CRITICAL - Data Leakage: Scaler fitted only on UNSEEN training data
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)  # Transform test data using the train fit

    print(f"Data split for LSTM:")
    print(f"Train set size: {len(scaled_train_data)}")
    print(f"Test set size: {len(scaled_test_data)}")
    print(f"Number of features: {len(feature_columns)}")

    # Check if sufficient data for sequences
    if len(data) <= window_size:
        raise ValueError(f"Insufficient data: {len(data)} data points, but window_size is {window_size}")

    # Create sequences for multivariate data
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, :])  # All features
            y.append(data[i, -1])  # Target is last column
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_train_data, window_size)
    X_test, y_test = create_sequences(scaled_test_data, window_size)

    # Reshape for LSTM input (samples, timesteps, features)
    # X_train and X_test already have shape (samples, timesteps, features)

    return X_train, y_train, X_test, y_test, scaler

def build_lstm_model(input_shape, dropout_rate=0.3, learning_rate=0.0005):
    """
    Build an improved stacked LSTM model architecture with optimized hyperparameters.
    (Updated for better trend and directional accuracy)
    """
    print("\nBuilding improved stacked LSTM model...")

    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(100, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(100, return_sequences=False),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Use Adam optimizer with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    return model

def build_hybrid_lstm_arima_model(X_train, y_train, X_test, y_test, scaler, arima_order=(5,1,0)):
    """
    Build a hybrid LSTM-ARIMA model.
    LSTM captures non-linear patterns, ARIMA models the residuals for improved accuracy.

    Parameters:
        X_train (np.ndarray): Training input sequences
        y_train (np.ndarray): Training target values
        X_test (np.ndarray): Test input sequences
        y_test (np.ndarray): Test target values
        scaler (MinMaxScaler): Fitted scaler
        arima_order (tuple): ARIMA order (p,d,q)

    Returns:
        tuple: (hybrid_predictions, lstm_model, arima_model)
    """
    print("\nBuilding hybrid LSTM-ARIMA model...")

    num_features = X_train.shape[2]

    # Train LSTM model
    lstm_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    train_lstm_model(lstm_model, X_train, y_train)

    # Get LSTM predictions on training data
    lstm_train_pred = lstm_model.predict(X_train, verbose=0)

    # Inverse transform predictions and actuals (multivariate scaler)
    full_train_pred = np.zeros((len(lstm_train_pred), num_features))
    full_train_pred[:, -1] = lstm_train_pred.flatten()
    lstm_train_pred_inv = scaler.inverse_transform(full_train_pred)[:, -1]

    full_y_train = np.zeros((len(y_train), num_features))
    full_y_train[:, -1] = y_train
    y_train_inv = scaler.inverse_transform(full_y_train)[:, -1]

    # Calculate residuals
    residuals = y_train_inv - lstm_train_pred_inv

    # Fit ARIMA on residuals
    print("Fitting ARIMA on LSTM residuals...")
    arima_model = ARIMA(residuals, order=arima_order)
    arima_fitted = arima_model.fit()

    # Get LSTM predictions on test data
    lstm_test_pred = lstm_model.predict(X_test, verbose=0)
    full_test_pred = np.zeros((len(lstm_test_pred), num_features))
    full_test_pred[:, -1] = lstm_test_pred.flatten()
    lstm_test_pred_inv = scaler.inverse_transform(full_test_pred)[:, -1]

    # Forecast residuals for test period
    arima_residual_forecast = arima_fitted.forecast(steps=len(X_test))

    # Combine predictions: LSTM + ARIMA residuals
    hybrid_pred = lstm_test_pred_inv + arima_residual_forecast

    print("Hybrid model built successfully.")
    return hybrid_pred, lstm_model, arima_fitted

def train_lstm_model(model, X_train, y_train):
    """
    Train the LSTM model.
    (Function remains unchanged)
    """
    print("\nTraining LSTM model...")

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        patience=10,
        monitor='loss',
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,  # Reduced for demonstration
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    return history

def evaluate_lstm_model(model, X_test, y_test, scaler):
    """
    Evaluate the LSTM model and calculate performance metrics.
    (Function remains unchanged)
    """
    print("\nEvaluating LSTM model...")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Inverse transform the predictions and actual values
    # Since scaler is fitted on multivariate data (5 features), but y_test and y_pred are for target only
    # Use the scale and min for the target feature (last column)
    target_scale = scaler.scale_[-1]
    target_min = scaler.min_[-1]
    y_test_actual = y_test * target_scale + target_min
    y_pred_inversed = y_pred.flatten() * target_scale + target_min

    # Calculate error metrics
    mae = np.mean(np.abs(y_test_actual - y_pred_inversed))
    mse = mean_squared_error(y_test_actual, y_pred_inversed)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_inversed) * 100

    # Calculate directional accuracy
    # Predicted direction: if predicted > previous actual, 'Up', else 'Down'
    # Actual direction: if actual > previous actual, 'Up', else 'Down'
    predicted_directions = []
    actual_directions = []
    for i in range(1, len(y_pred_inversed)):
        pred_dir = 'Up' if y_pred_inversed[i] > y_test_actual[i-1] else 'Down'
        actual_dir = 'Up' if y_test_actual[i] > y_test_actual[i-1] else 'Down'
        predicted_directions.append(pred_dir)
        actual_directions.append(actual_dir)
    directional_accuracy = np.mean(np.array(predicted_directions) == np.array(actual_directions)) * 100

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }

    print("\nLSTM Model Performance:")
    for metric, value in metrics.items():
        if metric == 'Directional_Accuracy':
            print(f"{metric}: {value:.2f}%")
        else:
            print(f"{metric}: {value:.2f}")

    return y_pred_inversed.flatten(), metrics

# FIX: CRITICAL - Logic Error: Added 'freq' parameter
def forecast_arima_future(fitted_model, train_data, freq='D', forecast_days=30):
    """
    Forecast future gold prices using ARIMA model.

    Parameters:
        fitted_model (ARIMAResults): Fitted ARIMA model
        train_data (pd.Series): Training data
        freq (str): The time series frequency (e.g., 'D', '30min'). (FIX: Used for future dates)
        forecast_days (int): Number of days to forecast (default: 30)

    Returns:
        pd.DataFrame: DataFrame with future dates and forecasted prices
    """
    print(f"\nForecasting next {forecast_days} periods with ARIMA using freq: {freq}...")

    # Forecast future values
    forecast = fitted_model.forecast(steps=forecast_days)

    # Determine last date in training data
    last_date = pd.to_datetime(train_data.index[-1])

    # Generate future dates
    # FIX: CRITICAL - Logic Error: Replaced pd.Timedelta(days=i) with dynamic freq
    future_dates = [last_date + pd.Timedelta(f'{i}{freq}') for i in range(1, forecast_days + 1)]

    # Create DataFrame with dates and predictions
    future_forecast_df = pd.DataFrame({
        'predicted_close': forecast
    }, index=future_dates)

    print(f"Future forecast completed. Sample predictions:")
    print(future_forecast_df.head())

    return future_forecast_df

def get_exchange_rate():
    """Get USD to MYR exchange rate with fallback. (Note: Fallback is hardcoded - SUB-OPTIMAL)"""
    try:
        # Try real-time API call first
        import requests
        response = requests.get('https://open.er-api.com/v6/latest/USD')
        response.raise_for_status()
        data = response.json()

        if 'MYR' in data['rates']:
            return data['rates']['MYR']
        else:
            print("Warning: MYR not found in API response, using fallback rate.")
            return 4.25  # Fallback to average historical rate (SUB-OPTIMAL: should load last known rate)
    except Exception as e:
        print(f"Error fetching exchange rate from API: {e}. Using fallback rate.")
        return 4.25


def plot_lstm_results(df, predictions, train_size, window_size):
    """
    Plot the LSTM forecasting results.

    FIX: CRITICAL - Indexing/Plotting Errors: Refactored to correctly map predictions
    to historical test dates using train_size and window_size.

    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame
        predictions (np.ndarray): Inverse transformed LSTM predictions (full test set)
        train_size (int): Size of the training data
        window_size (int): Window size used for sequence creation
    """
    test_data_len = len(predictions)

    # 1. Identify the correct historical index for the predictions
    # Predictions (y_test) start after the training data PLUS the lookback window size
    actual_test_dates = df.index[train_size + window_size:]

    # Safety check for size mismatch (if data prep was slightly off)
    if len(actual_test_dates) != test_data_len:
        min_len = min(len(actual_test_dates), test_data_len)
        actual_test_dates = actual_test_dates[:min_len]
        predictions = predictions[:min_len]

    # 2. Get the actual test values aligned with the dates
    actual_test_values = df['Close'].loc[actual_test_dates]

    # 3. Create the plot DataFrame using the correct historical index
    test_df = pd.DataFrame({
        'Actual': actual_test_values.values,
        'LSTM Predictions': predictions
    }, index=actual_test_dates)

    # Separate training data for plotting
    train_data = df['Close'].iloc[:train_size + window_size]

    # Plot results
    plt.figure(figsize=(15, 6))
    plt.plot(train_data.index, train_data, label='Training Data', color='blue')
    plt.plot(test_df['Actual'], label='Actual Price (Test)', color='orange')
    plt.plot(test_df['LSTM Predictions'], label='LSTM Predictions', linestyle='--', color='red')
    plt.title('Gold Price Forecasting with LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.grid(True)
    plt.show()

# FIX: CRITICAL - Logic Error: Added 'freq' parameter and fixed multivariate handling
def forecast_lstm_future(model, X_train_last, scaler, freq='D', forecast_days=30):
    """
    Forecast future gold prices using LSTM model.

        model (tf.keras.Model): Trained LSTM model
        X_train_last (np.ndarray): Last sequence from training data for forecasting
        scaler (MinMaxScaler): Fitted scaler object
        freq (str): The time series frequency (e.g., 'D', '30min'). (FIX: Used for future dates)
        forecast_days (int): Number of days to forecast (default: 30)

    Returns:
        pd.DataFrame: DataFrame with future dates and forecasted prices
    """
    print(f"\nForecasting next {forecast_days} periods with LSTM using freq: {freq}...")

    # Prepare input for forecasting
    last_sequence = X_train_last.copy()  # Shape: (timesteps, features)

    # Get number of features
    num_features = last_sequence.shape[1]

    # Initialize list to store predictions
    forecasts = []

    # Forecast one day at a time, updating the sequence each time
    for _ in range(forecast_days):
        # Get prediction (scaled) - reshape to (1, timesteps, features)
        pred_scaled = model.predict(last_sequence.reshape(1, last_sequence.shape[0], num_features), verbose=0)  # Suppress verbose output

        # Inverse transform to get actual price
        # Create full feature array with predicted target and last exogenous values
        full_pred = np.zeros((1, num_features))
        full_pred[0, -1] = pred_scaled[0, 0]  # Target is last feature
        pred_actual = scaler.inverse_transform(full_pred)[0, -1]

        # Store the prediction
        forecasts.append(pred_actual)

        # Update the sequence: remove first timestep and append new row
        # New row: shift exogenous features from last timestep, use predicted target
        new_row = last_sequence[-1].copy()  # Start with last timestep's values
        new_row[-1] = pred_scaled[0, 0]  # Update target with prediction
        last_sequence = np.vstack([last_sequence[1:], new_row])  # Remove first row, append new

    # Determine last date in original data
    if hasattr(X_train_last, 'index'):
        last_date = pd.to_datetime(X_train_last.index[-1])
    else:
        # Fallback to last date in the original training index
        last_date = datetime.datetime.now()

    # Generate future dates
    # FIX: CRITICAL - Logic Error: Replaced pd.Timedelta(days=i) with dynamic freq
    future_dates = [last_date + pd.Timedelta(f'{i}{freq}') for i in range(1, forecast_days + 1)]

    # Create DataFrame with dates and predictions
    future_forecast_df = pd.DataFrame({
        'predicted_close': forecasts
    }, index=future_dates)

    print(f"Future LSTM forecast completed. Sample predictions:")
    print(future_forecast_df.head())

    return future_forecast_df


# --- MAIN PIPELINE EXECUTION ---

def main(file_path='XAU_1d_data.csv', test_size=0.2, window_size=60, forecast_days=30, use_yfinance=False, ticker='GC=F', period='2y'):
    """
    The main execution flow for the gold price forecasting pipeline.
    """
    print("=== Gold Price Forecasting Pipeline ===")
    
    # FIX: Get dynamic frequency
    data_freq = get_frequency_from_filename(file_path)

    try:
        # Step 1: Load and preprocess data
        # FIX: Pass dynamic frequency
        df = load_and_preprocess_data(file_path, freq=data_freq, use_yfinance=use_yfinance, ticker=ticker, period=period)
        
        # Determine train_size for plotting and indexing consistency
        train_size = int(len(df) * (1 - test_size))
        
        # Step 2: ARIMA Model
        train_data_arima, test_data_arima = prepare_data_for_arima(df, test_size=test_size)
        model_arima = fit_arima_model(train_data_arima)
        test_df_arima, arima_metrics = evaluate_arima_model(model_arima, test_data_arima)
        # plot_arima_results(test_df_arima)

        # Step 3: Future Forecasting (ARIMA)
        # FIX: Pass dynamic frequency
        arima_future = forecast_arima_future(model_arima, train_data_arima, freq=data_freq, forecast_days=forecast_days)
        
        # Step 4: LSTM Model
        # FIX: Data leakage prevented inside this function
        # Determine target column based on data source
        target_column = 'Gold_Close' if use_yfinance else 'Close'
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler = prepare_data_for_lstm(
            df,
            target_column=target_column,
            window_size=window_size,
            test_size=test_size
        )

        # Build and Train LSTM
        lstm_model = build_lstm_model(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
        train_lstm_model(lstm_model, X_train_lstm, y_train_lstm)

        # Evaluate LSTM
        lstm_predictions, lstm_metrics = evaluate_lstm_model(lstm_model, X_test_lstm, y_test_lstm, scaler)

        # FIX: CRITICAL - Indexing/Plotting Errors: Corrected call to the new function
        # plot_lstm_results(df, lstm_predictions, train_size=train_size, window_size=window_size)

        # Step 4.5: Hybrid LSTM-ARIMA Model
        hybrid_predictions, hybrid_lstm_model, hybrid_arima_model = build_hybrid_lstm_arima_model(
            X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler
        )

        # Evaluate Hybrid Model
        hybrid_metrics = evaluate_lstm_model(hybrid_lstm_model, X_test_lstm, y_test_lstm, scaler)  # Reuse LSTM evaluation for hybrid

        # Step 5: Future Forecasting (LSTM)
        # FIX: Pass dynamic frequency
        lstm_future = forecast_lstm_future(lstm_model, X_train_lstm[-1], scaler, freq=data_freq, forecast_days=forecast_days)

        print("\nPipeline Complete.")
        
        # Note: Financial Error (0.935) fix cannot be applied as the conversion logic is missing from this script.
        # If the conversion logic existed, you would change (pred * 0.935) to (pred * 0.916).

    except Exception as e:
        print(f"\nFATAL ERROR in pipeline execution: {e}")

if __name__ == "__main__":
    # --- Example Execution Calls ---
    
    # 1. Run with default daily data (XAU_1d_data.csv) - Recommended for long-term forecasts
    main(file_path='XAU_1d_data.csv') 
    
    # 2. Run with 30-minute data (Requires file XAU_30m_data.csv to be present for local run)
    # main(file_path='XAU_30m_data.csv', window_size=240) 
    
    # 3. Run using live Yahoo Finance data
    # main(use_yfinance=True, period='5y')
