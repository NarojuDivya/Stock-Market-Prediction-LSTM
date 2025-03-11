import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import streamlit as st

# Streamlit App Title
st.title('📊 Stock Market Prediction App')

# Sidebar Input for Stock Symbol or File Upload
option = st.sidebar.radio('Choose Data Source:', ['Fetch Live Data', 'Upload CSV File'])

if option == 'Fetch Live Data':
    stock_symbols = st.sidebar.text_input('Enter Stock Symbols (comma-separated, e.g., AAPL, TSLA)', 'AAPL, TSLA')
    symbols = [s.strip() for s in stock_symbols.split(',')]

    for stock_symbol in symbols:
        try:
            data = yf.download(stock_symbol, period='5y')

            if data.isnull().values.any():
                data.fillna(method='ffill', inplace=True)

            if data.empty:
                st.error(f'❌ No data found for {stock_symbol}.')
                continue

            # Feature Scaling
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            # Prepare Training Data
            X, Y = [], []
            time_step = 50
            for i in range(time_step, len(data_scaled)):
                X.append(data_scaled[i-time_step:i, 0])
                Y.append(data_scaled[i, 0])

            X, Y = np.array(X), np.array(Y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            # Build the LSTM Model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            # Compile and Fit the Model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, Y, epochs=10, batch_size=64, verbose=0)

            # Predict Future Stock Prices
            predicted_prices = model.predict(X)
            predicted_prices = scaler.inverse_transform(predicted_prices)

            # Plot the Results
            st.subheader(f'📊 Closing Price vs Predicted Price for {stock_symbol}')
            fig, ax = plt.subplots()
            ax.plot(data.index[time_step:], data['Close'][time_step:], color='blue', label='Actual Price')
            ax.plot(data.index[time_step:], predicted_prices, color='orange', label='Predicted Price')
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f'❌ An error occurred with {stock_symbol}: {e}')

elif option == 'Upload CSV File':
    uploaded_file = st.file_uploader('Upload CSV File', type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Assume the CSV has 'Date' and 'Close' columns
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        # Feature Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        # Prepare Training Data
        X, Y = [], []
        time_step = 50
        for i in range(time_step, len(data_scaled)):
            X.append(data_scaled[i-time_step:i, 0])
            Y.append(data_scaled[i, 0])

        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build the LSTM Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile and Fit the Model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, Y, epochs=10, batch_size=64, verbose=0)

        # Predict Future Stock Prices
        predicted_prices = model.predict(X)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # Plot the Results
        st.subheader('📊 Closing Price vs Predicted Price')
        fig, ax = plt.subplots()
        ax.plot(data.index[time_step:], data['Close'][time_step:], color='blue', label='Actual Price')
        ax.plot(data.index[time_step:], predicted_prices, color='orange', label='Predicted Price')
        ax.legend()
        st.pyplot(fig)

st.success('✅ Prediction Completed Successfully!')
