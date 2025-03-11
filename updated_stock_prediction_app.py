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

# Sidebar Input for Stock Symbol
stock_symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL)', 'AAPL')

# Fetch Stock Data
try:
    data = yf.download(stock_symbol, period='5y')

    # Handle missing values
    if data.isnull().values.any():
        data.fillna(method='ffill', inplace=True)

    # Check if data is empty after download
    if data.empty:
        st.error('❌ Invalid stock symbol or no data found. Please enter a valid stock symbol like AAPL, TSLA, MSFT, etc.')
    else:
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

        # Reshape input to 3D for LSTM
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
        model.fit(X, Y, epochs=20, batch_size=64)

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

        # Display Closing Price
        st.subheader('💹 Latest Closing Price')
        st.write(f"The latest closing price of {stock_symbol} is ${data['Close'].iloc[-1]:.2f}")

        # Display Predicted Price
        st.subheader('🔮 Predicted Closing Price')
        st.write(f"The predicted closing price of {stock_symbol} is approximately ${predicted_prices[-1][0]:.2f}")

except Exception as e:
    st.error(f'❌ An error occurred: {e}')

st.success('✅ Prediction Completed Successfully!')
