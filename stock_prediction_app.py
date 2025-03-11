import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import streamlit as st

# Streamlit App
st.title('📊 Stock Market Prediction App')

# Sidebar for user input
stock_symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL)', 'AAPL')

# Fetch stock data
data = yf.download(stock_symbol, period='5y')
st.write(f"### Showing Data for {stock_symbol}")
st.write(data.tail())

# Plot stock prices
fig, ax = plt.subplots()
ax.plot(data['Close'], label='Close Price')
ax.set_title(f'{stock_symbol} Stock Price')
ax.legend()
st.pyplot(fig)

# Data Preprocessing
data = data[['Close']]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare the data
X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)

# Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Prediction
last_60_days = data_scaled[-60:]
last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
predicted_price = model.predict(last_60_days)
predicted_price = scaler.inverse_transform(predicted_price)

# Show prediction
st.write(f"## 📈 Predicted Stock Price: ${predicted_price[0][0]:.2f}")

# Deploy on Streamlit
st.write("App deployed successfully on Streamlit. ✅")
