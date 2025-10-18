# Week 13 - LSTM for Time Series Forecasting
# Author: baby

# ========== IMPORT REQUIRED LIBRARIES ==========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ========== STEP 1: LOAD DATA ==========
# You can use any stock/time series CSV file with a 'Close' column
# Example: 'AAPL.csv' or 'NIFTY.csv' downloaded from Yahoo Finance
# Or use synthetic data for testing

try:
    data = pd.read_csv("stock_data.csv")
except FileNotFoundError:
    # If no file found, generate synthetic sine wave data
    print("⚠️ 'stock_data.csv' not found — using synthetic data instead.")
    time = np.arange(0, 200, 0.1)
    close_prices = np.sin(time) + np.random.normal(scale=0.1, size=len(time))
    data = pd.DataFrame({"Close": close_prices})

# ========== STEP 2: PREPROCESSING ==========
prices = data['Close'].values.reshape(-1, 1)

# Normalize data (LSTM works best with scaled data)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Create training sequences
X, y = [], []
window_size = 60  # past 60 values → next prediction

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# ========== STEP 3: BUILD LSTM MODEL ==========
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print("✅ Model Summary:")
model.summary()

# ========== STEP 4: TRAIN MODEL ==========
print("\n🚀 Training started...")
history = model.fit(X, y, epochs=20, batch_size=32, verbose=1)
print("✅ Training complete.")

# ========== STEP 5: PREDICTION ==========
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y.reshape(-1, 1))

# ========== STEP 6: VISUALIZATION ==========
plt.figure(figsize=(10, 6))
plt.plot(real_prices, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('📈 LSTM Time Series Forecasting')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()

# ========== STEP 7: SAVE MODEL AND RESULTS ==========
model.save("lstm_model.h5")
print("💾 Model saved as lstm_model.h5")

# Save predictions to CSV
results = pd.DataFrame({
    "Actual": real_prices.flatten(),
    "Predicted": predicted_prices.flatten()
})
results.to_csv("lstm_predictions.csv", index=False)
print("📁 Predictions saved as lstm_predictions.csv")

print("\n✨ DONE — LSTM time series forecasting completed successfully!")
