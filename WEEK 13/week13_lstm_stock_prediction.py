# Week 13 - LSTM for Time Series (Stock Price) Prediction
# Author: baby

# ========== IMPORT LIBRARIES ==========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
import os

# ========== STEP 1: LOAD DATA ==========
if os.path.exists("stock_data.csv"):
    data = pd.read_csv("stock_data.csv")
    print("✅ Loaded dataset from stock_data.csv")
else:
    # If no CSV found, create synthetic sine-wave data for demo
    print("⚠️ No dataset found. Using synthetic data instead.")
    t = np.arange(0, 300, 0.1)
    prices = np.sin(t) + np.random.normal(scale=0.1, size=len(t))
    data = pd.DataFrame({"Close": prices})

# Extract closing prices
prices = data['Close'].values.reshape(-1, 1)

# ========== STEP 2: NORMALIZATION ==========
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# ========== STEP 3: CREATE SEQUENCES ==========
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window_size = 60  # Use past 60 days to predict the next
X, y = create_sequences(scaled_prices, window_size)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# Split into training and testing
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ========== STEP 4: BUILD MODEL ==========
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

# You can switch to RNN easily by replacing above 2 lines with:
# model = Sequential([
#     SimpleRNN(50, return_sequences=True, input_shape=(X.shape[1], 1)),
#     SimpleRNN(50),
#     Dense(1)
# ])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ========== STEP 5: TRAIN MODEL ==========
print("\n🚀 Training started...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
print("✅ Training complete.")

# ========== STEP 6: PREDICTIONS ==========
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# ========== STEP 7: VISUALIZATION ==========
plt.figure(figsize=(10,6))
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('📈 LSTM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# ========== STEP 8: SAVE MODEL & RESULTS ==========
model.save("lstm_stock_model.h5")
print("💾 Model saved as lstm_stock_model.h5")

results = pd.DataFrame({
    "Actual": actual_prices.flatten(),
    "Predicted": predicted_prices.flatten()
})
results.to_csv("lstm_stock_predictions.csv", index=False)
print("📁 Predictions saved as lstm_stock_predictions.csv")

print("\n✨ DONE — Stock price prediction completed successfully!")
