import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# --- API KEYS ---
API_KEY = "YOURKEY"
API_SECRET = "YOURSECRETKEY"


# --- CONFIG ---
SYMBOL = "NVDA"
START_DATE = "2022-01-01"
END_DATE = "2026-01-15"

client = StockHistoricalDataClient(API_KEY, API_SECRET)

request_params = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=START_DATE,
    end=END_DATE,
)

# --- FETCH DATA ---
bars = client.get_stock_bars(request_params)
df = bars.df.reset_index()

df = df[df["symbol"] == SYMBOL]


# --- FEATURE ENGINEERING ---
df["return"] = df["close"].pct_change()
df["volatility"] = df["return"].rolling(5).std()
df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()
df["ma_20"] = df["close"].rolling(20).mean()

df["target"] = df["close"].shift(-1)

df = df.dropna()

features = [
    "close",
    "volume",
    "volatility",
    "ma_5",
    "ma_10",
    "ma_20",
]

X = df[features]
y = df["target"]


# --- TRAIN TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


# --- MODEL ---
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"Mean Absolute Error: ${mae:.2f}")


# --- PLOT RESULTS ---
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Price")
plt.plot(preds, label="Predicted Price")
plt.title("NVDA Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


# --- NEXT DAY PREDICTION ---
latest_data = X.iloc[-1:].values
prediction = model.predict(latest_data)[0]

print(f"Next Day Predicted Close for NVDA: ${prediction:.2f}")