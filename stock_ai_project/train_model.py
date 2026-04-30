import yfinance as yf
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import (
    STOCK_OPTIONS,
    START_DATE,
    MODEL_FILE,
    ACCURACY_FILE
)

from utils.indicators import calculate_indicators


print("🚀 Training Model Started...\n")

all_data = []

# ---------------------------------
# Download + Prepare Data
# ---------------------------------
for stock in STOCK_OPTIONS.keys():
    print(f"Downloading data for {stock}...")

    data = yf.download(
        stock,
        start=START_DATE,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        print(f"Skipping {stock} → No data found\n")
        continue

    data = calculate_indicators(data)
    data = data.dropna()

    if data.empty:
        print(f"Skipping {stock} → No valid ML data\n")
        continue

    all_data.append(data)

# ---------------------------------
# Combine All Stocks Data
# ---------------------------------
if not all_data:
    print("❌ No data available for training.")
    exit()

final_data = pd.concat(
    all_data,
    ignore_index=True
)

# ---------------------------------
# Features
# ---------------------------------
features = [
    "MA50",
    "MA200",
    "RSI",
    "MACD",
    "Signal_Line"
]

X = final_data[features]
y = final_data["Target"]

# ---------------------------------
# STEP 1 → Accuracy Check
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\n📊 Model Accuracy: {accuracy * 100:.2f}%")

# ---------------------------------
# STEP 2 → Retrain Full Data
# ---------------------------------
model.fit(X, y)

# ---------------------------------
# STEP 3 → Save Model + Accuracy
# ---------------------------------
joblib.dump(model, MODEL_FILE)
joblib.dump(accuracy, ACCURACY_FILE)

print(f"💾 Accuracy saved as: {ACCURACY_FILE}")
print(f"\n✅ Model trained successfully!")
print(f"💾 Model saved as: {MODEL_FILE}")