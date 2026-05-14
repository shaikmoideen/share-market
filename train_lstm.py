import os
import joblib
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from apscheduler.schedulers.blocking import BlockingScheduler

from config import NSE_STOCK_MASTER

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

SEQUENCE_LENGTH = 60
PERIOD = "2y"

DAILY_LIMIT = 300     # Top 300 → daily
WEEKLY_LIMIT = 500    # Next 500 → weekly

BASE_MODEL_PATH = "models"
DAILY_PATH = os.path.join(BASE_MODEL_PATH, "daily")
WEEKLY_PATH = os.path.join(BASE_MODEL_PATH, "weekly")
ONDEMAND_PATH = os.path.join(BASE_MODEL_PATH, "ondemand")

for path in [DAILY_PATH, WEEKLY_PATH, ONDEMAND_PATH]:
    os.makedirs(path, exist_ok=True)

# --------------------------------------------------
# TRAIN SINGLE STOCK
# --------------------------------------------------

def train_single_stock(symbol, save_folder):
    symbol = symbol.upper().strip().replace(".NS", "")

    ticker = f"{symbol}.NS"
    model_path = os.path.join(save_folder, f"{symbol}_model.keras")
    scaler_path = os.path.join(save_folder, f"{symbol}_scaler.pkl")
    seq_path = os.path.join(BASE_MODEL_PATH, "lstm_seq_len.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"{symbol} already trained → skipping")
        return

    print(f"\nTraining model for: {ticker}")

    try:
        data = yf.download(
            ticker,
            period=PERIOD,
            auto_adjust=True,
            progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna()

        if len(data) < 100:
            print("Not enough data → skipping")
            return

        close_data = data[["Close"]].values

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data.astype(float))

        X = []
        y = []

        for i in range(SEQUENCE_LENGTH, len(scaled_data)):
            X.append(scaled_data[i - SEQUENCE_LENGTH:i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)

        if len(X) == 0:
            print("Sequence creation failed → skipping")
            return

        X = X.reshape((X.shape[0], X.shape[1], 1))

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            Input(shape=(X.shape[1], 1)),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(
            optimizer="adam",
            loss="mean_squared_error"
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )

        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(SEQUENCE_LENGTH, seq_path)

        print(f"Saved → {model_path} ✅")

    except Exception as e:
        print(f"Error with {ticker}: {e}")


# --------------------------------------------------
# GET STOCK GROUPS
# --------------------------------------------------


def get_stock_groups():
    df = pd.read_csv(NSE_STOCK_MASTER)
    symbols = df["symbol"].dropna().tolist()

    daily_stocks = symbols[:DAILY_LIMIT]
    weekly_stocks = symbols[DAILY_LIMIT:DAILY_LIMIT + WEEKLY_LIMIT]

    return daily_stocks, weekly_stocks


# --------------------------------------------------
# DAILY JOB
# --------------------------------------------------


def run_daily_training():
    print("\n===== DAILY TRAINING (Top 300 Stocks) =====")

    daily_stocks, _ = get_stock_groups()

    for symbol in daily_stocks:
        train_single_stock(symbol, DAILY_PATH)


# --------------------------------------------------
# WEEKLY JOB
# --------------------------------------------------


def run_weekly_training():
    print("\n===== WEEKLY TRAINING (Next 500 Stocks) =====")

    _, weekly_stocks = get_stock_groups()

    for symbol in weekly_stocks:
        train_single_stock(symbol, WEEKLY_PATH)


# --------------------------------------------------
# ON-DEMAND FROM DASHBOARD
# --------------------------------------------------


def run_on_demand_training(symbol):
    """
    This should be called from dashboard.py

    Example:
        from train_lstm import run_on_demand_training
        run_on_demand_training("ANURAS")
    """

    print(f"\n===== ON-DEMAND TRAINING ({symbol}) =====")
    train_single_stock(symbol, ONDEMAND_PATH)


# --------------------------------------------------
# AUTO SCHEDULER
# --------------------------------------------------


def start_scheduler():
    scheduler = BlockingScheduler()

    # Choice 1 → Every day auto scheduler
    scheduler.add_job(
        run_daily_training,
        trigger="cron",
        day_of_week="mon-fri",
        hour=8,
        minute=0,
        id="daily_training"
    )

    # Choice 2 → Weekly auto scheduler
    scheduler.add_job(
        run_weekly_training,
        trigger="cron",
        day_of_week="sun",
        hour=9,
        minute=0,
        id="weekly_training"
    )

    print("Scheduler Started")
    print("Daily Training  → Mon-Fri at 8:00 AM")
    print("Weekly Training → Sunday at 9:00 AM")

    scheduler.start()


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    start_scheduler()
