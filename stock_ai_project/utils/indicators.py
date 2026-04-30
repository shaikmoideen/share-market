import pandas as pd


def calculate_indicators(data):
    """
    Calculate:
    - MA50
    - MA200
    - RSI
    - MACD
    - Signal Line
    - Target column for ML prediction
    """

    # ---------------------------------
    # Close price safely converted
    # ---------------------------------
    close_price = data["Close"].squeeze()

    # ---------------------------------
    # Moving Averages
    # ---------------------------------
    data["MA50"] = close_price.rolling(window=50).mean()
    data["MA200"] = close_price.rolling(window=200).mean()

    # ---------------------------------
    # RSI
    # ---------------------------------
    delta = close_price.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # ---------------------------------
    # MACD
    # ---------------------------------
    data["EMA12"] = close_price.ewm(
        span=12,
        adjust=False
    ).mean()

    data["EMA26"] = close_price.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = data["EMA12"] - data["EMA26"]

    data["Signal_Line"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # ---------------------------------
    # ML Target Column
    # ---------------------------------
    data["Next_Close"] = close_price.shift(-1)

    data["Target"] = (
        data["Next_Close"] > close_price
    ).astype(int)

    return data