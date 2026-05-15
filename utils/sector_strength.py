import pandas as pd

# ---------------------------------
# Stock → Sector Mapping
# ---------------------------------

sector_map = {
    "TCS": "IT",
    "INFY": "IT",
    "WIPRO": "IT",

    "HDFCBANK": "BANKING",
    "ICICIBANK": "BANKING",
    "KOTAKBANK": "BANKING",
    "SBIN": "BANKING",

    "SUNPHARMA": "PHARMA",
    "DRREDDY": "PHARMA",
    "CIPLA": "PHARMA",

    "RELIANCE": "ENERGY",
    "ONGC": "ENERGY",
    "BPCL": "ENERGY",

    "NTPC": "POWER",
    "POWERGRID": "POWER",

    "MARUTI": "AUTO",
    "EICHERMOT": "AUTO",
    "BAJAJ-AUTO": "AUTO",

    "TATASTEEL": "METALS",
    "HINDALCO": "METALS",
    "JSWSTEEL": "METALS",

    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG"
}

# ---------------------------------
# Sector → Benchmark Leaders
# ---------------------------------

SECTOR_BENCHMARK = {
    "IT": [
        "TCS",
        "INFY",
        "WIPRO"
    ],

    "BANKING": [
        "HDFCBANK",
        "ICICIBANK",
        "SBIN"
    ],

    "PHARMA": [
        "SUNPHARMA",
        "DRREDDY",
        "CIPLA"
    ],

    "AUTO": [
        "MARUTI",
        "EICHERMOT",
        "BAJAJ-AUTO"
    ],

    "ENERGY": [
        "RELIANCE",
        "ONGC",
        "BPCL"
    ],

    "METALS": [
        "TATASTEEL",
        "HINDALCO",
        "JSWSTEEL"
    ],

    "FMCG": [
        "HINDUNILVR",
        "ITC",
        "NESTLEIND"
    ]
}


def get_sector(stock):
    return sector_map.get(
        stock.upper(),
        "UNKNOWN"
    )


def calculate_sector_strength(data_dict):
    """
    data_dict = {
        "TCS": dataframe,
        "INFY": dataframe
    }
    """

    sector_performance = {}

    for stock, df in data_dict.items():
        sector = get_sector(stock)

        if df is None or df.empty:
            continue

        if len(df) < 5:
            continue

        returns = (
            (df["Close"].iloc[-1] - df["Close"].iloc[-5])
            / df["Close"].iloc[-5]
        ) * 100

        if sector not in sector_performance:
            sector_performance[sector] = []

        sector_performance[sector].append(returns)

    sector_strength = {}

    for sector, values in sector_performance.items():
        avg_return = sum(values) / len(values)
        sector_strength[sector] = round(avg_return, 2)

    return sector_strength


def get_sector_label(strength):
    if strength > 2:
        return "🟢 Strong"

    elif strength < -2:
        return "🔴 Weak"

    else:
        return "🟡 Neutral"