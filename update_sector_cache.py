import joblib
import pandas as pd
import yfinance as yf

from config import START_DATE
from utils.sector_strength import (
    SECTOR_BENCHMARK,
    calculate_sector_strength
)

CACHE_FILE = "sector_strength_cache.pkl"


def update_sector_strength_cache():
    print("Updating sector strength cache...")

    data_dict = {}

    for sector, symbols in SECTOR_BENCHMARK.items():
        for symbol in symbols:
            try:
                symbol = str(symbol).strip().upper().replace(".NS", "")
                ticker = f"{symbol}.NS"

                stock_data = yf.download(
                    ticker,
                    start=START_DATE,
                    progress=False,
                    auto_adjust=True
                )

                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = (
                        stock_data.columns.get_level_values(0)
                    )

                if not stock_data.empty:
                    data_dict[symbol] = stock_data
                else:
                    print(f"Skipped empty data: {ticker}")

            except Exception as e:
                print(f"Failed for {ticker}: {e}")

    sector_strengths = calculate_sector_strength(data_dict)

    joblib.dump(sector_strengths, CACHE_FILE)

    print(f"Saved → {CACHE_FILE}")


if __name__ == "__main__":
    update_sector_strength_cache()