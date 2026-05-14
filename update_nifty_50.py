import pandas as pd
import os
from config import NIFTY_50_DATA

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

print("🚀 Updating NIFTY 50 stocks automatically...\n")

try:
    # NSE NIFTY 50 constituent source
    url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"

    df = pd.read_csv(url)

    # Keep only required columns
    df = df[[
        "Symbol",
        "Company Name"
    ]]

    # Rename columns
    df.columns = [
        "symbol",
        "name"
    ]

    # Save locally
    df.to_csv(
        NIFTY_50_DATA,
        index=False
    )

    print("✅ nifty_50.csv updated successfully!")
    print(df.head())

except Exception as e:
    print("❌ Error updating NIFTY 50 list:")
    print(e)