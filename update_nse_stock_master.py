import pandas as pd
from config import NSE_STOCK_MASTER
import os

# Example official NSE equity list source
# Can be refreshed regularly

url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

try:
    df = pd.read_csv(url)

    # Keep only needed columns
    df = df[[
        "SYMBOL",
        "NAME OF COMPANY"
    ]]

    df.columns = [
        "symbol",
        "name"
    ]

    os.makedirs("data", exist_ok=True)
    
    df.to_csv(
        NSE_STOCK_MASTER,
        index=False
    )

    print("✅ NSE Stock Master Updated Successfully")

except Exception as e:
    print("Error:", e)