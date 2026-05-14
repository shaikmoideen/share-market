import requests
from config import ALPHA_VANTAGE_API_KEY
from config import NSE_STOCK_MASTER
import pandas as pd

# ---------------------------------
# Indian Priority Stocks
# ---------------------------------
INDIAN_STOCK_PRIORITY = {
    "TCS": {
        "symbol": "TCS.NS",
        "name": "Tata Consultancy Services"
    },
    "INFY": {
        "symbol": "INFY.NS",
        "name": "Infosys"
    },
    "RELIANCE": {
        "symbol": "RELIANCE.NS",
        "name": "Reliance Industries"
    },
    "HDFC": {
        "symbol": "HDFCBANK.NS",
        "name": "HDFC Bank"
    },
    "ICICI": {
        "symbol": "ICICIBANK.NS",
        "name": "ICICI Bank"
    },
    "SBI": {
        "symbol": "SBIN.NS",
        "name": "State Bank of India"
    },
    "ITC": {
        "symbol": "ITC.NS",
        "name": "ITC Limited"
    },
    "TATAMOTORS": {
        "symbol": "TATAMOTORS.NS",
        "name": "Tata Motors"
    },
    "LT": {
        "symbol": "LT.NS",
        "name": "Larsen & Toubro"
    },
    "WIPRO": {
        "symbol": "WIPRO.NS",
        "name": "Wipro"
    }
}

def search_from_alpha(query):
    query_upper = query.upper().replace(" ", "")

    # ---------------------------------
    # STEP 1 → Indian Priority Search
    # ---------------------------------
    results = []

    for key, value in INDIAN_STOCK_PRIORITY.items():
        if query_upper in key or key in query_upper:
            results.append(value)

    if results:
        return results

    # ---------------------------------
    # STEP 2 → Alpha Vantage Fallback
    # ---------------------------------
    try:
        url = (
            "https://www.alphavantage.co/query?"
            f"function=SYMBOL_SEARCH"
            f"&keywords={query}"
            f"&apikey={ALPHA_VANTAGE_API_KEY}"
        )

        response = requests.get(url, timeout=10)
        data = response.json()

        if "bestMatches" not in data:
            return []

        fallback_results = []
        seen = set()
        
        for item in data["bestMatches"][:10]:
            symbol = item["1. symbol"]

            # Force NSE symbol for Indian stocks
            exchange = item.get("4. region", "")

            if "India" in exchange and not symbol.endswith(".NS"):
                symbol += ".NS"
            
            if symbol in seen:
                continue

            seen.add(symbol)

            fallback_results.append({
                "symbol": symbol,
                "name": item["2. name"]
            })

        return fallback_results

    except Exception:
        return []
    
def search_from_nse_csv(query):
    query = query.upper()

    try:
        df = pd.read_csv(
            NSE_STOCK_MASTER
        )

        filtered = df[
            df["symbol"].str.contains(
                query,
                case=False,
                na=False
            ) |
            df["name"].str.contains(
                query,
                case=False,
                na=False
            )
        ]

        results = []

        for _, row in filtered.head(10).iterrows():
            results.append({
                "symbol": row["symbol"] + ".NS",
                "name": row["name"]
            })

        return results

    except Exception:
        return []
    
def search_stock(query):

    # STEP 1 → Search NSE CSV

    results = search_from_nse_csv(query)

    if results:
        return results

    # STEP 2 → Alpha Fallback

    results = search_from_alpha(query)

    if results:
        return results

    # STEP 3 → Nothing found

    return []