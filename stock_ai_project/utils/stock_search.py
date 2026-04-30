import requests
from config import ALPHA_VANTAGE_API_KEY

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

def search_stock(query):
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

        response = requests.get(url)
        data = response.json()

        if "bestMatches" not in data:
            return []

        fallback_results = []

        for item in data["bestMatches"][:10]:
            fallback_results.append({
                "symbol": item["1. symbol"],
                "name": item["2. name"]
            })

        return fallback_results

    except Exception:
        return []