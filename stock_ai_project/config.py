from datetime import datetime
from dateutil.relativedelta import relativedelta

NEWS_API_KEY = "612a7281b32f4a9abf0edc18aae5effd"
ALPHA_VANTAGE_API_KEY = "W2653K1FAP713260"

# Dynamic start date → Last 10 years
today = datetime.today()

START_DATE = (
    today - relativedelta(years=10)
).strftime("%Y-%m-%d")

MODEL_FILE = "stock_model.pkl"
ACCURACY_FILE = "model_accuracy.pkl"