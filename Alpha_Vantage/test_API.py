import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

if API_KEY is None:
    raise ValueError("ALPHAVANTAGE_API_KEY is not found. Check your .env file.")

symbol = "AAPL"

url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey=" + API_KEY

params = {
    "function": "TIME_SERIES_DAILY",
    "symbol": symbol,
    "outputsize": "compact",
}

data = requests.get(url, params=params, timeout=30).json()

print(data.keys())

if "Time Series (Daily)" in data:
    daily_data = data["Time Series (Daily)"]
    print(list(daily_data.items())[:2])
elif "Information" in data:
    print("API message:", data["Information"])
elif "Note" in data:
    print("Rate limit message:", data["Note"])
elif "Error Message" in data:
    print("Error:", data["Error Message"])
else:
    print(data)