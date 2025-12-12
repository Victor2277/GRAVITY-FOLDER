import yfinance as yf
import pandas as pd

def check_structure(ticker):
    print(f"--- Checking {ticker} ---")
    s = yf.Ticker(ticker)
    info = s.info
    print(f"Type: {info.get('quoteType')}")
    print(f"LongName: {info.get('longName')}")
    
    print("Financials Empty?", s.financials.empty)
    print("Cashflow Empty?", s.cashflow.empty)
    print("Quarterly Financials Empty?", s.quarterly_financials.empty)
    
    print(f"Trailing EPS: {info.get('trailingEps')}")
    print(f"Forward EPS: {info.get('forwardEps')}")
    print(f"Revenue: {info.get('totalRevenue')}")

if __name__ == "__main__":
    check_structure("0050.TW")
    check_structure("2330.TW")
