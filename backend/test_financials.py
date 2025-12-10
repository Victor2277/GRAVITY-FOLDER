from fastapi.testclient import TestClient
import sys
import os

# Add the current directory to sys.path so we can import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app

client = TestClient(app)

def test_financials():
    ticker = "AAPL"
    print(f"Testing financials for {ticker}...")
    response = client.get(f"/api/stock/{ticker}/financials")
    
    if response.status_code != 200:
        print(f"Failed: {response.status_code}")
        print(response.json())
        return

    data = response.json()
    print("Success! Data received:")
    print(f"Ticker: {data['ticker']}")
    print(f"Free Cash Flow: {data['freeCashFlow']}")
    print(f"Net Debt: {data['netDebt']}")
    print(f"Shares Outstanding: {data['sharesOutstanding']}")
    print(f"Beta: {data['beta']}")
    print("-" * 20)
    
    # Basic validation
    assert data['ticker'] == ticker
    assert isinstance(data['freeCashFlow'], (int, float))
    assert isinstance(data['netDebt'], (int, float))
    assert isinstance(data['sharesOutstanding'], (int, float))

if __name__ == "__main__":
    test_financials()
