import pandas as pd
import numpy as np
import sys
import os

# Ensure backend can be imported
sys.path.append(os.getcwd())
from backend.tech_analysis import analyze_macd_signals

def test_signals():
    print("--- Testing MACD Signals ---")
    
    # 1. Create Synthetic Data for Golden Cross
    # Scenario: Price trends down then up
    dates = pd.date_range(start='2023-01-01', periods=50)
    data = {'Close': np.concatenate([
        np.linspace(100, 90, 20), # Down
        np.linspace(90, 110, 30)  # Up
    ])}
    # Add High/Low for divergence checks
    data['High'] = data['Close'] + 1
    data['Low'] = data['Close'] - 1
    
    df = pd.DataFrame(data, index=dates)
    
    # Analyze
    signals = analyze_macd_signals(df)
    print(f"Signals Found (Expected Golden Cross approx): {signals}")
    
    # 2. Test Death Cross
    # Scenario: Price up then down
    data2 = {'Close': np.concatenate([
        np.linspace(90, 110, 20), # Up
        np.linspace(110, 90, 30)  # Down
    ])}
    data2['High'] = data2['Close'] + 1
    data2['Low'] = data2['Close'] - 1
    df2 = pd.DataFrame(data2, index=dates)
    
    signals2 = analyze_macd_signals(df2)
    print(f"Signals Found (Expected Death Cross approx): {signals2}")

if __name__ == "__main__":
    test_signals()
