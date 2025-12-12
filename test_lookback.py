import pandas as pd
import numpy as np
import sys
import os

# Ensure backend can be imported
sys.path.append(os.getcwd())
from backend.tech_analysis import analyze_macd_signals

def test_lookback():
    print("--- Testing MACD Lookback (5 Days) ---")
    
    # Create periods
    days = 30
    dates = pd.date_range(end='2024-12-12', periods=days)
    
    # Scenario: Golden Cross 3 days ago
    # We want indices:
    # -1 (Today): Bullish
    # -2 (Yesterday): Bullish
    # -3 (2 days ago): Bullish
    # -4 (3 days ago): CROSSOVER (Curr > Sig, Prev < Sig)
    # -5 (4 days ago): Bearish
    
    # Signal Line: Flat at 0 for simplicity
    # MACD Line: Starts at -10, crosses to +10 around index -4
    
    macd_vals = np.linspace(-10, 10, days)
    # Shift crossover to 3 days ago (index -4)
    # If len=30, -1 is index 29. -4 is index 26.
    # We want crossover between 25 and 26.
    
    # Let's manually construct to be sure
    macd = np.zeros(days)
    signal = np.zeros(days)
    
    # Days 0-25: MACD < Signal (Bearish)
    macd[0:26] = -1.0 
    signal[0:26] = 0.0
    
    # Days 26-29: MACD > Signal (Bullish) - Cross happened at index 26 (4th from last: -4)
    # Wait, lookback loop:
    # i=1 (idx -1, -2): Today vs Yesterday (Both > 0) -> No Cross
    # i=2 (idx -2, -3): Yes vs 2Ago (Both > 0) -> No Cross
    # i=3 (idx -3, -4): 2Ago vs 3Ago (2Ago > 0, 3Ago < 0) -> GOLDEN CROSS!
    # "3 Days Ago" means i=3 ? Yes.
    
    macd[26:] = 1.0
    
    df = pd.DataFrame(index=dates)
    df['Close'] = 100 # Dummy
    df['High'] = 101
    df['Low'] = 99
    
    # Fake the calculations results
    df['MACD'] = macd
    df['Signal'] = signal
    df['Histogram'] = macd - signal
    
    # Analyze
    signals = analyze_macd_signals(df, lookback_days=5)
    
    print(f"Signals Found: {len(signals)}")
    for s in signals:
        print(f" - {s['name']} | Type: {s['type']} | {s['days_ago']} Days Ago | Date: {s['date']}")

if __name__ == "__main__":
    test_lookback()
