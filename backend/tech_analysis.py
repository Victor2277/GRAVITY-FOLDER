import pandas as pd
import numpy as np

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculates MACD values for a given DataFrame. 
    Assumes 'Close' column exists.
    Returns df with added columns: 'MACD', 'Signal', 'Histogram'.
    """
    df = df.copy()
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    df['MACD'] = macd
    df['Signal'] = signal_line
    df['Histogram'] = histogram
    return df

def analyze_macd_signals(df, lookback_days=5):
    """
    Analyzes the DataFrame for MACD signals with a lookback window.
    
    Args:
        df: DataFrame with OHLC data
        lookback_days: Number of days to check back for signals (default 5)
        
    Returns:
        List of signal dictionaries:
        [{
            'type': 'Bullish'/'Bearish', 
            'name': 'Signal Name', 
            'desc': 'Description',
            'date': 'YYYY-MM-DD',
            'days_ago': int
        }, ...]
    """
    
    # Ensure MACD columns exist
    if 'MACD' not in df.columns or 'Signal' not in df.columns:
        df = calculate_macd(df)
        
    signals = []
    
    # Need at least 20 points for divergence, and enough for lookback
    if len(df) < max(20, lookback_days + 2):
        return []

    # Get data required for lookback
    # Check from i=1 (Yesterday vs Today) to i=lookback_days
    # Note: df.iloc[-1] is latest. df.iloc[-2] is prev.
    # We want to check crossover at index -k.
    # Crossover at -k means: MACD[-k] > Sig[-k] AND MACD[-k-1] < Sig[-k-1] (Golden)
    
    # Loop from 1 (Latest day check) up to lookback_days
    # range(1, 6) -> 1, 2, 3, 4, 5
    for i in range(1, lookback_days + 1):
        idx_curr = -i      # e.g., -1 (Today)
        idx_prev = -i - 1  # e.g., -2 (Yesterday)
        
        try:
            curr_macd = df['MACD'].iloc[idx_curr]
            curr_sig = df['Signal'].iloc[idx_curr]
            prev_macd = df['MACD'].iloc[idx_prev]
            prev_sig = df['Signal'].iloc[idx_prev]
            
            curr_date = df.index[idx_curr].strftime('%Y-%m-%d')
            days_ago = i - 1 # i=1 -> 0 days ago (Today)
            
            # 1. Golden Cross
            if prev_macd < prev_sig and curr_macd > curr_sig:
                signals.append({
                    "type": "Bullish",
                    "name": "MACD 黃金交叉 (Golden Cross)",
                    "desc": "MACD 快線向上突破訊號線，短線多頭趨勢確立。",
                    "date": curr_date,
                    "days_ago": days_ago
                })

            # 2. Death Cross
            if prev_macd > prev_sig and curr_macd < curr_sig:
                signals.append({
                    "type": "Bearish",
                    "name": "MACD 死亡交叉 (Death Cross)",
                    "desc": "MACD 快線向下跌破訊號線，短線空頭趨勢確立。",
                    "date": curr_date,
                    "days_ago": days_ago
                })
        except IndexError:
            continue
        
    # --- Divergence Logic (Still check relative to LATEST price action) ---
    # Divergence is usually a buildup pattern, so we check the latest setup.
    window = 20
    recent_df = df.iloc[-window:]
    
    curr_price_low = df['Low'].iloc[-1]
    min_price_low = recent_df['Low'].min()
    
    curr_price_high = df['High'].iloc[-1]
    max_price_high = recent_df['High'].max()
    
    # Bullish Divergence
    if curr_price_low == min_price_low:
        min_hist = recent_df['Histogram'].min()
        curr_hist = df['Histogram'].iloc[-1]
        min_dif = recent_df['MACD'].min()
        curr_dif = df['MACD'].iloc[-1]
        
        if (curr_hist > min_hist) or (curr_dif > min_dif):
             signals.append({
                "type": "Bullish",
                "name": "MACD 正背離 (Bullish Divergence)",
                "desc": "股價創近期新低，但指標底部墊高，可能蘊釀反彈。",
                "date": df.index[-1].strftime('%Y-%m-%d'),
                "days_ago": 0
            })

    # Bearish Divergence
    if curr_price_high == max_price_high:
        max_hist = recent_df['Histogram'].max()
        curr_hist = df['Histogram'].iloc[-1]
        max_dif = recent_df['MACD'].max()
        curr_dif = df['MACD'].iloc[-1]
        
        if (curr_hist < max_hist) or (curr_dif < max_dif):
             signals.append({
                "type": "Bearish",
                "name": "MACD 負背離 (Bearish Divergence)",
                "desc": "股價創近期新高，但指標頭部降低，上漲動能減弱。",
                "date": df.index[-1].strftime('%Y-%m-%d'),
                "days_ago": 0
            })

    return signals
