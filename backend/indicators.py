import pandas as pd
import ta

def get_indicators(df: pd.DataFrame) -> dict:
    """
    Calculate technical indicators for the given dataframe.
    Expects dataframe with columns: 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    
    # Ensure columns are correct
    if df.empty:
        return {}
        
    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Stochastic Oscillator (KD)
    stoch = ta.momentum.StochasticOscillator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14,
        smooth_window=3
    )
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()
    
    # RSI
    rsi
    
     = ta.momentum.RSIIndicator(close=df['Close'])
    df['RSI'] = rsi.rsi()
    
    # Get the latest values
    latest = df.iloc[-1]
    
    return {
        "MACD": latest['MACD'],
        "MACD_Signal": latest['MACD_Signal'],
        "MACD_Diff": latest['MACD_Diff'],
        "K": latest['K'],
        "D": latest['D'],
        "RSI": latest['RSI'],
        "history": df[['MACD', 'MACD_Signal', 'MACD_Diff', 'K', 'D', 'RSI']].tail(50).to_dict(orient='records')
    }
