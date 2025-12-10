
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as solver
from datetime import datetime, timedelta

def fetch_stock_data(tickers, start_date=None, end_date=None):
    """
    Fetches historical stock data from Yahoo Finance.
    Defaults to 1 year of data if no dates provided.
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download data
    try:
        # group_by='column' ensures we can access 'Adj Close' easily
        # auto_adjust=False ensures we get the structure we expect (or True if we want adjusted)
        # using 'Adj Close' is standard.
        raw_data = yf.download(tickers, start=start_date, end=end_date)
        
        if raw_data.empty:
            return pd.DataFrame()

        # Extract Adj Close
        if 'Adj Close' in raw_data:
            data = raw_data['Adj Close']
        else:
            # Fallback if single level columns (e.g. only one ticker or auto_adjust=True)
            # But yf usually returns Open, High... 
            # If auto_adjust=True, 'Close' is the adjusted close.
            if 'Close' in raw_data:
                data = raw_data['Close']
            else:
                data = raw_data # Should not happen usually
        
        # Handle single ticker case (Series -> DataFrame)
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = [tickers] if isinstance(tickers, str) else tickers
            
        # Drop columns that are entirely NaN (invalid tickers)
        data = data.dropna(axis=1, how='all')
        
        return data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_metrics(data):
    """
    Calculates annualized mean returns and covariance matrix.
    Assumes 252 trading days.
    """
    # Calculate daily returns
    returns = data.pct_change(fill_method=None).dropna()
    
    if returns.empty:
        return None, None

    # Annualize
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    return mean_returns, cov_matrix

def get_ret_vol_sr(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    """
    Calculates portfolio Return, Volatility, and Sharpe Ratio for given weights.
    """
    weights = np.array(weights)
    ret = np.sum(weights * mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sr = (ret - risk_free_rate) / vol if vol != 0 else 0
    return np.array([ret, vol, sr])

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    """
    Objective function for Max Sharpe ratio (minimizing negative sharpe).
    """
    return -get_ret_vol_sr(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def minimize_volatility(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    """
    Objective function for Minimum Volatility.
    """
    return get_ret_vol_sr(weights, mean_returns, cov_matrix, risk_free_rate)[1]

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.04):
    """
    Finds the Maximum Sharpe Ratio and Minimum Volatility portfolios.
    Returns dictionaries containing weights and metrics.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds: 0 <= weight <= 1 (No shorting)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Initial guess: Equal weights
    init_guess = num_assets * [1. / num_assets,]

    # 1. Max Sharpe Ratio
    max_sharpe_res = solver.minimize(
        negative_sharpe, 
        init_guess, 
        args=args, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    # 2. Min Volatility
    min_vol_res = solver.minimize(
        minimize_volatility, 
        init_guess, 
        args=args, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    # Organize results
    results = {
        "max_sharpe": {
            "weights": max_sharpe_res.x,
            "metrics": get_ret_vol_sr(max_sharpe_res.x, *args) # [ret, vol, sr]
        },
        "min_vol": {
            "weights": min_vol_res.x,
            "metrics": get_ret_vol_sr(min_vol_res.x, *args)
        }
    }
    
    return results

def get_efficient_frontier(mean_returns, cov_matrix, risk_free_rate=0.04, num_points=20):
    """
    Generates efficient frontier points by minimizing volatility for a range of target returns.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Determine range of returns
    min_ret = np.min(mean_returns)
    max_ret = np.max(mean_returns)
    
    # Avoid extremely narrow ranges
    if max_ret - min_ret < 0.001:
        target_returns = np.linspace(min_ret, min_ret * 1.1, num_points)
    else:
        target_returns = np.linspace(min_ret, max_ret, num_points)
    
    frontier_volatility = []
    frontier_weights = []
    
    for r in target_returns:
        # Constraints: Weights sum to 1 AND Expected Return = r
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) - r}
        )
        
        result = solver.minimize(
            minimize_volatility, 
            num_assets * [1. / num_assets,], 
            args=args, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        if result.success:
            frontier_volatility.append(result.fun)
            frontier_weights.append(result.x)
        else:
            # Fallback or skip if optimization fails for a specific return
            frontier_volatility.append(None)
            frontier_weights.append(None)
            
    return target_returns, frontier_volatility, frontier_weights

def calculate_buy_only_rebalancing(current_shares, target_weights, current_prices):
    """
    Calculates the 'Buy Only' rebalancing strategy.
    Finds the minimum cash injection required to reach target weights without selling any shares.
    """
    # Defensive copy and type conversion
    current_shares = np.array(current_shares)
    target_weights = np.array(target_weights)
    current_prices = np.array(current_prices)
    
    current_values = current_shares * current_prices
    current_total_value = np.sum(current_values)
    
    if current_total_value == 0:
        # If portfolio is empty, simple allocation
        # We need a base amount to start, assume $10,000 for calculation reference or return 0
        # Actually returning 0 implies no action. Let's handle this in UI layer.
        return 0, np.zeros(len(current_prices)), np.zeros(len(current_prices))

    # 1. Calculate implied total wealth for each asset to satisfy its target weight without selling
    # Implied Total = Current Value / Target Weight
    # If Target Weight is 0, this implies infinite total wealth (impossible to satisfy without selling)
    # But in "Buy Only", if current > 0 and target = 0, we simply hold (weight drifts down).
    # We ignore 0 target weight items for this specific max calculation.
    
    implied_totals = []
    for val, w in zip(current_values, target_weights):
        if w > 0.0001:
            implied_totals.append(val / w)
        else:
            implied_totals.append(0)
            
    # 2. The required new total portfolio value is the maximum of these implied totals
    # This ensures that for every asset, New_Total * Target_Weight >= Current_Value
    required_new_total = max(np.max(implied_totals), current_total_value)
    
    # 3. Calculate target values for each asset in the new portfolio
    target_values = required_new_total * target_weights
    
    # 4. Calculate required cash to buy (Difference)
    # Should be non-negative by definition of required_new_total, but clip to 0 just in case
    buy_values = np.maximum(target_values - current_values, 0)
    
    # 5. Convert to shares
    buy_shares = buy_values / current_prices
    
    # Total cash needed
    total_cash_needed = np.sum(buy_values)
    
    return total_cash_needed, buy_shares, target_values
