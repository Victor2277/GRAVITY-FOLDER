import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

class PortfolioManager:
    def __init__(self, filepath='user_portfolios.json'):
        self.filepath = filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def load_portfolio(self, username):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get(username, [])

    def save_portfolio(self, username, portfolio_data):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data[username] = portfolio_data
        
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def add_trade(self, username, symbol, cost, qty, date=None):
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        portfolio = self.load_portfolio(username)
        # Check if symbol exists, maybe average the cost?
        # For simplicity, we just add a new lot or list entry. 
        # But UI wants a Summary. Let's append new trade for now.
        trade = {
            "symbol": symbol.upper(),
            "cost": float(cost),
            "qty": float(qty),
            "date": date
        }
        portfolio.append(trade)
        self.save_portfolio(username, portfolio)
        return True

    def remove_trade(self, username, symbol):
        # Remove all instances of symbol for now, or specific?
        # User requirement: "remove_trade(username, symbol)"
        # Let's remove all trades for this symbol to be simple, 
        # or we need a specific ID.
        portfolio = self.load_portfolio(username)
        new_portfolio = [t for t in portfolio if t['symbol'] != symbol]
        self.save_portfolio(username, new_portfolio)
        return True

    def get_analysis(self, username):
        portfolio = self.load_portfolio(username)
        if not portfolio:
            return pd.DataFrame(), 0, 0

        # Aggregate by symbol
        holdings = {}
        for trade in portfolio:
            sym = trade['symbol']
            if sym not in holdings:
                holdings[sym] = {'qty': 0, 'total_cost': 0}
            
            holdings[sym]['qty'] += trade['qty']
            holdings[sym]['total_cost'] += trade['cost'] * trade['qty']

        analysis_list = []
        total_market_value = 0
        total_pl = 0
        
        for sym, data in holdings.items():
            qty = data['qty']
            avg_cost = data['total_cost'] / qty if qty != 0 else 0
            
            # Fetch Data
            try:
                stock = yf.Ticker(sym)
                # Try fast_info for price (faster)
                current_price = stock.fast_info.last_price
                if current_price is None:
                    # Fallback to history
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                    else:
                        current_price = avg_cost # No data, assume even to avoid crash
                
                # Try to get valuation info
                info = {}
                try:
                    info = stock.info
                except:
                    pass # Soft fail
                
                # Logic for Action
                # 1. Analyst Target
                target = info.get('targetMeanPrice')
                # 2. PE based (very rough)
                pe = info.get('forwardPE')
                
                suggestion = "Hold"
                fair_value = target if target else 0
                
                if fair_value > 0:
                    if current_price < fair_value * 0.9:
                        suggestion = "🟢 Buy/Add"
                    elif current_price > fair_value * 1.1:
                        suggestion = "🔴 Sell/Trim"
                    else:
                        suggestion = "⚪ Hold"
                
                market_val = current_price * qty
                pl = market_val - data['total_cost']
                pl_pct = (pl / data['total_cost']) * 100 if data['total_cost'] != 0 else 0
                
                analysis_list.append({
                    "Symbol": sym,
                    "Qty": round(qty, 2),
                    "Avg Cost": round(avg_cost, 2),
                    "Current Price": round(current_price, 2),
                    "Market Value": round(market_val, 2),
                    "Unrealized P/L": round(pl, 2),
                    "ROI (%)": round(pl_pct, 2),
                    "Target Price": round(fair_value, 2) if fair_value else "N/A",
                    "Suggestion": suggestion
                })
                
                total_market_value += market_val
                total_pl += pl
                
            except Exception as e:
                # Handle error for single stock
                print(f"Error analyzing {sym}: {e}")
                continue
                
        df = pd.DataFrame(analysis_list)
        return df, total_market_value, total_pl
