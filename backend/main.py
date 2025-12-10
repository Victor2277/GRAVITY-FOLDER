from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from .dcf import calculate_dcf
from .indicators import get_indicators

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DCFRequest(BaseModel):
    ticker: str
    fcf: float
    growth_rate: float
    discount_rate: float
    terminal_growth_rate: float
    years: int = 5
    shares_outstanding: float
    net_debt: float

@app.get("/")
def read_root():
    return {"Hello": "Gravity Invest"}

@app.get("/api/stock/{ticker}")
def get_stock_info(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{ticker}/history")
def get_stock_history(ticker: str, period: str = "1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        # Convert index to string for JSON serialization
        hist.index = hist.index.strftime('%Y-%m-%d')
        return hist.to_dict(orient="index")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{ticker}/indicators")
def get_stock_indicators(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y") # Need enough data for indicators
        if hist.empty:
             raise HTTPException(status_code=404, detail="No history found")
        
        indicators = get_indicators(hist)
        return indicators
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{ticker}/financials")
def get_stock_financials(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get balance sheet and cash flow
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        if balance_sheet.empty or cash_flow.empty:
             raise HTTPException(status_code=404, detail="Financial data not found")

        # Extract latest available data
        # Note: yfinance returns data with most recent date first (column 0)
        
        # Free Cash Flow
        # Try to get Free Cash Flow directly, or calculate it (Operating Cash Flow - Capital Expenditure)
        try:
            fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
        except KeyError:
            try:
                ocf = cash_flow.loc['Total Cash From Operating Activities'].iloc[0]
                capex = cash_flow.loc['Capital Expenditures'].iloc[0]
                fcf = ocf + capex # Capex is usually negative
            except KeyError:
                 fcf = 0 # Fallback
        
        # Net Debt
        # Total Debt - Cash & Cash Equivalents
        try:
            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
        except KeyError:
            total_debt = 0
            
        try:
            cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
        except KeyError:
            cash = 0
            
        net_debt = total_debt - cash
        
        # Shares Outstanding
        shares_outstanding = info.get('sharesOutstanding', 0)
        
        # Beta
        beta = info.get('beta', 1.0)
        
        return {
            "ticker": ticker,
            "freeCashFlow": fcf,
            "netDebt": net_debt,
            "sharesOutstanding": shares_outstanding,
            "beta": beta,
            "currency": info.get('currency', 'USD')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/valuation/dcf")
def calculate_dcf_endpoint(data: DCFRequest):
    try:
        result = calculate_dcf(
            fcf=data.fcf,
            growth_rate=data.growth_rate,
            discount_rate=data.discount_rate,
            terminal_growth_rate=data.terminal_growth_rate,
            years=data.years,
            shares_outstanding=data.shares_outstanding,
            net_debt=data.net_debt
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommendation/{ticker}")
def get_recommendation(ticker: str):
    # Simple recommendation logic based on technicals
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
             return {"recommendation": "Neutral", "reason": "No data"}
             
        indicators = get_indicators(hist)
        
        # Logic
        score = 0
        reasons = []
        
        # RSI
        rsi = indicators.get("RSI", 50)
        if rsi < 30:
            score += 1
            reasons.append("RSI indicates Oversold (Bullish)")
        elif rsi > 70:
            score -= 1
            reasons.append("RSI indicates Overbought (Bearish)")
            
        # MACD
        if indicators.get("MACD_Diff", 0) > 0:
            score += 0.5
            reasons.append("MACD Histogram positive (Bullish)")
        else:
            score -= 0.5
            reasons.append("MACD Histogram negative (Bearish)")
            
        # KD
        k = indicators.get("K", 50)
        d = indicators.get("D", 50)
        if k < 20 and k > d:
             score += 1
             reasons.append("KD Golden Cross in oversold zone (Bullish)")
        elif k > 80 and k < d:
             score -= 1
             reasons.append("KD Death Cross in overbought zone (Bearish)")
             
        if score >= 1.5:
            rec = "Strong Buy"
        elif score >= 0.5:
            rec = "Buy"
        elif score <= -1.5:
            rec = "Strong Sell"
        elif score <= -0.5:
            rec = "Sell"
        else:
            rec = "Hold"
            
        return {
            "ticker": ticker,
            "recommendation": rec,
            "score": score,
            "reasons": reasons,
            "indicators": indicators
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
