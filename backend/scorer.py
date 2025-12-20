import yfinance as yf
import pandas as pd
import numpy as np

class StockScorer:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        
    def analyze(self):
        """
        Analyzes the stock based on 10 criteria.
        Returns a dictionary with 'score', 'total', and 'details' (list of dicts).
        """
        score = 0
        details = []
        
        try:
            # optimize: fetch all data at once might be slow, but needed for 10 pts
            info = self.ticker.info
            financials = self.ticker.financials
            balance_sheet = self.ticker.balance_sheet
            cashflow = self.ticker.cashflow
            
            # Helper to safely get latest and previous values from DF
            def get_latest_prev(df, row_name):
                try:
                    if row_name not in df.index:
                        return None, None
                    row = df.loc[row_name]
                    # Valid columns only
                    row = row.dropna()
                    if len(row) < 2:
                        return row.iloc[0], None
                    return row.iloc[0], row.iloc[1] # Most recent is usually 0
                except:
                    return None, None

            # 1. Valuation Safety
            pe = info.get('trailingPE')
            peg = info.get('pegRatio')
            passed = False
            val_str = ""
            
            if pe is not None and pe < 25:
                passed = True
                val_str = f"P/E = {pe:.2f}"
            elif peg is not None and peg < 1.0:
                passed = True
                val_str = f"PEG = {peg:.2f}"
            else:
                val_str = f"P/E={pe}, PEG={peg}"
                
            if passed: score += 1
            details.append({"name": "估值安全 (Valuation)", "passed": passed, "msg": f"{val_str} (Target: PE<25 or PEG<1)", "value": val_str})

            # 2. Revenue Growth
            curr_rev, prev_rev = get_latest_prev(financials, 'Total Revenue')
            passed = curr_rev is not None and prev_rev is not None and curr_rev > prev_rev
            val_str = f"Growth: {((curr_rev/prev_rev)-1)*100:.1f}%" if passed else "No Growth or Data"
            if passed: score += 1
            details.append({"name": "營收成長 (Revenue Growth)", "passed": passed, "msg": f"Latest: {curr_rev:,.0f} > Prev: {prev_rev:,.0f}", "value": val_str})

            # 3. Operating Income Growth
            curr_op, prev_op = get_latest_prev(financials, 'Operating Income')
            passed = curr_op is not None and prev_op is not None and curr_op > prev_op
            if passed: score += 1
            details.append({"name": "營業利潤成長 (Op. Income)", "passed": passed, "msg": f"Latest: {curr_op:,.0f} > Prev: {prev_op:,.0f}", "value": f"{curr_op}"})
            
            # 4. Net Income Growth
            curr_ni, prev_ni = get_latest_prev(financials, 'Net Income')
            passed = curr_ni is not None and prev_ni is not None and curr_ni > prev_ni
            if passed: score += 1
            details.append({"name": "淨利成長 (Net Income)", "passed": passed, "msg": f"Latest: {curr_ni:,.0f} > Prev: {prev_ni:,.0f}", "value": f"{curr_ni}"})

            # 5. Current Ratio (Liquidity)
            # Some yfinance versions use 'Total Current Assets' or 'Current Assets'
            curr_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else (balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in balance_sheet.index else None)
            curr_liab = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else (balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else None)
            
            passed = False
            ratio = 0
            if curr_assets and curr_liab:
                ratio = curr_assets / curr_liab
                if ratio > 1.0:
                    passed = True
            
            if passed: score += 1
            details.append({"name": "償債能力-短期 (Current Ratio)", "passed": passed, "msg": f"Ratio: {ratio:.2f} > 1.0", "value": f"{ratio:.2f}"})

            # 6. Long Term Debt / Net Income (<4)
            lt_debt = balance_sheet.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in balance_sheet.index else 0
            # Note: Sometimes LT debt is null if 0. 
            if pd.isna(lt_debt): lt_debt = 0
            
            passed = False
            debt_ratio = 999
            if curr_ni and curr_ni > 0:
                debt_ratio = lt_debt / curr_ni
                if debt_ratio < 4.0:
                    passed = True
            elif lt_debt == 0: # No debt is good
                 passed = True
                 debt_ratio = 0
            
            if passed: score += 1
            details.append({"name": "償債能力-長期 (LT Debt/NI)", "passed": passed, "msg": f"Ratio: {debt_ratio:.2f} < 4.0", "value": f"{debt_ratio:.2f}"})

            # 7. Stockholder Equity Growth
            curr_eq, prev_eq = get_latest_prev(balance_sheet, 'Stockholders Equity')
            # fallback name
            if curr_eq is None:
                 curr_eq, prev_eq = get_latest_prev(balance_sheet, 'Total Stockholder Equity')
                 
            passed = curr_eq is not None and prev_eq is not None and curr_eq > prev_eq
            if passed: score += 1
            details.append({"name": "股東權益成長 (Equity)", "passed": passed, "msg": f"Latest: {curr_eq:,.0f} > Prev: {prev_eq:,.0f}", "value": f"{curr_eq}"})

            # 8. Shares Outstanding Stability (Repurchase)
            # Note: Balance sheet 'Share Issued' or 'Ordinary Shares Number'
            curr_shares, prev_shares = get_latest_prev(balance_sheet, 'Share Issued')
            if curr_shares is None:
                 curr_shares, prev_shares = get_latest_prev(balance_sheet, 'Ordinary Shares Number')
            
            passed = False
            if curr_shares is not None and prev_shares is not None:
                if curr_shares <= prev_shares:
                    passed = True
            
            if passed: score += 1
            details.append({"name": "籌碼穩定 (Shares Out)", "passed": passed, "msg": f"Latest: {curr_shares:,.0f} <= Prev: {prev_shares:,.0f}", "value": f"{curr_shares}"})

            # 9. OCF > CapEx (Cash Content)
            # Cashflow keys vary. 'Operating Cash Flow', 'Capital Expenditure'
            ocf = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else None
            # CapEx is usually negative in cashflow statement
            capex = cashflow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cashflow.index else None
            
            passed = False
            if ocf is not None and capex is not None:
                # CapEx is negative, take abs
                if ocf > abs(capex):
                    passed = True
            
            if passed: score += 1
            details.append({"name": "含金量 (OCF > CapEx)", "passed": passed, "msg": f"OCF: {ocf:,.0f} > CapEx: {abs(capex) if capex else 0:,.0f}", "value": f"{ocf}"})

            # 10. FCF Growth
            # Calculate FCF manually if not present: OCF + CapEx (since CapEx is negative)
            curr_ocf, prev_ocf = get_latest_prev(cashflow, 'Operating Cash Flow')
            curr_cap, prev_cap = get_latest_prev(cashflow, 'Capital Expenditure')
            
            passed = False
            val_fcf = 0
            if curr_ocf is not None and curr_cap is not None and prev_ocf is not None and prev_cap is not None:
                curr_fcf = curr_ocf + curr_cap
                prev_fcf = prev_ocf + prev_cap
                val_fcf = curr_fcf
                if curr_fcf > 0 and curr_fcf > prev_fcf:
                    passed = True
            
            if passed: score += 1
            details.append({"name": "自由現金流成長 (FCF Growth)", "passed": passed, "msg": f"Latest: {val_fcf:,.0f} > Prev", "value": f"{val_fcf}"})
            
            return {
                "score": score,
                "total": 10,
                "details": details
            }

        except Exception as e:
            # Return empty or error state
            print(f"Scoring failed: {e}")
            return {
                "score": 0,
                "total": 10,
                "details": [{"name": "Error", "passed": False, "msg": str(e), "value": 0}]
            }
