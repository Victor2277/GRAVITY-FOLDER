import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import portfolio_optimizer as po
from scipy import optimize
from datetime import datetime
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from backend.tech_analysis import analyze_macd_signals
from backend.emailer import EmailService
import secrets
import time
import traceback
import requests

# Set page config
# Set page config
st.set_page_config(page_title="台美股價值投資分析平台", layout="wide")

# --- Email Verification Handler (Top Level) ---
query_params = st.query_params
if "verify_token" in query_params and "user" in query_params:
    token = query_params["verify_token"]
    user_to_verify = query_params["user"]
    
    with open('config.yaml') as file:
        v_config = yaml.load(file, Loader=SafeLoader)
    
    # Check token matches
    user_entry = v_config['credentials']['usernames'].get(user_to_verify)
    if user_entry and user_entry.get('verification_token') == token:
        # Verify Success
        v_config['credentials']['usernames'][user_to_verify]['verified'] = True
        v_config['credentials']['usernames'][user_to_verify]['verification_token'] = None # Consume token
        
        with open('config.yaml', 'w') as file:
            yaml.dump(v_config, file, default_flow_style=False)
            
        st.success(f"🎉 帳號 {user_to_verify} 驗證成功！請重新登入。 (Verification Successful! Please login.)")
        st.query_params.clear() # Clean URL
        time.sleep(3)
        st.rerun()
    else:
        st.error("❌ 驗證連結無效或已過期。 (Invalid or expired verification link)")
        if st.button("Close"):
             st.query_params.clear()
             st.rerun()


# --- User Authentication ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login()

if st.session_state["authentication_status"]:
    # --- Approval Check ---
    username = st.session_state["username"]
    user_data = config['credentials']['usernames'].get(username, {})
    
    if not user_data.get('is_approved', False):
        st.error("⚠️ 您的帳號尚待管理員審核，請稍後再試。 (Account Pending Approval)")
        authenticator.logout('Logout', 'main')
        st.stop()

    # --- Admin Dashboard ---
    if username == 'admin':
        # Admin is auto-verified or trusted
        pass
    else:
        # Check Verification Status
        if not user_data.get('verified', False):
            # Allow legacy users who don't have 'verified' field at all? 
            # Policy: If 'verified' key is missing, assume True (Legacy) OR False (Strict)?
            # User request implies new system. Let's assume missing = False for safety, 
            # BUT for existing users this might lock them out. 
            # Better approach: If key missing, default to True (Legacy support), 
            # only newly registered users will have verified=False explicitly.
            is_verified = user_data.get('verified', True) 
            
            if not is_verified:
                st.error("⚠️ 您的帳號尚未驗證 Email，請至信箱收取驗證信啟用帳號。 (Account not verified. Please check your email.)")
                if st.button("Resend Verification Email"):
                    # Resend Logic
                    emailer = EmailService()
                    token = user_data.get('verification_token')
                    if not token:
                        token = secrets.token_urlsafe(16)
                        config['credentials']['usernames'][username]['verification_token'] = token
                        with open('config.yaml', 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                            
                    app_url = emailer.app_url if hasattr(emailer, 'app_url') else "http://localhost:8501"
                    if "localhost" in app_url and "smtp" in st.secrets:
                         app_url = st.secrets["smtp"].get("app_url", app_url)

                    verify_link = f"{app_url}?verify_token={token}&user={username}"
                    
                    content = f"""
                    <h3>Welcome to Gravity!</h3>
                    <p>Please click the link below to verify your account:</p>
                    <a href="{verify_link}">{verify_link}</a>
                    <br><p>If you did not register, please ignore this email.</p>
                    """
                    success, msg = emailer.send_email(user_data['email'], "Gravity Account Verification", content)
                    if success:
                        st.success("驗證信已重發 (Verification email resent)!")
                    else:
                        st.error(f"发送失败: {msg}")
                
                authenticator.logout('Logout', 'main')
                st.stop()
    if username == 'admin':
        st.sidebar.divider()
        if st.sidebar.checkbox("🔧 Admin Panel (管理員後台)"):
            st.title("👥 使用者權限管理")
            
            # Refresh config to ensure we have latest data
            with open('config.yaml') as file:
                current_config = yaml.load(file, Loader=SafeLoader)
            
            users = current_config['credentials']['usernames']
            
            # Simple Stats
            total = len(users)
            pending = sum(1 for u in users.values() if not u.get('is_approved', False))
            st.metric("Total Users", total, delta=f"{pending} Pending", delta_color="inverse")
            
            st.divider()
            
            for u, data in users.items():
                if u == 'admin': continue # Skip self
                
                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
                    c1.markdown(f"**{u}**")
                    c2.caption(data.get('email', 'No Email'))
                    
                    is_approved = data.get('is_approved', False)
                    status_icon = "✅ Approved" if is_approved else "⏳ Pending"
                    c3.write(status_icon)
                    
                    if not is_approved:
                        if c4.button("Approve", key=f"app_{u}"):
                            current_config['credentials']['usernames'][u]['is_approved'] = True
                            with open('config.yaml', 'w') as file:
                                yaml.dump(current_config, file, default_flow_style=False)
                            st.success(f"{u} approved!")
                            st.rerun()
                    else:
                        if c4.button("Revoke", key=f"rev_{u}"):
                            current_config['credentials']['usernames'][u]['is_approved'] = False
                            with open('config.yaml', 'w') as file:
                                yaml.dump(current_config, file, default_flow_style=False)
                            st.warning(f"{u} revoked!")
                            st.rerun()
                    st.divider()
            
            if st.button("Close Admin Panel"):
                st.rerun()
                
            st.stop() # Stop rendering the dashboard when in admin panel
            
        if st.sidebar.checkbox("📢 Admin Push (廣告推播)"):
            st.title("📢 發送廣告推播 (Push Notification)")
            
            with st.form("push_form"):
                subject = st.text_input("信件標題 (Subject)")
                content = st.text_area("信件內容 (Content - HTML supported)", height=200)
                submit_push = st.form_submit_button("🚀 發送給所有會員 (Send to All)")
                
                if submit_push:
                    with open('config.yaml') as file:
                        current_config = yaml.load(file, Loader=SafeLoader)
                    
                    # Extract all emails
                    all_emails = [
                        data['email'] 
                        for u, data in current_config['credentials']['usernames'].items() 
                        if data.get('email')
                    ]
                    
                    if not all_emails:
                        st.warning("No emails found in the user database.")
                    else:
                        st.info(f"Preparing to send to {len(all_emails)} recipients...")
                        emailer = EmailService()
                        success, df_msg = emailer.send_bulk_email(all_emails, subject, content)
                        
                        if success:
                            st.success(f"✅ Push Notification Sent! ({df_msg})")
                        else:
                            st.error(f"❌ Failed to send: {df_msg}")
            
            st.stop()

    authenticator.logout('Logout', 'sidebar')
    st.sidebar.write(f'Welcome *{st.session_state["name"]}*')
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
    st.stop()
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
    
    # Register User Feature (Only visible when not logged in)
    try:
        email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user()
        if email_of_registered_user:
            # New users default to unapproved AND unverified
            config['credentials']['usernames'][username_of_registered_user]['is_approved'] = False
            config['credentials']['usernames'][username_of_registered_user]['verified'] = False
            
            # Generate Token
            token = secrets.token_urlsafe(16)
            config['credentials']['usernames'][username_of_registered_user]['verification_token'] = token
            
            st.success('註冊成功！驗證信已發送至您的 Email，請前往點擊連結啟用帳號。')
            
            # Send Verification Email
            emailer = EmailService()
            # Try to get app_url from secrets or use verified default
            app_url = emailer.app_url if hasattr(emailer, 'app_url') else "http://localhost:8501"
            
            # Construct Link
            verify_link = f"{app_url}?verify_token={token}&user={username_of_registered_user}"
            
            content = f"""
            <h3>Welcome to Gravity!</h3>
            <p>Please click the link below to verify your account:</p>
            <a href="{verify_link}">{verify_link}</a>
            <br>
            <p>After verification, wait for admin approval.</p>
            """
            
            try:
                success, msg = emailer.send_email(email_of_registered_user, "Gravity Account Verification", content)
                if not success:
                    st.error(f"Email sending failed: {msg}")
            except Exception as e:
                st.error(f"Email Error: {str(e)}")

            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)
        
    st.stop()

    # Forgot Password Feature
    try:
        username_forgot_pw, email_forgot_pw, new_random_pw = authenticator.forgot_password()
        if username_forgot_pw:
            # Send Email
            emailer = EmailService()
            subject = "🔒 Gravity Platform - Password Reset"
            content = f"""
            <h3>Password Reset Request</h3>
            <p>Hello <b>{username_forgot_pw}</b>,</p>
            <p>Your password has been reset.</p>
            <p><b>New Password:</b> {new_random_pw}</p>
            <p>Please log in and change your password immediately.</p>
            <br>
            <p>Regards,<br>Gravity Team</p>
            """
            
            success, msg = emailer.send_email(email_forgot_pw, subject, content)
            
            if success:
                st.success('新密碼已發送至您的 Email (New password sent to your email)')
                
                # Update config with new password hash (handled by authenticator internally in session state, but we must save to file)
                # Wait, authenticator.forgot_password() returns the PLAIN text password.
                # We need to HASH it if we were doing it manually, but does authenticator update the cookie/session?
                # Actually, streamlit-authenticator 0.3.x usually requires us to update the config manually or use its helper if avail.
                # But typically it returns the random pw, and we need to save the hashed version or just the fact that it changed.
                # Actually, the 'authenticator.credentials' is a reference to the config dict passed in.
                
                # IMPORTANT: 'authenticator.forgot_password' modifies 'authenticator.credentials' in place with the hashed password?
                # Let's check typical behavior. Usually it does NOT save to file. We must save.
                
                # Update the config object directly from authenticator.credentials which should be updated
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
                    
            else:
                st.error(f"Failed to send email: {msg}")
                
    except Exception as e:
        st.error(e)

# --- Custom CSS for Background ---
# --- Custom CSS for Background ---
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    img_file = "assets/background.png"
    bin_str = get_base64_of_bin_file(img_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
except:
    # Fallback if image not found
    pass

st.markdown("""
<style>
    /* Modern Fintech Dark Theme */
    
    /* Card Style Containers - Semi-transparent Dark */
    .stDataFrame, .stPlotlyChart, [data-testid="stExpander"], div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
         background-color: rgba(17, 24, 39, 0.7); /* Dark with opacity */
         border: 1px solid rgba(255, 255, 255, 0.1);
         border-radius: 8px;
         backdrop-filter: blur(10px);
    }
    
    /* Headers - White/Light */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #f3f4f6 !important; 
    }
    
    h4, h5, h6, .stMarkdown, .stText, label, p, .stCaption {
        color: #e5e7eb !important; 
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #f9fafb !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-size: 14px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(17, 24, 39, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #9ca3af;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(37, 99, 235, 0.2);
        color: #60a5fa; /* Light Blue */
        border-bottom: 2px solid #60a5fa;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        color: white;
    }

</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("股票設定")

# Market Selector
market = st.sidebar.radio("市場來源", ["美股 (US)", "台股 (TW)"], index=0, horizontal=True)

if market == "台股 (TW)":
    default_ticker = "2330"
else:
    default_ticker = "AAPL"

ticker_input = st.sidebar.text_input("輸入股票代碼", value=default_ticker)

# Helper to handle TW stocks logic
if market == "台股 (TW)":
    # Auto append .TW if digit only
    if ticker_input.isdigit():
        ticker_input = f"{ticker_input}.TW"
    # User might input 2330.TW or 2330.TWO, trust them if they do.
    # If they mistype, yfinance will error, which is caught later.

ticker = ticker_input.upper()
st.sidebar.write(f"目前分析: **{ticker}**")

# Timeframe Selector
timeframe = st.sidebar.selectbox("K線週期", 
                                 options=["日 (Day)", "週 (Week)", "月 (Month)", "5分 (5 Min)", "15分 (15 Min)"])

interval_map = {
    "日 (Day)": "1d",
    "週 (Week)": "1wk",
    "月 (Month)": "1mo",
    "5分 (5 Min)": "5m",
    "15分 (15 Min)": "15m"
}
interval = interval_map[timeframe]

# --- Data Fetching ---
@st.cache_resource
def get_stock_data(ticker):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    stock = yf.Ticker(ticker, session=session)
    return stock

@st.cache_data
def get_ohlc(ticker, interval):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    stock = yf.Ticker(ticker, session=session)
    
    
    # Robust period selection
    if interval in ['5m', '15m']:
        period = "5d" 
    elif interval == '1d':
        period = "2y"
    else:
        period = "5y"
        
    try:
        df = stock.history(period=period, interval=interval)
        
        # Retry if empty for daily (sometime 2y is weird for some stocks)
        if df.empty and interval == '1d':
            df = stock.history(period="1y", interval=interval)
        
        if df.empty:
            st.error(f"⚠️ Yahoo Finance returned empty data for **{ticker}**. (Period: {period}, Interval: {interval})")
            return df
            
        # Normalize timezone
        df.index = df.index.tz_localize(None)
        return df
        
    except Exception as e:
        st.error(f"❌ yfinance history() failed: {str(e)}")
        st.code(traceback.format_exc())
        return pd.DataFrame()

stock = get_stock_data(ticker)



# Retry mechanism for fetching stock info
max_retries = 3
for attempt in range(max_retries):
    try:
        info = stock.info
        if info is None:
            raise ValueError("yfinance returned None for stock.info")
            
        # Fallback to ticker if longName is missing
        stock_name = info.get('longName', ticker)
        currency_symbol = "NT$" if info.get('currency') == 'TWD' else "$"
        st.sidebar.success(f"成功載入: {stock_name} ({currency_symbol})")
        break # Success, exit loop
        
    except Exception as e:
        if attempt < max_retries - 1:
            time.sleep(1) # Wait 1s before retry
            continue
        else:
            # Final attempt failed
            st.error(f"❌ 無法找到股票代碼: {ticker}")
            st.error(f"Error Details: {str(e)}")
            st.code(traceback.format_exc())
            st.info("💡 建議: 請確認代碼 (美股如 AAPL, 台股如 2330.TW) 或重新整理。")
            st.stop()

# --- ETF Detection ---
is_etf = False
if info.get('quoteType') == 'ETF':
    is_etf = True
    st.sidebar.info("ℹ️ 偵測到 ETF (Exchange Traded Fund)")
    
# --- Indicators Calculation ---
def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_kd(df, period=9):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    rsv = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    rsv = rsv.fillna(50)
    
    k_values = [50]
    d_values = [50]
    
    for i in range(1, len(rsv)):
        k = (2/3) * k_values[-1] + (1/3) * rsv.iloc[i]
        d = (2/3) * d_values[-1] + (1/3) * k
        k_values.append(k)
        d_values.append(d)
        
    return pd.Series(k_values, index=df.index), pd.Series(d_values, index=df.index)

# --- DCF Logic Class (Adapted from User Provided Code) ---
class DCFValuator:
    def __init__(self, stock):
        self.stock = stock
        self.info = stock.info
        self.income = stock.income_stmt.T
        if self.income.empty: self.income = stock.quarterly_income_stmt.T
        
        self.balance = stock.balance_sheet.T
        self.cashflow = stock.cashflow.T
        if self.cashflow.empty: self.cashflow = stock.quarterly_cashflow.T
        
        # Currency check
        self.currency = self.info.get('currency', 'USD')
        
    def get_value(self, df, possible_names):
        if df.empty: return 0
        for name in possible_names:
            if name in df.columns: return df[name].iloc[0]
        return 0

    def get_risk_free_rate(self):
        # Adjust RF based on Currency/Market
        if self.currency == 'TWD':
            return 0.02 # ~2% for Taiwan 10Y Bond
        return 0.04 # ~4% for US 10Y Treasury

    def perform_valuation(self, custom_growth=None, custom_discount=None):
        try:
            # Beta & Risk Free
            beta = self.info.get('beta', 1.2)
            rf = self.get_risk_free_rate()
            
            # Cost of Equity (Ke)
            ke = rf + beta * (0.10 - rf)
            # WACC (Simplified to max of ke and 7% as per user code logic)
            wacc = max(ke, 0.07)
            if custom_discount:
                wacc = custom_discount

            # FCF Calculation
            ocf = self.get_value(self.cashflow, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
            capex = self.get_value(self.cashflow, ['Capital Expenditure', 'Capital Expenditures'])
            last_fcf = ocf + capex
            
            # Fallback to Net Income if FCF <= 0
            if last_fcf <= 0:
                net_income = self.get_value(self.income, ['Net Income', 'Net Income Common Stockholders'])
                last_fcf = net_income if net_income > 0 else None
            
            if not last_fcf:
                return None, "無法計算 FCF (數據為負或缺失)"

            # Growth Rate
            if custom_growth:
                growth_rate = custom_growth
            else:
                fwd_pe = self.info.get('forwardPE', 20.0)
                if not fwd_pe: fwd_pe = 20.0
                growth_rate = min((fwd_pe / 100) * 0.8, 0.35)
                growth_rate = max(growth_rate, 0.05)

            # Projections (5 Years)
            future_fcf = [last_fcf * ((1 + growth_rate) ** i) for i in range(1, 6)]
            
            # Terminal Value
            terminal_val = (future_fcf[-1] * 1.03) / (wacc - 0.03)
            
            # Discount
            dcf_val = sum([f / ((1+wacc)**(i+1)) for i, f in enumerate(future_fcf)])
            dcf_val += terminal_val / ((1+wacc)**5)
            
            # Equity Value
            cash = self.get_value(self.balance, ['Cash And Cash Equivalents'])
            debt = self.get_value(self.balance, ['Total Debt'])
            equity_val = dcf_val + cash - debt
            
            # Shares
            shares = self.info.get('sharesOutstanding')
            if not shares:
                # Fallback: Market Cap / Price
                mkt_cap = self.info.get('marketCap')
                price = self.info.get('currentPrice', self.info.get('regularMarketPrice'))
                if mkt_cap and price:
                    shares = mkt_cap / price
            
            if not shares:
                return None, "無法取得流通股數"

            intrinsic_value = round(equity_val / shares, 2)
            return intrinsic_value, f"OK (Growth: {growth_rate:.1%}, WACC: {wacc:.1%})"
            
        except Exception as e:
            return None, str(e)


# --- Main Layout ---
st.title(f"{stock_name} ({ticker}) 價值投資分析")

tab0, tab_lynch, tab1, tab2, tab3 = st.tabs(["技術分析", "彼得林區估值", "財務報表", "估值模型", "投資組合最佳化"])

# === Tab 0: Technical Analysis ===
with tab0:
    st.header(f"技術線圖 ({timeframe})")
    
    ohlc_df = get_ohlc(ticker, interval)
    
    if not ohlc_df.empty:
        # Check assertions
        st.write(f"資料筆數: {len(ohlc_df)} | 最新日期: {ohlc_df.index[-1]}")
        
        # Calculate Indicators
        macd, macd_signal, macd_hist = calculate_macd(ohlc_df)
        k_line, d_line = calculate_kd(ohlc_df)
        
        # Calculate Moving Averages (MA5, MA20, MA60)
        ohlc_df['MA5'] = ohlc_df['Close'].rolling(window=5).mean()
        ohlc_df['MA20'] = ohlc_df['Close'].rolling(window=20).mean()
        ohlc_df['MA60'] = ohlc_df['Close'].rolling(window=60).mean()
        
        # Subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            subplot_titles=(f'{ticker} Price (MA5/20/60)', 'MACD', 'KD'),
                            row_heights=[0.6, 0.2, 0.2])
                            
        # Row 1: Candle (Taiwan Style: Red=Up, Green=Down)
        fig.add_trace(go.Candlestick(x=ohlc_df.index,
                                     open=ohlc_df['Open'], high=ohlc_df['High'],
                                     low=ohlc_df['Low'], close=ohlc_df['Close'],
                                     increasing_line_color='#FF3333', 
                                     decreasing_line_color='#00CC00',
                                     name='K線'), row=1, col=1)
                                     
        # Add MAs
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df['MA5'], line=dict(color='#FFFF00', width=1), name='MA5'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df['MA20'], line=dict(color='#FF00FF', width=1), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df['MA60'], line=dict(color='#00FFFF', width=1), name='MA60'), row=1, col=1)
                                     
        # Row 2: MACD
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=macd, line=dict(color='blue', width=1), name='DIF'), row=2, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=macd_signal, line=dict(color='orange', width=1), name='Signal'), row=2, col=1)
        fig.add_trace(go.Bar(x=ohlc_df.index, y=macd_hist, name='MACD Hist', 
                             marker_color=np.where(macd_hist < 0, 'green', 'red')), row=2, col=1)
                             
        # Row 3: KD
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=k_line, line=dict(color='purple', width=1), name='K'), row=3, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=d_line, line=dict(color='orange', width=1, dash='dot'), name='D'), row=3, col=1)
        
        fig.add_hrect(y0=80, y1=100, row=3, col=1, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_hrect(y0=0, y1=20, row=3, col=1, fillcolor="green", opacity=0.1, line_width=0)

        # Range Breaks: Only for Intraday 
        if interval in ['5m', '15m']:
             rangebreaks = [dict(values=["sat", "sun"])]
             if ".TW" in ticker:
                 rangebreaks.append(dict(bounds=[13.5, 9], pattern="hour"))
             else:
                 rangebreaks.append(dict(bounds=[16, 9.5], pattern="hour"))
             fig.update_xaxes(rangebreaks=rangebreaks)
            
        # Update Layout for Dark Theme
        fig.update_layout(
            height=800, 
            xaxis_rangeslider_visible=False, 
            showlegend=True, 
            hovermode="x unified", 
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=14), # Increased size and white color
            # Make grid subtle
            xaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
        )
        
        # Explicitly update subplot titles color
        fig.update_annotations(font=dict(color="white", size=16))


        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning(f"無法取得股價資料 (Interval: {interval})。")

# === MACD Signals Alert (Above Chart) ===
    with tab0:
         # Only run if we have data
         if not ohlc_df.empty:
             signals = analyze_macd_signals(ohlc_df)
             
             if signals:
                 st.write("### 🔔 技術指標快報 (AI Signal)")
                 for sig in signals:
                     days = sig.get('days_ago', 0)
                     date_str = sig.get('date', '')
                     
                     # Format Message
                     if days == 0:
                         time_msg = "(今日發生)"
                         if sig['type'] == 'Bullish':
                             st.success(f"**{sig['name']}** {time_msg}: {sig['desc']}")
                         else:
                             st.error(f"**{sig['name']}** {time_msg}: {sig['desc']}")
                     else:
                         time_msg = f"(偵測到發生於 {days} 天前 - {date_str})"
                         if sig['type'] == 'Bullish':
                             st.info(f"ℹ️ **{sig['name']}** {time_msg}")
                         else:
                             st.warning(f"⚠️ **{sig['name']}** {time_msg}")
             else:
                 st.info("ℹ️ 目前趨勢中性，無特殊 MACD 訊號 (Neutral Trend)")

# === Tab Lynch: Peter Lynch Valuation (Equity Only) ===
    if not is_etf:
        with tab_lynch:
            st.subheader("彼得林區估值線 (Peter Lynch Fair Value)")
            st.caption("股價 vs 內在價值 (Fair Value = EPS × Multiplier)")
            
            # --- 1. Fetch & Prepare Data First (Need for median calc) ---
            pl_data_ready = False
            pl_mult_final = 15.0 # Default
            median_pe_str = "N/A"
            growth_rate_est = 15.0
            
            try:
                 # Fetch Quarterly EPS
                 q_fin = stock.quarterly_financials
                 if not q_fin.empty and 'Basic EPS' in q_fin.index:
                     eps_series = q_fin.loc['Basic EPS'].sort_index()
                     eps_series = pd.to_numeric(eps_series, errors='coerce').dropna()
                     
                     if not eps_series.empty:
                         eps_ttm = eps_series.rolling(window=4).sum().dropna()
                         
                         if not eps_ttm.empty:
                             # Fetch Price
                             pl_df = stock.history(period="5y", interval="1d")
                             if not pl_df.empty:
                                 pl_df.index = pl_df.index.tz_localize(None)
                                 
                                 # Align
                                 aligned_eps = eps_ttm.reindex(pl_df.index, method='ffill')
                                 pl_df['EPS_TTM'] = aligned_eps
                                 pl_df = pl_df.dropna(subset=['EPS_TTM'])
                                 
                                 # Calculate Historical PE
                                 # Filter out negative EPS for PE calc as they distort median
                                 valid_pe_mask = (pl_df['EPS_TTM'] > 0) & (pl_df['Close'] > 0)
                                 if valid_pe_mask.any():
                                     pl_df.loc[valid_pe_mask, 'PE'] = pl_df.loc[valid_pe_mask, 'Close'] / pl_df.loc[valid_pe_mask, 'EPS_TTM']
                                     median_pe = pl_df['PE'].median()  # Historical Median PE
                                     if pd.isna(median_pe): median_pe = 15.0
                                 else:
                                     median_pe = 15.0
                                     
                                 median_pe_str = f"{median_pe:.1f}"
                                 pl_data_ready = True
                                 
                                 # Estimate Growth (Simplified for Option B)
                                 # growth_rate_est is derived from previous steps or assume 15%
            except Exception as e:
                st.error(f"Data Fetch Error: {e}")
    
            
            pl_col1, pl_col2 = st.columns([1, 4])
            
            with pl_col1:
                st.info("參數設定")
                
                # Valuation Method Selector
                val_method = st.selectbox(
                    "估值模型 (Valuation Model)",
                    ["Historical Median PE", "PEG = 1 (Growth)", "Fixed 15x"],
                    index=0,
                    help="選擇計算合理價值的倍數來源"
                )
                
                if val_method == "Historical Median PE":
                    display_val = float(median_pe_str) if pl_data_ready else 15.0
                    st.metric("5年本益比中位數", f"{display_val:.1f}x")
                    pl_mult_tab = st.number_input("倍數微調", value=display_val, step=0.5)
                    
                elif val_method == "PEG = 1 (Growth)":
                    # User manually inputs growth expectation
                    pl_mult_tab = st.number_input("預期成長率 (Growth Rate)", value=15.0, step=0.5, help="假設合理本益比 = 成長率 (PEG=1)")
                    st.caption(f"Multiplier = {pl_mult_tab}x")
                    
                else: # Fixed
                    pl_mult_tab = st.number_input("固定倍數 (Fixed Multiplier)", value=15.0, step=1.0)
                
                pl_mult_final = pl_mult_tab
    
                st.markdown("""
                **參考倍數:**
                - **10-15x**: 穩定成長 / 成熟企業
                - **15-20x**: 標準成長股
                - **20-30x**: 高速成長股
                """)
                
            with pl_col2:
                if pl_data_ready:
                    try:
                        # Apply final multiplier
                        pl_df['Fair_Value'] = pl_df['EPS_TTM'] * pl_mult_final
                        
                        # Drop NaN
                        plot_df = pl_df.dropna(subset=['Fair_Value'])
                        
                        if not plot_df.empty:
                            # 4. Plot logic (Reused)
                            fig_pl = go.Figure()
                            
                            # Common Index
                            common_idx = plot_df.index
                            p_vals = plot_df['Close']
                            fv_vals = plot_df['Fair_Value']
                            
                            # Baseline (Price) - Invisible for Reference
                            fig_pl.add_trace(go.Scatter(x=common_idx, y=p_vals, showlegend=False, line=dict(width=0), hoverinfo='skip'))
                            
                            # Green Zone (Undervalued)
                            green_y = np.maximum(p_vals, fv_vals)
                            fig_pl.add_trace(go.Scatter(
                                x=common_idx,
                                y=green_y,
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(16, 185, 129, 0.2)', # Green
                                name='被低估 (Undervalued)',
                                hoverinfo='skip'
                            ))
                            
                            # Reset Baseline (Price)
                            fig_pl.add_trace(go.Scatter(x=common_idx, y=p_vals, showlegend=False, line=dict(width=0), hoverinfo='skip'))
    
                            # Red Zone (Overvalued): Fill Min(Price, FV) down to Price
                            red_y = np.minimum(p_vals, fv_vals)
                            fig_pl.add_trace(go.Scatter(
                                x=common_idx,
                                y=red_y,
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(239, 68, 68, 0.2)', # Red
                                name='被高估 (Overvalued)',
                                hoverinfo='skip'
                            ))
                            
                            # Actual Lines
                            fig_pl.add_trace(go.Scatter(
                                x=common_idx, y=p_vals,
                                mode='lines', name='股價 (Price)',
                                line=dict(color='#1f2937', width=2)
                            ))
                            
                            fig_pl.add_trace(go.Scatter(
                                x=common_idx, y=fv_vals,
                                mode='lines', name=f'合理價值 ({pl_mult_final:.1f}x EPS)',
                                line=dict(color='#F59E0B', width=2, dash='dash')
                            ))
    
                            fig_pl.update_layout(
                                title=dict(text=f"{ticker} 彼得林區通道 (Peter Lynch Channel)", font=dict(color='#ffffff')),
                                template='plotly_dark',
                                height=500,
                                hovermode='x unified',
                                yaxis_title=f"Price ({currency_symbol})",
                                legend=dict(orientation="h", y=1.05, font=dict(color='#ffffff')),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#ffffff', size=14),
                                xaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.2)'),
                                yaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.2)')
                            )
                            st.plotly_chart(fig_pl, use_container_width=True)
                            
                            with st.expander("📊 查看詳細 EPS 數據"):
                                st.write("Quarterly EPS (Basic):")
                                st.dataframe(eps_series.tail(12).sort_index(ascending=False), use_container_width=True)
                                st.write("TTM EPS (Rolling 4Q Sum):")
                                st.dataframe(eps_ttm.tail(12).sort_index(ascending=False), use_container_width=True)
                                        
                        else:
                            st.warning("無法對齊股價與 EPS 數據 (日期範圍可能不重疊)")
                
                    except Exception as e:
                        st.error(f"發生錯誤: {str(e)}")
                elif not pl_data_ready:
                     st.warning("數據準備不足 (EPS 或 歷史股價缺失)")
    elif is_etf:
        with tab_lynch:
            st.info("ℹ️ ETF 不適用彼得林區估值模型 (No EPS data). 請參考技術分析。")

# === Tab 1: Financials (Equity Only) ===
if not is_etf:
    with tab1:
        st.header("財務報表分析 (近 5 年)")
        try:
            financials = stock.financials.T.sort_index(ascending=True).tail(5)
            cashflow = stock.cashflow.T.sort_index(ascending=True).tail(5)
            
            col1, col2, col3 = st.columns(3)
            rev_col = next((c for c in ['Total Revenue', 'Revenue'] if c in financials.columns), None)
            eps_col = 'Basic EPS' if 'Basic EPS' in financials.columns else None
            
            fcf_series = None
            if 'Operating Cash Flow' in cashflow.columns and 'Capital Expenditure' in cashflow.columns:
                fcf_series = cashflow['Operating Cash Flow'] + cashflow['Capital Expenditure']
            elif 'Free Cash Flow' in cashflow.columns:
                fcf_series = cashflow['Free Cash Flow']
    
            with col1:
                 if rev_col: 
                     fig = px.bar(financials, x=financials.index, y=rev_col, title=f"Revenue ({currency_symbol})", template="plotly_dark")
                     fig.update_layout(
                         title_font=dict(color='#ffffff'),
                         paper_bgcolor='rgba(0,0,0,0)', 
                         plot_bgcolor='rgba(0,0,0,0)', 
                         font=dict(color='#ffffff'),
                         xaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.2)'),
                         yaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.2)')
                     )
                     st.plotly_chart(fig, use_container_width=True)
            with col2:
                 if eps_col: 
                     fig = px.line(financials, x=financials.index, y=eps_col, title=f"EPS ({currency_symbol})", markers=True, template="plotly_dark")
                     fig.update_layout(
                         title_font=dict(color='#ffffff'),
                         paper_bgcolor='rgba(0,0,0,0)', 
                         plot_bgcolor='rgba(0,0,0,0)', 
                         font=dict(color='#ffffff'),
                         xaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.2)'),
                         yaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.2)')
                     )
                     st.plotly_chart(fig, use_container_width=True)
            with col3:
                 if fcf_series is not None: 
                     fig = px.bar(x=cashflow.index, y=fcf_series, title=f"FCF ({currency_symbol})", template="plotly_dark")
                     fig.update_layout(
                         title_font=dict(color='#ffffff'),
                         paper_bgcolor='rgba(0,0,0,0)', 
                         plot_bgcolor='rgba(0,0,0,0)', 
                         font=dict(color='#ffffff'),
                         xaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.2)'),
                         yaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.2)')
                     )
                     st.plotly_chart(fig, use_container_width=True)
        except:
            st.write("暫無詳細財報數據")
elif is_etf:
    with tab1:
        st.info("ℹ️ ETF 無傳統財報數據 (No Financials).")

# === Tab 2: Valuation (Equity Only) ===
if not is_etf:
    with tab2:
        st.header("估值模型")
        
        # --- DCF Model (New Logic) ---
        with st.expander("1. 現金流折現模型 (DCF - 強化版)", expanded=True):
            st.caption("基於 WACC 與 高成長期預測模型")
            
            col_dcf_in, col_dcf_out = st.columns([1, 2])
            
            with col_dcf_in:
                st.subheader("自訂參數")
                st.subheader("自訂參數")
                
                # --- AI Smart Defaults ---
                # 1. Calc WACC
                rf = 0.04
                beta = info.get('beta', 1.2)
                if beta is None: beta = 1.2
                calc_wacc = rf + beta * (0.10 - rf)
                default_wacc = round(max(calc_wacc, 0.07) * 100, 1) # Min 7%
                
                # 2. Calc Growth (CAGR from Revenue - 3 Years)
                # Logic: (Rev_current / Rev_3yrs_ago)^(1/3) - 1
                default_growth = 10.0 # Fallback
                growth_msg = "⚠️ Default setting (data missing)"
                
                try:
                    # Use financials (Revenue) primarily
                    # financials cols are dates, ascending (oldest -> newest)
                    if not financials.empty and 'Total Revenue' in financials.columns:
                         rev_series = financials['Total Revenue']
                    elif not financials.empty and 'Revenue' in financials.columns:
                         rev_series = financials['Revenue']
                    else:
                         rev_series = None
    
                    if rev_series is not None and len(rev_series) >= 4:
                         # Get Current (last) and 3 years ago (last-3)
                         # Index: ... [3 ago], [2 ago], [1 ago], [Current]
                         rev_current = rev_series.iloc[-1]
                         rev_3y_ago = rev_series.iloc[-4] # pandas df is naturally sorted by date asc here? verify
                         # Actually earlier code says: financials = stock.financials.T.sort_index(ascending=True)
                         # So yes: -1 is latest, -4 is 3 years prior to latest.
                         
                         if rev_current > 0 and rev_3y_ago > 0:
                             cagr = ((rev_current / rev_3y_ago) ** (1/3) - 1) * 100
                             
                             # Cap 5% - 30%
                             default_growth = min(max(cagr, 5.0), 30.0)
                             default_growth = round(default_growth, 1)
                             growth_msg = f"📊 Auto-calculated based on 3-year historical Revenue CAGR ({cagr:.1f}%)"
                    
                    # Check 3 years duration if only 3 items exist?
                    elif rev_series is not None and len(rev_series) == 3:
                         # Fallback to 2-year CAGR if only 3 years data
                         rev_current = rev_series.iloc[-1]
                         rev_2y_ago = rev_series.iloc[0]
                         if rev_current > 0 and rev_2y_ago > 0:
                             cagr = ((rev_current / rev_2y_ago) ** (1/2) - 1) * 100
                             default_growth = min(max(cagr, 5.0), 30.0)
                             default_growth = round(default_growth, 1)
                             growth_msg = f"📊 Auto-calculated based on 2-year historical Revenue CAGR ({cagr:.1f}%)"
                         
                except Exception as e:
                    # print(e)
                    pass
    
                custom_growth = st.slider("預期成長率 (Growth)", 0.0, 50.0, default_growth, 0.5) / 100
                custom_discount = st.slider("WACC / 折現率", 5.0, 20.0, default_wacc, 0.1) / 100
                
                st.caption(f"🤖 AI Suggestion: Based on Beta {beta}, recommended WACC is {default_wacc}%.")
                st.caption(growth_msg)
                
            with col_dcf_out:
                st.subheader("估值結果")
                
                valuator = DCFValuator(stock)
                intrinsic_val, status = valuator.perform_valuation(custom_growth=custom_growth, custom_discount=custom_discount)
                
                if intrinsic_val:
                    st.metric("內在價值 (Intrinsic Value)", f"{currency_symbol}{intrinsic_val}")
                    st.success(f"計算成功: {status}")
                    
                    # Check vs Market Price
                    curr_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    if curr_price:
                        diff = (intrinsic_val - curr_price) / curr_price * 100
                        st.caption(f"現價: {currency_symbol}{curr_price} | 潛在漲幅: {diff:.2f}%")
                        
                        # Analyst Target
                        target_mean = info.get('targetMeanPrice')
                        if target_mean:
                             st.write(f"華爾街平均目標價: {currency_symbol}{target_mean}")
                             if curr_price < target_mean * 0.8:
                                 st.success("目前價格顯著低於華爾街目標價 (>20%)")
                else:
                    st.error(f"計算失敗: {status}")
    
        # --- Reverse DCF (New Feature) ---
        with st.expander("1.5 反向 DCF 分析 (Reverse DCF)", expanded=False):
            st.caption("🔍 推算市場隱含的成長率 (Implied Growth Rate) - 檢查目前股價是否已過度反應")
            
            rdcf_col1, rdcf_col2 = st.columns(2)
            
            with rdcf_col1:
                # Inputs
                r_curr_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0.0
                r_eps_ttm = info.get('trailingEps') or 0.0
                
                # Identify a starting FCF per share if possible, else use EPS
                r_fcf_per_share = r_eps_ttm # Default to EPS
                
                # Try to get FCF
                try:
                    cf = stock.cashflow
                    if not cf.empty:
                        # Get latest year
                        latest_cf = cf.iloc[:, 0] # Most recent
                        ocf = latest_cf.get('Operating Cash Flow', 0)
                        capex = latest_cf.get('Capital Expenditure', 0)
                        fcf = ocf + capex
                        shares = info.get('sharesOutstanding')
                        if shares and shares > 0:
                            r_fcf_per_share = fcf / shares
                except: pass
                
                st.subheader("參數")
                in_price = st.number_input("目前股價 (Price)", value=float(r_curr_price), step=1.0)
                in_metric = st.number_input("每股盈餘/現金流 (EPS/FCF)", value=float(r_eps_ttm), step=0.1, help="預設為 EPS，可自行改為 FCF per Share")
                
                # --- Robust Sync: Update slider if main WACC changes ---
                if 'last_main_wacc' not in st.session_state:
                    st.session_state.last_main_wacc = custom_discount
                
                # Sync if main WACC changed
                if st.session_state.last_main_wacc != custom_discount:
                    st.session_state.rdcf_disc_unlocked = float(custom_discount)
                    st.session_state.last_main_wacc = custom_discount
    
                # --- Sanitization: Fix Decimal/Percent Mismatch ---
                # If the stored state is < 1.0 (e.g., 0.16), it's likely a decimal error. Convert to % (16.0).
                # The slider expects 5.0 to 20.0.
                current_state_val = st.session_state.get('rdcf_disc_unlocked', float(custom_discount))
                if current_state_val < 1.0:
                     st.session_state.rdcf_disc_unlocked = current_state_val * 100.0
                     st.rerun() # Rerun to apply the fixed value to the slider immediately
    
                # --- Unlocked Slider (Default to Main WACC) ---
                # Key ensures persistence, sync ensures default updates.
                in_discount = st.slider("折現率 (Discount Rate)", 5.0, 20.0, float(custom_discount), 0.1, key="rdcf_disc_unlocked") / 100.0
                
                in_term_growth = st.number_input("終端成長率 (Terminal Growth)", value=0.03, step=0.005, max_value=0.05)
            with rdcf_col2:
                st.subheader("推算結果")
                
                # 1. Robustness Check: Negative FCF/EPS
                if in_metric <= 0:
                    st.warning("⚠️ 無法計算：輸入的 EPS 或 FCF 為負值或 0。")
                    st.info("反向 DCF 需要正向的現金流才能推算成長率。")
                elif in_price <= 0:
                    st.warning("⚠️ 無法計算：目前股價無效。")
                else:
                    def dcf_objective(g):
                        # DCF Formula
                        # Sum (Metric * (1+g)^t / (1+r)^t) for t=1..5
                        # + Terminal / (1+r)^5
                        # Terminal = Metric * (1+g)^5 * (1+tg) / (r - tg)
                        
                        if in_discount <= in_term_growth: return -999999 # Invalid inputs for formula
                        
                        values = [in_metric * ((1 + g) ** i) for i in range(1, 6)]
                        
                        terminal_val = (values[-1] * (1 + in_term_growth)) / (in_discount - in_term_growth)
                        
                        dcf_sum = sum([v / ((1 + in_discount) ** (i + 1)) for i, v in enumerate(values)])
                        dcf_sum += terminal_val / ((1 + in_discount) ** 5)
                        
                        return dcf_sum - in_price
    
                    try:
                        # Solve for g
                        # Wided bounds: -50% to 300% (some high growth stocks imply >100%)
                        implied_g = optimize.brentq(dcf_objective, -0.5, 3.0)
                        
                        st.metric("市場隱含成長率 (Implied Growth)", f"{implied_g:.2%}")
                        st.write(f"以目前股價 **${in_price}** 計算，市場預期未來年化成長率為 **{implied_g:.2%}**。")
                        
                        if implied_g > 0.40:
                            st.error("⚠️ 市場預期極度樂觀 (>40%)，需有爆炸性成長支撐。")
                        elif implied_g > 0.20:
                            st.warning("🔥 市場預期高度成長 (20%~40%)。")
                        elif implied_g > 0.05:
                            st.success("✅ 市場預期合理 (5%~20%)。")
                        else:
                            st.info("📉 市場預期保守 (<5%)。")
                            
                    except ValueError:
                        st.error("⚠️ 無法求解 (Implied Growth is too extreme)")
                        st.caption("可能原因：股價相對於 EPS/FCF 過高(成長率 >300%) 或 過低(成長率 <-50%)。")
    
elif is_etf:
    with tab2:
        st.info("ℹ️ ETF 不適用現金流折現模型 (No Cash Flow data).")


# === Tab 3: Efficient Frontier & Portfolio Optimization ===
with tab3:
    st.header("⚖️ 效率前緣與投資組合最佳化")
    st.caption("基於現代投資組合理論 (MPT) 與蒙地卡羅模擬")

    col_input, col_chart = st.columns([1, 2])

    with col_input:
        with st.expander("🛠️ 參數設定", expanded=True):
            # --- Custom Ticker Addition ---
            if 'portfolio_tickers' not in st.session_state:
                st.session_state.portfolio_tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'SPY', 'QQQ', 'TLT', 'GLD']
            
            # Allow user to add new tickers
            new_ticker = st.text_input("➕ 新增自選股 (輸入代碼後按 Enter)", placeholder="例如: AMD, INTC, 2330.TW").upper().strip()
            if new_ticker:
                if new_ticker not in st.session_state.portfolio_tickers:
                    st.session_state.portfolio_tickers.insert(0, new_ticker)
                    # Use rerun to update the multiselect options immediately
                    st.rerun()

            # Ensure current page ticker is in the list
            if ticker not in st.session_state.portfolio_tickers:
                 st.session_state.portfolio_tickers.insert(0, ticker)

            # Ticker Selector
            selected_tickers = st.multiselect(
                "選擇資產池 (至少 2 檔)", 
                options=st.session_state.portfolio_tickers,
                default=st.session_state.portfolio_tickers[:5]
            )
            
            start_date = st.date_input("開始日期", value=pd.to_datetime('2023-01-01'))
            end_date = st.date_input("結束日期", value=datetime.now())
            
            risk_free_rate = st.number_input("無風險利率 (Risk Free Rate)", value=0.04, step=0.005, format="%.3f")

        st.divider()
        
        # Holdings Input
        st.subheader("目前持倉 (股數)")
        current_holdings = {}
        with st.container(height=300): # Scrollable container for many assets
            for t in selected_tickers:
                current_holdings[t] = st.number_input(f"{t} 持股數", min_value=0.0, value=0.0, step=1.0, key=f"hold_{t}")

        run_opt = st.button("🚀 開始最佳化計算", type="primary", use_container_width=True)

    if run_opt:
        if len(selected_tickers) < 2:
            st.error("請至少選擇 2 檔資產進行最佳化。")
        else:
            with st.spinner("正在下載數據並進行運算..."):
                # 1. Fetch Data
                df_prices = po.fetch_stock_data(selected_tickers, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                
                if df_prices.empty:
                    st.error("無法取得股價數據，請檢查代碼或日期範圍。")
                else:
                    # Check for dropped tickers
                    returned_tickers = df_prices.columns.tolist()
                    missing_tickers = list(set(selected_tickers) - set(returned_tickers))
                    if missing_tickers:
                        st.warning(f"⚠️ 以下代碼無數據或已被移除: {', '.join(missing_tickers)}")
                    
                    if len(returned_tickers) < 2:
                         st.error("有效資產少於 2 檔，無法進行最佳化。")
                    else:
                        # 2. Metrics
                        mu, S = po.calculate_metrics(df_prices)
                    
                    if mu is None:
                        st.error("數據不足以計算回報率 (可能含有無效數據)。")
                    else:
                        # 3. Optimize
                        opt_results = po.optimize_portfolio(mu, S, risk_free_rate)
                        
                        # 4. Efficient Frontier
                        frontier_ret, frontier_vol, frontier_weights = po.get_efficient_frontier(mu, S, risk_free_rate)
                        
                        # 5. Current Portfolio Stats
                        current_shares = np.array([current_holdings[t] for t in selected_tickers])
                        current_prices = df_prices.iloc[-1].values
                        current_val = np.sum(current_shares * current_prices)
                        
                        current_perf = None
                        if current_val > 0:
                            current_weights = (current_shares * current_prices) / current_val
                            current_perf = po.get_ret_vol_sr(current_weights, mu, S, risk_free_rate)

                        # --- Visualization ---
                        with col_chart:
                            # Scatter Plot
                            fig_ef = go.Figure()

                            # Frontier Line
                            fig_ef.add_trace(go.Scatter(x=frontier_vol, y=frontier_ret, mode='lines', name='效率前緣', line=dict(color='cyan', width=2)))

                            # Points
                            max_sr = opt_results['max_sharpe']['metrics']
                            min_vol = opt_results['min_vol']['metrics']
                            
                            fig_ef.add_trace(go.Scatter(x=[max_sr[1]], y=[max_sr[0]], mode='markers', name='最大夏普比率', marker=dict(color='gold', size=15, symbol='star')))
                            fig_ef.add_trace(go.Scatter(x=[min_vol[1]], y=[min_vol[0]], mode='markers', name='最小波動率', marker=dict(color='lightgreen', size=12, symbol='triangle-up')))
                            
                            if current_perf is not None:
                                fig_ef.add_trace(go.Scatter(x=[current_perf[1]], y=[current_perf[0]], mode='markers', name='目前投資組合', marker=dict(color='red', size=12, symbol='diamond')))

                            fig_ef.update_layout(
                                title=dict(text="效率前緣 (Efficient Frontier)", font=dict(color='#ffffff')),
                                xaxis_title="年化波動率 (Risk)",
                                yaxis_title="年化報酬率 (Return)",
                                template="plotly_dark",
                                height=600,
                                font=dict(color='#ffffff'),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.2)'),
                                yaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.2)'),
                                legend=dict(font=dict(color='#ffffff'))
                            )
                            st.plotly_chart(fig_ef, use_container_width=True)
                            
                            # --- Metrics & Rebalancing ---
                            st.divider()
                            
                            # Compare Metrics
                            metrics_data = {
                                "Metric": ["Annual Return", "Annual Volatility", "Sharpe Ratio"],
                                "Max Sharpe": [f"{max_sr[0]:.1%}", f"{max_sr[1]:.1%}", f"{max_sr[2]:.2f}"],
                                "Min Volatility": [f"{min_vol[0]:.1%}", f"{min_vol[1]:.1%}", f"{min_vol[2]:.2f}"]
                            }
                            if current_perf is not None:
                                metrics_data["Your Portfolio"] = [f"{current_perf[0]:.1%}", f"{current_perf[1]:.1%}", f"{current_perf[2]:.2f}"]
                                
                            st.subheader("📊 績效指標比較")
                            st.dataframe(pd.DataFrame(metrics_data).set_index("Metric"), use_container_width=True)
                            
                            # Rebalancing Plan (Buy Only)
                            st.subheader("🔄 再平衡建議 (Buy Only Strategy)")
                            st.caption("此策略假設「不賣出」任何持股，僅透過「加碼」來達成目標配置 (Max Sharpe)。")
                            
                            if current_val > 0:
                                target_w = opt_results['max_sharpe']['weights']
                                cash_needed, buy_shares, target_vals = po.calculate_buy_only_rebalancing(current_shares, target_w, current_prices)
                                
                                rebal_df = pd.DataFrame({
                                    "Ticker": selected_tickers,
                                    "Current Shares": current_shares,
                                    "Current Value": current_shares * current_prices,
                                    "Target Weight": [f"{w:.1%}" for w in target_w],
                                    "Shares to Buy": [f"{s:.2f}" for s in buy_shares],
                                    "Est. Cost": buy_shares * current_prices
                                })
                                
                                st.dataframe(rebal_df, use_container_width=True)
                                st.success(f"💰 預估總資金需求: ${cash_needed:,.2f}")
                            else:
                                st.info("請輸入目前持股以產生再平衡建議。")
