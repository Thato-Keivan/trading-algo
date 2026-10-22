import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import math

st.set_page_config(page_title="Algo Trading Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .metric-card { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; text-align: center; }
    .buy-signal { background-color: rgba(46, 160, 67, 0.15); border-left: 5px solid #2ea043; padding: 15px; border-radius: 4px; color: #3fb950; font-weight: bold; }
    .sell-signal { background-color: rgba(248, 81, 73, 0.15); border-left: 5px solid #f85149; padding: 15px; border-radius: 4px; color: #f85149; font-weight: bold; }
    .neutral-signal { background-color: rgba(139, 148, 158, 0.15); border-left: 5px solid #8b949e; padding: 15px; border-radius: 4px; color: #8b949e; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

ASSETS = {
    "BTC/USD": {"ticker": "BTC-USD", "pip_value": 1.0, "type": "Crypto"},
    "XAU/USD": {"ticker": "GC=F", "pip_value": 10.0, "type": "Commodity"}
}

# --- Technical Indicators ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def calculate_fisher(data, length=9):
    high = data.rolling(length).max()
    low = data.rolling(length).min()
    value = 2.0 * ((data - low) / (high - low + 1e-10)) - 1.0
    value = np.clip(value, -0.999, 0.999)
    fish_list = []
    for i in range(len(value)):
        v = value.iloc[i]
        f = 0.5 * np.log((1.0 + v) / (1.0 - v)) if i == 0 else 0.5 * np.log((1.0 + v) / (1.0 - v)) + 0.5 * fish_list[-1]
        fish_list.append(f)
    fish_series = pd.Series(fish_list, index=data.index)
    return fish_series, fish_series.shift(1)

def generate_signals(df):
    df = df.copy()
    df['EMA_Bullish'] = df['EMA_21'] > df['EMA_50']
    df['Fisher_Cross_Up'] = (df['Fisher'] > df['Fisher_Signal']) & (df['Fisher'].shift(1) <= df['Fisher_Signal'].shift(1))
    df['Fisher_Cross_Down'] = (df['Fisher'] < df['Fisher_Signal']) & (df['Fisher'].shift(1) >= df['Fisher_Signal'].shift(1))
    
    df['Signal_Score'] = np.where(df['EMA_Bullish'], 1, -1) + np.where(df['Fisher_Cross_Up'], 1, 0) - np.where(df['Fisher_Cross_Down'], 1, 0)
    
    latest_score = df['Signal_Score'].iloc[-1]
    if latest_score >= 2 and df['Fisher_Cross_Up'].iloc[-1]: return df, "STRONG BUY (Long Trigger)", "buy-signal"
    if latest_score <= -2 and df['Fisher_Cross_Down'].iloc[-1]: return df, "STRONG SELL (Short Trigger)", "sell-signal"
    if latest_score > 0: return df, "BULLISH BIAS (Looking for Entries)", "buy-signal"
    if latest_score < 0: return df, "BEARISH BIAS (Looking for Entries)", "sell-signal"
    return df, "NEUTRAL (Consolidation)", "neutral-signal"

@st.cache_data(ttl=60)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df

# --- Sidebar ---
st.sidebar.header("🎛️ Terminal Controls")
selected_asset = st.sidebar.selectbox("Asset Pair", list(ASSETS.keys()))
asset_info = ASSETS[selected_asset]

tf_map = {"1 Hour": ("60d", "1h"), "4 Hours": ("1y", "4h"), "Daily": ("max", "1d")}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_map.keys()))
period, interval = tf_map[selected_tf]

st.sidebar.divider()
st.sidebar.subheader("🧮 Position Calculator")
account_balance = st.sidebar.number_input("Account Balance ($)", value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
atr_multiplier = st.sidebar.number_input("ATR Stop Multiplier", value=1.5, step=0.1)

# --- Data Processing ---
df = load_data(asset_info["ticker"], period, interval)
if df.empty: st.stop()

df['EMA_21'] = calculate_ema(df['Close'], 21)
df['EMA_50'] = calculate_ema(df['Close'], 50)
df['ATR'] = calculate_atr(df)
df['Fisher'], df['Fisher_Signal'] = calculate_fisher(df['Close'])
df, action, css_class = generate_signals(df)

latest_price = df['Close'].iloc[-1]
current_atr = df['ATR'].iloc[-1]
stop_distance = current_atr * atr_multiplier
risk_amount = account_balance * (risk_pct / 100)
position_size = risk_amount / stop_distance if stop_distance > 0 else 0

# --- Dashboard UI ---
st.title(f"⚡ {selected_asset} Algorithmic Terminal")
st.markdown(f"<div class='{css_class}'><h3>🤖 AI Trigger: {action}</h3></div>", unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Price", f"${latest_price:,.2f}")
c2.metric("ATR (Volatility)", f"${current_atr:,.2f}")
c3.metric("Suggested Stop Distance", f"${stop_distance:,.2f}")
c4.metric("Suggested Position Size", f"{position_size:,.4f} Units")

st.divider()

# --- Charting ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25], subplot_titles=["Price & Trend", "Fisher Momentum"])

fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name="EMA 21", line=dict(color="#ffa726")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name="EMA 50", line=dict(color="#29b6f6")), row=1, col=1)

buy_sigs = df[df['Fisher_Cross_Up'] & (df['Signal_Score'] > 0)]
sell_sigs = df[df['Fisher_Cross_Down'] & (df['Signal_Score'] < 0)]
fig.add_trace(go.Scatter(x=buy_sigs.index, y=buy_sigs['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00e676'), name="Buy"), row=1, col=1)
fig.add_trace(go.Scatter(x=sell_sigs.index, y=sell_sigs['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff1744'), name="Sell"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['Fisher'], name="Fisher", line=dict(color="#26a69a")), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Fisher_Signal'], name="Signal", line=dict(color="#ef5350")), row=2, col=1)

fig.update_layout(template="plotly_dark", height=700, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
