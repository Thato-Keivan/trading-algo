import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

st.set_page_config(page_title="MT5 Terminal", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    .stSelectbox, .stNumberInput, .stSlider { background-color: #1c1c1c; }
    .signal-buy { color: #00FF00; font-weight: bold; font-size: 1.2rem; background: #003300; padding: 10px; border-radius: 5px;}
    .signal-sell { color: #FF0000; font-weight: bold; font-size: 1.2rem; background: #330000; padding: 10px; border-radius: 5px;}
    .signal-neutral { color: #888888; font-weight: bold; font-size: 1.2rem; background: #222222; padding: 10px; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

ASSETS = {"BTC/USD": "BTC-USD", "XAU/USD (Gold)": "GC=F"}

@st.cache_data(ttl=60)
def get_market_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=300)
def get_usd_zar():
    zar = yf.download("ZAR=X", period="1d", interval="1d", progress=False)
    return zar['Close'].iloc[-1].item() if not zar.empty else 18.00

def calc_indicators(df):
    df = df.copy()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    ranges = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()
    
    high, low = df['Close'].rolling(9).max(), df['Close'].rolling(9).min()
    val = np.clip(2.0 * ((df['Close'] - low) / (high - low + 1e-10)) - 1.0, -0.999, 0.999)
    fish = [0.5 * np.log((1.0 + val.iloc[0]) / (1.0 - val.iloc[0]))]
    for v in val.iloc[1:]: fish.append(0.5 * np.log((1.0 + v) / (1.0 - v)) + 0.5 * fish[-1])
    
    df['Fisher'] = pd.Series(fish, index=df.index)
    df['Fisher_Signal'] = df['Fisher'].shift(1)
    
    # Signal Logic (1 = Buy, -1 = Sell)
    df['Buy_Condition'] = (df['EMA_21'] > df['EMA_50']) & (df['Fisher'] > df['Fisher_Signal']) & (df['Fisher'].shift(1) <= df['Fisher_Signal'].shift(1))
    df['Sell_Condition'] = (df['EMA_21'] < df['EMA_50']) & (df['Fisher'] < df['Fisher_Signal']) & (df['Fisher'].shift(1) >= df['Fisher_Signal'].shift(1))
    
    return df

st.title("📈 MQL5-Style Trading Terminal")
usd_zar = get_usd_zar()

with st.expander("⚙️ Controls & Settings", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        asset_name = st.selectbox("Symbol", list(ASSETS.keys()))
        tf_map = {
            "1 Minute": ("7d", "1m"), "5 Minutes": ("60d", "5m"), 
            "15 Minutes": ("60d", "15m"), "30 Minutes": ("60d", "30m"),
            "1 Hour": ("60d", "1h"), "4 Hours": ("1y", "4h"), "Daily": ("max", "1d")
        }
        tf = st.selectbox("Timeframe", list(tf_map.keys()))
    with c2:
        balance_zar = st.number_input("Account Balance (ZAR)", value=20000.0, step=1000.0)
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
    with c3:
        atr_mult = st.slider("Stop Loss Looseness (ATR)", 1.0, 3.0, 1.5, 0.1)
        rr_ratio = st.number_input("Take Profit Multiplier", value=2.0, step=0.1)

ticker, (period, interval) = ASSETS[asset_name], tf_map[tf]
df = calc_indicators(get_market_data(ticker, period, interval))
if df.empty: st.stop()

# Performance Optimization: Only plot the last 300 candles to eliminate lag
plot_df = df.tail(300)

latest = plot_df.iloc[-1]
entry = latest['Close'].item()
stop_dist = latest['ATR'].item() * atr_mult

# Live Signal Detection
signal, sig_class = "NEUTRAL (No Trade)", "signal-neutral"
sl, tp = 0.0, 0.0

if latest['Buy_Condition']:
    signal, sig_class = "🟢 BUY SIGNAL", "signal-buy"
    sl, tp = entry - stop_dist, entry + (stop_dist * rr_ratio)
elif latest['Sell_Condition']:
    signal, sig_class = "🔴 SELL SIGNAL", "signal-sell"
    sl, tp = entry + stop_dist, entry - (stop_dist * rr_ratio)

risk_usd = (balance_zar * (risk_pct / 100)) / usd_zar
pos_size = risk_usd / stop_dist if stop_dist > 0 else 0
prof_zar = (abs(entry - tp) * pos_size) * usd_zar if tp > 0 else 0

st.markdown(f"<div class='{sig_class}'>Live Status: {signal}</div>", unsafe_allow_html=True)
st.write("")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Price", f"${entry:,.2f}")
m2.metric("Required Stop Loss", f"${sl:,.2f}" if sl else "-")
m3.metric("Take Profit Target", f"${tp:,.2f}" if tp else "-")
m4.metric("Potential Profit", f"R {prof_zar:,.0f}" if prof_zar else "-")

st.divider()

# Chart Rendering
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)

fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'],
    increasing_line_color='#00FF00', increasing_fillcolor='#00FF00',
    decreasing_line_color='#FF0000', decreasing_fillcolor='#FF0000', name="OHLC"
), row=1, col=1)

fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_21'], line=dict(color="yellow", width=1), name="EMA 21"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_50'], line=dict(color="blue", width=1), name="EMA 50"), row=1, col=1)

# Explicit Buy/Sell Arrows on Chart
buys = plot_df[plot_df['Buy_Condition']]
sells = plot_df[plot_df['Sell_Condition']]
fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00FF00'), name="BUY"), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#FF0000'), name="SELL"), row=1, col=1)

fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Fisher'], line=dict(color="#00FF00", width=1), name="Fisher"), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Fisher_Signal'], line=dict(color="#FF0000", width=1), name="Signal"), row=2, col=1)

fig.update_layout(
    plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#A0A0A0'),
    height=650, margin=dict(l=0, r=0, t=5, b=0), xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_xaxes(showgrid=True, gridcolor='#222222', gridwidth=1)
fig.update_yaxes(showgrid=True, gridcolor='#222222', gridwidth=1)

st.plotly_chart(fig, use_container_width=True)
