import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

st.set_page_config(page_title="MT5 Web Terminal", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# --- Custom MT5 Styling ---
st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    .stSelectbox, .stNumberInput, .stSlider { background-color: #1c1c1c; }
    .metric-container { background-color: #111111; border: 1px solid #333333; padding: 10px; border-radius: 5px; }
    .signal-buy { color: #00FF00; font-weight: bold; font-size: 1.2rem; }
    .signal-sell { color: #FF0000; font-weight: bold; font-size: 1.2rem; }
    .signal-neutral { color: #888888; font-weight: bold; font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

ASSETS = {
    "BTC/USD": {"ticker": "BTC-USD"},
    "XAU/USD (Gold)": {"ticker": "GC=F"}
}

# --- Core Functions ---
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
    # EMAs
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # ATR
    ranges = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()
    
    # Fisher Transform
    high, low = df['Close'].rolling(9).max(), df['Close'].rolling(9).min()
    val = np.clip(2.0 * ((df['Close'] - low) / (high - low + 1e-10)) - 1.0, -0.999, 0.999)
    
    fish = []
    for i, v in enumerate(val):
        f = 0.5 * np.log((1.0 + v) / (1.0 - v)) + (0.5 * fish[-1] if i > 0 else 0)
        fish.append(f)
    df['Fisher'] = pd.Series(fish, index=df.index)
    df['Fisher_Signal'] = df['Fisher'].shift(1)
    
    return df

# --- UI Header & Controls ---
st.title("📈 MQL5-Style Trading Terminal")
usd_zar = get_usd_zar()
st.caption(f"Live Exchange Rate: 1 USD = R {usd_zar:.2f}")

with st.expander("⚙️ Setup & Risk Calculator", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        asset_name = st.selectbox("Symbol", list(ASSETS.keys()))
        tf_map = {"1H": ("60d", "1h"), "4H": ("1y", "4h"), "1D": ("max", "1d")}
        tf = st.selectbox("Period", list(tf_map.keys()))
    with c2:
        balance_zar = st.number_input("Account Balance (ZAR)", value=20000.0, step=1000.0)
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
    with c3:
        atr_mult = st.number_input("SL (ATR Multiplier)", value=1.5, step=0.1)
        rr_ratio = st.number_input("Take Profit (R:R Ratio)", value=2.0, step=0.1)
    with c4:
        st.write("Calculated Risk:")
        risk_zar = balance_zar * (risk_pct / 100)
        risk_usd = risk_zar / usd_zar
        st.write(f"**R {risk_zar:,.2f}** (${risk_usd:,.2f})")

# --- Process Data & Signals ---
ticker, (period, interval) = ASSETS[asset_name]["ticker"], tf_map[tf]
df = calc_indicators(get_market_data(ticker, period, interval))
if df.empty: st.stop()

close_px = df['Close'].iloc[-1].item()
atr = df['ATR'].iloc[-1].item()
fish, fish_sig = df['Fisher'].iloc[-1], df['Fisher_Signal'].iloc[-1]
ema21, ema50 = df['EMA_21'].iloc[-1].item(), df['EMA_50'].iloc[-1].item()

# Signal Logic
signal, entry, sl, tp, sig_class = "NEUTRAL", close_px, 0.0, 0.0, "signal-neutral"
stop_dist = atr * atr_mult

if ema21 > ema50 and fish > fish_sig:
    signal, sig_class = "🟢 BUY (LONG)", "signal-buy"
    sl, tp = entry - stop_dist, entry + (stop_dist * rr_ratio)
elif ema21 < ema50 and fish < fish_sig:
    signal, sig_class = "🔴 SELL (SHORT)", "signal-sell"
    sl, tp = entry + stop_dist, entry - (stop_dist * rr_ratio)

pos_size_units = risk_usd / stop_dist if stop_dist > 0 else 0
potential_profit_usd = (abs(entry - tp) * pos_size_units) if tp > 0 else 0
potential_profit_zar = potential_profit_usd * usd_zar

# --- Dashboard Metrics ---
st.markdown(f"<div class='{sig_class}'>Action: {signal} @ ${entry:,.2f}</div>", unsafe_allow_html=True)
st.write("")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Stop Loss", f"${sl:,.2f}")
m2.metric("Take Profit", f"${tp:,.2f}")
m3.metric("Position Size", f"{pos_size_units:,.4f}")
m4.metric("Risk (ZAR)", f"R {risk_zar:,.0f}")
m5.metric("Proj. Profit (ZAR)", f"R {potential_profit_zar:,.0f}")

st.divider()

# --- MT5 Style Charting ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)

# Candlesticks
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    increasing_line_color='#00FF00', increasing_fillcolor='#00FF00',
    decreasing_line_color='#FF0000', decreasing_fillcolor='#FF0000', name="OHLC"
), row=1, col=1)

# EMAs
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], line=dict(color="yellow", width=1), name="EMA 21"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color="blue", width=1), name="EMA 50"), row=1, col=1)

# Plot Active Target Lines if signal is active
if signal != "NEUTRAL":
    fig.add_hline(y=entry, line_dash="dot", line_color="white", row=1, col=1, annotation_text="ENTRY")
    fig.add_hline(y=sl, line_dash="dash", line_color="red", row=1, col=1, annotation_text="SL")
    fig.add_hline(y=tp, line_dash="dash", line_color="green", row=1, col=1, annotation_text="TP")

# Fisher Subplot
fig.add_trace(go.Scatter(x=df.index, y=df['Fisher'], line=dict(color="#00FF00", width=1), name="Fisher"), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Fisher_Signal'], line=dict(color="#FF0000", width=1), name="Signal"), row=2, col=1)

# Apply Black MT5 Theme
fig.update_layout(
    plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#A0A0A0'),
    height=600, margin=dict(l=5, r=5, t=5, b=5), xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_xaxes(showgrid=True, gridcolor='#333333', gridwidth=0.5)
fig.update_yaxes(showgrid=True, gridcolor='#333333', gridwidth=0.5)

st.plotly_chart(fig, use_container_width=True)
