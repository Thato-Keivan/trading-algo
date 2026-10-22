import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import time

st.set_page_config(page_title="Institutional Trading Terminal", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    .stSelectbox, .stNumberInput, .stSlider, .stCheckbox { background-color: #1c1c1c; }
    .signal-buy { color: #00FF00; font-weight: bold; font-size: 1.1rem; background: #003300; padding: 15px; border-radius: 5px; text-align: center; border: 1px solid #00FF00;}
    .signal-sell { color: #FF0000; font-weight: bold; font-size: 1.1rem; background: #330000; padding: 15px; border-radius: 5px; text-align: center; border: 1px solid #FF0000;}
    .signal-neutral { color: #888888; font-weight: bold; font-size: 1.1rem; background: #222222; padding: 15px; border-radius: 5px; text-align: center;}
    </style>
""", unsafe_allow_html=True)

ASSETS = {"BTC/USD": "BTC-USD", "XAU/USD (Gold)": "GC=F"}

@st.cache_data(ttl=10)
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
    
    # Align Timezone to New York (Required for 9 AM CR Model)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
        
    # Baseline Indicators (EMA & ATR)
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    ranges = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()
    
    # Fisher Transform Oscillator
    high, low = df['Close'].rolling(9).max(), df['Close'].rolling(9).min()
    val = np.clip(2.0 * ((df['Close'] - low) / (high - low + 1e-10)) - 1.0, -0.999, 0.999)
    fish = [0.5 * np.log((1.0 + val.iloc[0]) / (1.0 - val.iloc[0]))]
    for v in val.iloc[1:]: fish.append(0.5 * np.log((1.0 + v) / (1.0 - v)) + 0.5 * fish[-1])
    df['Fisher'] = pd.Series(fish, index=df.index)
    df['Fisher_Signal'] = df['Fisher'].shift(1)
    
    # --- SMART MONEY CONCEPTS (SMC) LOGIC ---
    
    # 1. Candle Range Theory (CRT) Liquidity Sweeps
    # Bullish: Sweeps the previous candle's low, but closes back above it (Trap)
    df['CRT_Bullish'] = (df['Low'] < df['Low'].shift(1)) & (df['Close'] > df['Low'].shift(1))
    # Bearish: Sweeps the previous candle's high, but closes back below it (Trap)
    df['CRT_Bearish'] = (df['High'] > df['High'].shift(1)) & (df['Close'] < df['High'].shift(1))
    
    # 2. 9 AM Session Identification
    df['Is_9AM'] = df.index.hour == 9
    
    # 3. Strategy Confluence Engine
    # Strategy A: Standard Trend + Momentum
    std_buy = (df['EMA_21'] > df['EMA_50']) & (df['Fisher'] > df['Fisher_Signal']) & (df['Fisher'].shift(1) <= df['Fisher_Signal'].shift(1))
    std_sell = (df['EMA_21'] < df['EMA_50']) & (df['Fisher'] < df['Fisher_Signal']) & (df['Fisher'].shift(1) >= df['Fisher_Signal'].shift(1))
    
    # Strategy B: CRT Liquidity Sweep + Fisher Momentum
    crt_fisher_buy = df['CRT_Bullish'] & (df['Fisher'] > df['Fisher_Signal']) & (df['Fisher'].shift(1) <= df['Fisher_Signal'].shift(1))
    crt_fisher_sell = df['CRT_Bearish'] & (df['Fisher'] < df['Fisher_Signal']) & (df['Fisher'].shift(1) >= df['Fisher_Signal'].shift(1))
    
    # Strategy C: 9 AM CR Model (Ultimate Setup)
    model_9am_buy = df['CRT_Bullish'] & df['Is_9AM'] & (df['Fisher'] > df['Fisher_Signal'])
    model_9am_sell = df['CRT_Bearish'] & df['Is_9AM'] & (df['Fisher'] < df['Fisher_Signal'])
    
    df['Buy_Condition'] = std_buy | crt_fisher_buy | model_9am_buy
    df['Sell_Condition'] = std_sell | crt_fisher_sell | model_9am_sell
    
    df['Signal_Type'] = "NONE"
    df.loc[std_buy, 'Signal_Type'] = "Standard Trend Breakout"
    df.loc[std_sell, 'Signal_Type'] = "Standard Trend Breakout"
    df.loc[crt_fisher_buy, 'Signal_Type'] = "CRT Liquidity Sweep"
    df.loc[crt_fisher_sell, 'Signal_Type'] = "CRT Liquidity Sweep"
    df.loc[model_9am_buy, 'Signal_Type'] = "🔥 9AM CR MODEL (HIGH CONFLUENCE)"
    df.loc[model_9am_sell, 'Signal_Type'] = "🔥 9AM CR MODEL (HIGH CONFLUENCE)"
    
    return df

st.title("📈 Institutional Algorithmic Terminal (CRT + 9AM Model)")
usd_zar = get_usd_zar()

with st.expander("⚙️ Controls & Risk/Leverage Setup", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        asset_name = st.selectbox("Symbol", list(ASSETS.keys()))
        tf_map = {
            "1 Minute": ("7d", "1m"), "5 Minutes": ("60d", "5m"), 
            "15 Minutes": ("60d", "15m"), "30 Minutes": ("60d", "30m"),
            "1 Hour": ("60d", "1h"), "4 Hours": ("1y", "4h"), "Daily": ("max", "1d")
        }
        tf = st.selectbox("Timeframe", list(tf_map.keys()), index=4) # Default to 1H for CR models
    with c2:
        balance_zar = st.number_input("Account Balance (ZAR)", value=20000.0, step=1000.0)
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
    with c3:
        leverage = st.number_input("Broker Leverage (1:X)", value=100, step=50)
        auto_refresh = st.checkbox("🟢 Enable Live Auto-Refresh (60s)")
    with c4:
        atr_mult = st.slider("SL ATR Multiplier", 1.0, 3.0, 1.5, 0.1)
        rr_ratio = st.number_input("Take Profit (R:R)", value=2.0, step=0.1)

ticker, (period, interval) = ASSETS[asset_name], tf_map[tf]
df = calc_indicators(get_market_data(ticker, period, interval))
if df.empty: st.stop()

plot_df = df.tail(300)
latest = plot_df.iloc[-1]
entry = latest['Close'].item()
stop_dist = latest['ATR'].item() * atr_mult

signal, sig_class, setup_type = "NEUTRAL (Searching for Setups...)", "signal-neutral", "N/A"
sl, tp = 0.0, 0.0

if latest['Buy_Condition']:
    setup_type = latest['Signal_Type']
    signal, sig_class = f"🟢 LIVE: BUY SIGNAL TRIGGERED<br><span style='font-size: 0.9rem;'>{setup_type}</span>", "signal-buy"
    sl, tp = entry - stop_dist, entry + (stop_dist * rr_ratio)
elif latest['Sell_Condition']:
    setup_type = latest['Signal_Type']
    signal, sig_class = f"🔴 LIVE: SELL SIGNAL TRIGGERED<br><span style='font-size: 0.9rem;'>{setup_type}</span>", "signal-sell"
    sl, tp = entry + stop_dist, entry - (stop_dist * rr_ratio)

risk_usd = (balance_zar * (risk_pct / 100)) / usd_zar
pos_size = risk_usd / stop_dist if stop_dist > 0 else 0
notional_value_usd = pos_size * entry
margin_req_zar = (notional_value_usd / leverage) * usd_zar
prof_zar = (abs(entry - tp) * pos_size) * usd_zar if tp > 0 else 0

st.markdown(f"<div class='{sig_class}'>{signal}</div>", unsafe_allow_html=True)
st.write("")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Stop Loss", f"${sl:,.2f}" if sl else "-")
m2.metric("Take Profit", f"${tp:,.2f}" if tp else "-")
m3.metric("Position Units", f"{pos_size:,.4f}")
m4.metric("Margin Required", f"R {margin_req_zar:,.2f}")
m5.metric("Proj. Profit", f"R {prof_zar:,.0f}" if prof_zar else "-")

if margin_req_zar > balance_zar:
    st.error(f"⚠️ Insufficient margin! Reduce risk % or increase leverage.")

st.divider()

# Chart Rendering
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], increasing_line_color='#00FF00', increasing_fillcolor='#00FF00', decreasing_line_color='#FF0000', decreasing_fillcolor='#FF0000', name="OHLC"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_21'], line=dict(color="yellow", width=1), name="EMA 21"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_50'], line=dict(color="blue", width=1), name="EMA 50"), row=1, col=1)

# Plot arrows with hover text showing which exact strategy triggered it
buys = plot_df[plot_df['Buy_Condition']]
sells = plot_df[plot_df['Sell_Condition']]
fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00FF00'), name="BUY", hovertext=buys['Signal_Type']), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#FF0000'), name="SELL", hovertext=sells['Signal_Type']), row=1, col=1)

fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Fisher'], line=dict(color="#00FF00", width=1), name="Fisher"), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Fisher_Signal'], line=dict(color="#FF0000", width=1), name="Signal"), row=2, col=1)

fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#A0A0A0'), height=650, margin=dict(l=0, r=0, t=5, b=0), xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_xaxes(showgrid=True, gridcolor='#222222', gridwidth=1)
fig.update_yaxes(showgrid=True, gridcolor='#222222', gridwidth=1)
st.plotly_chart(fig, use_container_width=True)

if auto_refresh:
    time.sleep(60)
    st.rerun()
