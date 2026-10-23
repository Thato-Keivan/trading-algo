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

def calc_indicators(df, mode):
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
        
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
    
    df['Signal_Type'] = "NONE"
    
    if mode == "Always-On (Continuous)":
        df['Buy_Condition'] = df['Fisher'] > df['Fisher_Signal']
        df['Sell_Condition'] = df['Fisher'] < df['Fisher_Signal']
        df.loc[df['Buy_Condition'], 'Signal_Type'] = "⚡ Continuous Upward Momentum"
        df.loc[df['Sell_Condition'], 'Signal_Type'] = "⚡ Continuous Downward Momentum"
        
    elif mode == "Scalper (Crossovers)":
        df['Buy_Condition'] = (df['Fisher'] > df['Fisher_Signal']) & (df['Fisher'].shift(1) <= df['Fisher_Signal'].shift(1))
        df['Sell_Condition'] = (df['Fisher'] < df['Fisher_Signal']) & (df['Fisher'].shift(1) >= df['Fisher_Signal'].shift(1))
        df.loc[df['Buy_Condition'], 'Signal_Type'] = "⚡ Pure Momentum Scalp (Buy)"
        df.loc[df['Sell_Condition'], 'Signal_Type'] = "⚡ Pure Momentum Scalp (Sell)"
        
    else:
        df['CRT_Bullish'] = (df['Low'] < df['Low'].shift(1)) & (df['Close'] > df['Low'].shift(1))
        df['CRT_Bearish'] = (df['High'] > df['High'].shift(1)) & (df['Close'] < df['High'].shift(1))
        df['Is_9AM'] = df.index.hour == 9
        
        std_buy = (df['EMA_21'] > df['EMA_50']) & (df['Fisher'] > df['Fisher_Signal']) & (df['Fisher'].shift(1) <= df['Fisher_Signal'].shift(1))
        std_sell = (df['EMA_21'] < df['EMA_50']) & (df['Fisher'] < df['Fisher_Signal']) & (df['Fisher'].shift(1) >= df['Fisher_Signal'].shift(1))
        crt_fisher_buy = df['CRT_Bullish'] & (df['Fisher'] > df['Fisher_Signal'])
        crt_fisher_sell = df['CRT_Bearish'] & (df['Fisher'] < df['Fisher_Signal'])
        model_9am_buy = df['CRT_Bullish'] & df['Is_9AM'] & (df['Fisher'] > df['Fisher_Signal'])
        model_9am_sell = df['CRT_Bearish'] & df['Is_9AM'] & (df['Fisher'] < df['Fisher_Signal'])
        
        df['Buy_Condition'] = std_buy | crt_fisher_buy | model_9am_buy
        df['Sell_Condition'] = std_sell | crt_fisher_sell | model_9am_sell
        
        df.loc[std_buy | std_sell, 'Signal_Type'] = "Standard Trend Breakout"
        df.loc[crt_fisher_buy | crt_fisher_sell, 'Signal_Type'] = "CRT Liquidity Sweep"
        df.loc[model_9am_buy | model_9am_sell, 'Signal_Type'] = "🔥 9AM CR MODEL"

    return df

st.title("📈 Multi-Mode Algorithmic Terminal")
usd_zar = get_usd_zar()

with st.expander("⚙️ Controls, Strategy & Risk Setup", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        asset_name = st.selectbox("Symbol", list(ASSETS.keys()))
        tf_map = {
            "1 Minute": ("7d", "1m"), "5 Minutes": ("60d", "5m"), 
            "15 Minutes": ("60d", "15m"), "30 Minutes": ("60d", "30m"),
            "1 Hour": ("60d", "1h"), "4 Hours": ("1y", "4h"), "Daily": ("max", "1d")
        }
        tf = st.selectbox("Timeframe", list(tf_map.keys()), index=0)
    with c2:
        strategy_mode = st.selectbox("Strategy Mode", ["Institutional (Strict)", "Scalper (Crossovers)", "Always-On (Continuous)"], index=2)
        balance_zar = st.number_input("Account Balance (ZAR)", value=20000.0, step=1000.0)
    with c3:
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
        leverage = st.number_input("Broker Leverage (1:X)", value=100, step=50)
    with c4:
        atr_mult = st.slider("SL ATR Multiplier", 1.0, 3.0, 1.5, 0.1)
        rr_ratio = st.number_input("Take Profit (R:R)", value=2.0, step=0.1)
        auto_refresh = st.checkbox("🟢 Live Auto-Refresh (60s)", value=True)

ticker, (period, interval) = ASSETS[asset_name], tf_map[tf]
df = calc_indicators(get_market_data(ticker, period, interval), strategy_mode)
if df.empty: st.stop()

plot_df = df.tail(300)
closed_candle = plot_df.iloc[-2]
current_px = plot_df['Close'].iloc[-1]
stop_dist = closed_candle['ATR'].item() * atr_mult

trend_dir = "UP" if closed_candle['EMA_21'] > closed_candle['EMA_50'] else "DOWN"
mom_dir = "BULLISH" if closed_candle['Fisher'] > closed_candle['Fisher_Signal'] else "BEARISH"
diag_status = f"Trend: {trend_dir} | Momentum: {mom_dir}"

signal, sig_class, setup_type = f"NEUTRAL ({diag_status})", "signal-neutral", "N/A"
sl, tp = 0.0, 0.0

if closed_candle['Buy_Condition']:
    setup_type = closed_candle['Signal_Type']
    signal, sig_class = f"🟢 CONFIRMED BUY SIGNAL (Last Candle)<br><span style='font-size: 0.9rem;'>{setup_type}</span>", "signal-buy"
    sl, tp = current_px - stop_dist, current_px + (stop_dist * rr_ratio)
elif closed_candle['Sell_Condition']:
    setup_type = closed_candle['Signal_Type']
    signal, sig_class = f"🔴 CONFIRMED SELL SIGNAL (Last Candle)<br><span style='font-size: 0.9rem;'>{setup_type}</span>", "signal-sell"
    sl, tp = current_px + stop_dist, current_px - (stop_dist * rr_ratio)

risk_usd = (balance_zar * (risk_pct / 100)) / usd_zar
pos_size = risk_usd / stop_dist if stop_dist > 0 else 0
notional_value_usd = pos_size * current_px
margin_req_zar = (notional_value_usd / leverage) * usd_zar if leverage > 0 else 0
prof_zar = (abs(current_px - tp) * pos_size) * usd_zar if tp > 0 else 0

st.markdown(f"<div class='{sig_class}'>{signal}</div>", unsafe_allow_html=True)
st.write("")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Stop Loss", f"${sl:,.2f}" if sl else "-")
m2.metric("Take Profit", f"${tp:,.2f}" if tp else "-")
m3.metric("Position Units", f"{pos_size:,.4f}")
m4.metric("Margin Required", f"R {margin_req_zar:,.2f}")
m5.metric("Proj. Profit", f"R {prof_zar:,.0f}" if prof_zar else "-")

st.divider()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], increasing_line_color='#00FF00', increasing_fillcolor='#00FF00', decreasing_line_color='#FF0000', decreasing_fillcolor='#FF0000', name="OHLC"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_21'], line=dict(color="yellow", width=1), name="EMA 21"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_50'], line=dict(color="blue", width=1), name="EMA 50"), row=1, col=1)

buys = plot_df[plot_df['Buy_Condition']]
sells = plot_df[plot_df['Sell_Condition']]
fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00FF00'), name="BUY", hovertext=buys['Signal_Type']), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#FF0000'), name="SELL", hovertext=sells['Signal_Type']), row=1, col=1)

fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Fisher'], line=dict(color="#00FF00", width=1), name="Fisher"), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Fisher_Signal'], line=dict(color="#FF0000", width=1), name="Signal"), row=2, col=1)

fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#A0A0A0'), height=550, margin=dict(l=0, r=0, t=5, b=0), xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 📝 24-Hour Historical Signal Log")
cutoff_time = df.index[-1] - pd.Timedelta(hours=24)
log_df = df[(df.index >= cutoff_time) & ((df['Buy_Condition']) | (df['Sell_Condition']))].copy()

if not log_df.empty:
    log_df['Direction'] = np.where(log_df['Buy_Condition'], "🟢 BUY", "🔴 SELL")
    display_log = log_df[['Direction', 'Signal_Type', 'Close', 'ATR']].copy()
    display_log['Stop Loss'] = np.where(log_df['Buy_Condition'], display_log['Close'] - (display_log['ATR'] * atr_mult), display_log['Close'] + (display_log['ATR'] * atr_mult))
    display_log['Take Profit'] = np.where(log_df['Buy_Condition'], display_log['Close'] + (display_log['ATR'] * atr_mult * rr_ratio), display_log['Close'] - (display_log['ATR'] * atr_mult * rr_ratio))
    display_log = display_log[['Direction', 'Signal_Type', 'Close', 'Stop Loss', 'Take Profit']]
    display_log.columns = ['Action', 'Setup Strategy', 'Entry Price', 'Stop Loss', 'Take Profit']
    display_log = display_log.iloc[::-1]
    display_log.index = display_log.index.strftime('%Y-%m-%d %H:%M:%S (NY Time)')
    st.dataframe(display_log.style.format({'Entry Price': '${:,.2f}', 'Stop Loss': '${:,.2f}', 'Take Profit': '${:,.2f}'}), use_container_width=True)
else:
    st.write("No setups detected in the last 24 hours for the selected timeframe and strategy mode.")

if auto_refresh:
    time.sleep(60)
    st.rerun()
