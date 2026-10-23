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
        # Ignores exact crossovers; triggers constantly based on current momentum state
        df['Buy_Condition'] = df['Fisher'] > df['Fisher_Signal']
        df['Sell_Condition'] = df['Fisher'] < df['Fisher_Signal']
        df.loc[df['Buy_Condition'], 'Signal_Type'] = "⚡ Continuous Upward Momentum"
        df.loc[df['Sell_Condition'], 'Signal_Type'] = "⚡ Continuous Downward Momentum"
        
    elif mode == "Scalper (Crossovers)":
        # Requires the exact minute the lines cross
        df['Buy_Condition'] = (df['Fisher'] > df['Fisher_Signal']) & (df['Fisher'].shift(1) <= df['Fisher_Signal'].shift(1))
        df['Sell_Condition'] = (df['Fisher'] < df['Fisher_Signal']) & (df['Fisher'].shift(1) >= df['Fisher_Signal'].shift(1))
        df.loc[df['Buy_Condition'], 'Signal_Type'] = "⚡ Pure Momentum Scalp (Buy)"
        df.loc[df['Sell_Condition'], 'Signal_Type'] = "⚡ Pure Momentum Scalp (Sell)"
        
    else:
        # Institutional Strict Mode
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
    c1, c2
