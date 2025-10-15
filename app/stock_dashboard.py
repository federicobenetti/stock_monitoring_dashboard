import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import yfinance as yf

from colors import color_of, candle_kwargs, ma_color
from kpis import (
    ret_n, ytd_ret, day_change_pct, range_pos, rsi_state, stoch_state, macd_state,
    vol_percentile_1y, bb_width_pct, curr_dd_1y, adv20_dollar, vol_zscore_today,
    ma_distances, ma_breadth, ma_slope
)

st.set_page_config(page_title="Stocks dashboard",
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide")

"""
# Stock Performance Dashboard
Review single stock performance and main indicators
"""

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_and_process_data(ticker: str, period: str = "2y"):
    """
    Fetch OHLCV data from yfinance and compute technical indicators.
    
    Args:
        ticker: Stock ticker symbol
        period: Period to fetch (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
    
    Returns:
        tuple: (main_df, processed_df) - OHLCV data and computed indicators
    """
    
    # Fetch data from yfinance
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Prepare main dataframe (OHLCV)
    main = hist.reset_index()
    main.columns = [col.lower().replace(' ', '_') for col in main.columns]
    
    # Handle different possible column names
    if 'adj_close' not in main.columns:
        if 'adjclose' in main.columns:
            main = main.rename(columns={'adjclose': 'adj_close'})
        else:
            main['adj_close'] = main['close']  # Fallback if adj_close doesn't exist
    
    main['ticker'] = ticker
    main = main[['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
    
    # Prepare processed dataframe with indicators
    df = main.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_100'] = df['close'].rolling(window=100).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_MA_20'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_UPPER_20_2'] = df['BB_MA_20'] + (2 * bb_std)
    df['BB_LOWER_20_2'] = df['BB_MA_20'] - (2 * bb_std)
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema_12 - ema_26
    df['MACD_SIGNAL_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD_12_26_9'] - df['MACD_SIGNAL_9']
    
    # RSI
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI_14'] = calculate_rsi(df['close'], 14)
    
    # Stochastic Oscillator
    def calculate_stochastic(high, low, close, k_period=14, d_period=3, smooth_k=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_smooth.rolling(window=d_period).mean()
        return k_smooth, d_percent
    
    df['STOCH_K_14_3'], df['STOCH_D_3'] = calculate_stochastic(
        df['high'], df['low'], df['close'], k_period=14, d_period=3, smooth_k=3
    )
    
    # CCI (Commodity Channel Index)
    def calculate_cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    df['CCI_20'] = calculate_cci(df['high'], df['low'], df['close'], 20)
    
    # ATR (Average True Range)
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    df['ATR_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    
    # Annualized Volatility (30D and 90D)
    df['ANNUALIZED_VOL_30D'] = df['close'].pct_change().rolling(30).std() * np.sqrt(252)
    df['ANNUALIZED_VOL_90D'] = df['close'].pct_change().rolling(90).std() * np.sqrt(252)
    
    # Max Drawdown (90D)
    def max_drawdown(series, window):
        roll_max = series.rolling(window, min_periods=1).max()
        drawdown = (series - roll_max) / roll_max
        return drawdown.rolling(window, min_periods=1).min()
    
    df['MAX_DRAWDOWN_90D'] = max_drawdown(df['close'], 90)
    
    # Signals
    # MACD Bullish Cross: MACD crosses above Signal
    df['SIG_MACD_BULLISH_CROSS'] = (
        (df['MACD_12_26_9'] > df['MACD_SIGNAL_9']) & 
        (df['MACD_12_26_9'].shift(1) <= df['MACD_SIGNAL_9'].shift(1))
    ).astype(int)
    
    # RSI Oversold (RSI < 30)
    df['SIG_RSI_OVERSOLD'] = (df['RSI_14'] < 30).astype(int)
    
    # RSI Overbought (RSI > 70)
    df['SIG_RSI_OVERBOUGHT'] = (df['RSI_14'] > 70).astype(int)
    
    # BB Breakout Up (Close > Upper Band)
    df['SIG_BB_BREAKOUT_UP'] = (df['close'] > df['BB_UPPER_20_2']).astype(int)
    
    # BB Breakout Down (Close < Lower Band)
    df['SIG_BB_BREAKOUT_DOWN'] = (df['close'] < df['BB_LOWER_20_2']).astype(int)
    
    # Select processed columns
    processed_cols = [
        'date', 'ticker', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
        'EMA_12', 'EMA_26', 'BB_MA_20', 'BB_UPPER_20_2', 'BB_LOWER_20_2',
        'MACD_12_26_9', 'MACD_SIGNAL_9', 'MACD_HIST', 'RSI_14',
        'STOCH_K_14_3', 'STOCH_D_3', 'CCI_20', 'ATR_14',
        'ANNUALIZED_VOL_30D', 'ANNUALIZED_VOL_90D', 'MAX_DRAWDOWN_90D',
        'SIG_MACD_BULLISH_CROSS', 'SIG_RSI_OVERSOLD', 'SIG_RSI_OVERBOUGHT',
        'SIG_BB_BREAKOUT_UP', 'SIG_BB_BREAKOUT_DOWN'
    ]
    
    pre = df[processed_cols].copy()
    
    # Merge main and processed to create the combined df
    combined = pd.merge(main, pre, on=['date', 'ticker'], how='left')
    
    return combined, main, pre

# --- Defaults from config ---
DEFAULTS_FALLBACK = {"ma_periods": [20, 50, 100, 200], "bb_period": 20, "bb_k": 2.0}

def load_param_defaults(path: str = "config/params.yaml") -> dict:
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            return cfg.get("defaults", DEFAULTS_FALLBACK)
    return DEFAULTS_FALLBACK

def init_params():
    d = load_param_defaults()
    st.session_state.setdefault("ma_periods", list(map(int, d["ma_periods"])))
    st.session_state.setdefault("bb_period", int(d["bb_period"]))
    st.session_state.setdefault("bb_k", float(d["bb_k"]))

def compute_overlays(frame: pd.DataFrame, ma_periods, bb_period: int, bb_k: float):
    ma_series = []
    for p in ma_periods:
        if p and p > 0:
            ma_series.append((f"MA {int(p)}", frame["close"].rolling(int(p)).mean()))
    bb_ma = frame["close"].rolling(int(bb_period)).mean()
    bb_std = frame["close"].rolling(int(bb_period)).std()
    bb_upper = bb_ma + float(bb_k) * bb_std
    bb_lower = bb_ma - float(bb_k) * bb_std
    return ma_series, bb_ma, bb_upper, bb_lower

# Initialize session state for parameters
init_params()

# ---------- Sidebar Controls ----------
st.sidebar.title("Controls")

# Preset tickers with ability to type new ones
preset_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT", 
                  "SPY", "QQQ", "DIA", "BA", "DIS", "NFLX", "INTC", "AMD", "CSCO", "ORCL"]

ticker = st.sidebar.selectbox(
    "Ticker symbol",
    preset_tickers,
    index=0,
    placeholder="Select a ticker or type a new one",
    accept_new_options=True,
)

if ticker:
    ticker = ticker.upper()

# Fetch data for selected ticker
with st.spinner(f'Fetching data for {ticker}...'):
    df, main, pre = fetch_and_process_data(ticker, period="2y")

if df is None:
    st.error(f"Unable to fetch data for ticker: {ticker}. Please check the ticker symbol.")
    st.stop()

# Date range selector (AFTER data is loaded)
date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range", (date_min, date_max), min_value=date_min, max_value=date_max)

# ---------- Parameter form (Confirm + Reset) ----------
with st.sidebar.form("params_form"):
    st.subheader("Parameters")
    c1, c2 = st.columns(2)
    ma1 = c1.number_input("MA 1", min_value=1, max_value=3650, step=1, value=int(st.session_state["ma_periods"][0]))
    ma2 = c2.number_input("MA 2", min_value=1, max_value=3650, step=1, value=int(st.session_state["ma_periods"][1]))
    ma3 = c1.number_input("MA 3", min_value=1, max_value=3650, step=1, value=int(st.session_state["ma_periods"][2]))
    ma4 = c2.number_input("MA 4", min_value=1, max_value=3650, step=1, value=int(st.session_state["ma_periods"][3]))
    bb_period = st.number_input("BB period", min_value=1, max_value=3650, step=1, value=int(st.session_state["bb_period"]))
    bb_k = st.number_input("BB std dev (k)", min_value=0.1, max_value=10.0, step=0.1, format="%.1f", value=float(st.session_state["bb_k"]))

    b1, b2 = st.columns(2)
    confirm = b1.form_submit_button("Confirm", use_container_width=True)
    reset = b2.form_submit_button("Reset to defaults", use_container_width=True)

if reset:
    d = load_param_defaults()
    st.session_state["ma_periods"] = list(map(int, d["ma_periods"]))
    st.session_state["bb_period"] = int(d["bb_period"])
    st.session_state["bb_k"] = float(d["bb_k"])
elif confirm:
    st.session_state["ma_periods"] = [int(ma1), int(ma2), int(ma3), int(ma4)]
    st.session_state["bb_period"] = int(bb_period)
    st.session_state["bb_k"] = float(bb_k)

MA_PERIODS = st.session_state["ma_periods"]
BB_PERIOD = st.session_state["bb_period"]
BB_K = st.session_state["bb_k"]

start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
# Remove timezone from date column for comparison
if df["date"].dt.tz is not None:
    df["date"] = df["date"].dt.tz_localize(None)

f = df[(df["ticker"] == ticker) & (df["date"].between(start, end))].copy()

ma_series, bb_ma, bb_upper, bb_lower = compute_overlays(f, MA_PERIODS, BB_PERIOD, BB_K)


col1, col2 = st.columns([0.6, 0.4])

# ---------- 1) Price + BB and MACD ----------
with col1:
    
    st.subheader("Price with Bollinger Bands and MACD")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=("Price + Bollinger Bands", "MACD")
    )

    fig.add_trace(
        go.Candlestick(
            x=f["date"],
            open=f["open"], high=f["high"], low=f["low"], close=f["close"],
            name="OHLC",
            **candle_kwargs()  # colors
        ),
        row=1, col=1
    )

    for i, (name, series) in enumerate(ma_series):
        fig.add_trace(
            go.Scatter(
                x=f["date"], y=series, name=name, mode="lines",
                line=dict(color=ma_color(i))  # colors
            ),
            row=1, col=1
        )

    fig.add_trace(go.Scatter(x=f["date"], y=bb_ma,    name=f"BB MA {BB_PERIOD}",    mode="lines",
                             line=dict(color=color_of("BB MA"))), row=1, col=1)
    fig.add_trace(go.Scatter(x=f["date"], y=bb_upper, name=f"BB Upper ({BB_K}σ)",   mode="lines",
                             line=dict(color=color_of("BB Upper"))), row=1, col=1)
    fig.add_trace(go.Scatter(x=f["date"], y=bb_lower, name=f"BB Lower ({BB_K}σ)",   mode="lines",
                             line=dict(color=color_of("BB Lower"))), row=1, col=1)

    fig.add_trace(go.Bar(x=f["date"], y=f["MACD_HIST"], name="MACD Hist",
                         marker_color=color_of("MACD Hist")), row=2, col=1)
    fig.add_trace(go.Scatter(x=f["date"], y=f["MACD_12_26_9"], name="MACD", mode="lines",
                             line=dict(color=color_of("MACD"))), row=2, col=1)
    fig.add_trace(go.Scatter(x=f["date"], y=f["MACD_SIGNAL_9"], name="Signal", mode="lines",
                             line=dict(color=color_of("Signal"))), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=700, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2) Oscillators ----------
with col1:
    st.subheader("Oscillators")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=f["date"], y=f["RSI_14"], name="RSI 14", mode="lines",
                                 line=dict(color=color_of("RSI 14"))))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color=color_of("RSI Upper", group="hline"))
    rsi_fig.add_hline(y=30, line_dash="dash", line_color=color_of("RSI Lower", group="hline"))
    rsi_fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(rsi_fig, use_container_width=True)

    stoch_fig = go.Figure()
    stoch_fig.add_trace(go.Scatter(x=f["date"], y=f["STOCH_K_14_3"], name="%K", mode="lines",
                                   line=dict(color=color_of("%K"))))
    stoch_fig.add_trace(go.Scatter(x=f["date"], y=f["STOCH_D_3"], name="%D", mode="lines",
                                   line=dict(color=color_of("%D"))))
    stoch_fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(stoch_fig, use_container_width=True)

    cci_fig = go.Figure()
    cci_fig.add_trace(go.Scatter(x=f["date"], y=f["CCI_20"], name="CCI 20", mode="lines",
                                 line=dict(color=color_of("CCI 20"))))
    cci_fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(cci_fig, use_container_width=True)

# ---------- 3) Metrics ----------
with col2:

    kpis_tab, guide_tab = st.tabs(["Indicators","How to read"])

    with kpis_tab:
        # ---- tiny formatters ----
        def fmt_pct(x): return "NA" if x is None or pd.isna(x) else f"{x:.2%}"
        def fmt_price(x): return "NA" if x is None or pd.isna(x) else f"{x:.2f}"
        def abbr(x):
            if x is None or pd.isna(x): return "NA"
            n = float(x)
            for u in ["","K","M","B","T"]:
                if abs(n) < 1000: return f"{n:,.0f}{u}"
                n /= 1000
            return f"{n:.1f}P"
        def status_emoji(s): return {"good":"🟢","warn":"🟡","bad":"🔴","neutral":"⚪"}.get(s,"⚪")

        # working frames
        latest = f.sort_values("date").iloc[-1]
        g = df[df["ticker"] == ticker].copy()  # full history for this ticker

        # ---------- Price snapshot ----------
        st.subheader("Price snapshot")
        c1, c2, c3 = st.columns(3)
        close_val = latest["close"]
        day_chg = day_change_pct(f["close"])
        c1.metric("Close", fmt_price(close_val))
        c2.metric("Day change %", fmt_pct(day_chg), delta=fmt_pct(day_chg) if day_chg is not None else None)
        rp, rp_state = range_pos(f["close"], 20)
        c3.metric(f"Range pos 20D {status_emoji(rp_state)}", "NA" if rp is None else f"{rp*100:.0f}th pct")

        # ---------- Returns ----------
        st.subheader("Returns")
        rcols = st.columns(5)
        periods = [("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("YTD", "ytd")]
        g_sorted_close = g.sort_values("date")["close"]
        for (label, n), col in zip(periods, rcols):
            r = ytd_ret(g) if n == "ytd" else ret_n(g_sorted_close, n)
            col.metric(label, fmt_pct(r))

        # ---------- Trend ----------
        st.subheader("Trend")
        t1, t2, t3 = st.columns(3)
        dists = ma_distances(latest["close"], ma_series)  # uses ma_series from earlier overlays
        breadth, breadth_state = ma_breadth(dists)
        t1.metric(f"Above MAs {status_emoji(breadth_state)}", f"{breadth}/{len(dists)}")
        # show two anchors near 20 and 200
        for target, col in zip([20, 200], (t2, t3)):
            name, dist = min(dists, key=lambda kv: abs(int(kv[0].split()[-1]) - target))
            col.metric(f"Dist to {name}", fmt_pct(dist))
        # optional slope info
        slope, slope_state = ma_slope(g, period=50, lookback=10)
        st.caption(f"MA50 slope 10 bars {status_emoji(slope_state)}: {fmt_price(slope)}")

        # ---------- Momentum ----------
        st.subheader("Momentum")
        m1, m2, m3 = st.columns(3)
        macd = latest.get("MACD_12_26_9", np.nan)
        signal = latest.get("MACD_SIGNAL_9", np.nan)
        hist = latest.get("MACD_HIST", np.nan)
        macd_s = macd_state(macd, signal, hist)
        m1.metric(f"MACD {status_emoji(macd_s)}", f"{macd:.2f} / {signal:.2f}" if pd.notna(macd) and pd.notna(signal) else "NA")
        rsi_val = latest.get("RSI_14", np.nan)
        rsi_s = rsi_state(rsi_val)
        m2.metric(f"RSI(14) {status_emoji(rsi_s)}", f"{rsi_val:.1f}" if pd.notna(rsi_val) else "NA")
        k = latest.get("STOCH_K_14_3", np.nan); d_ = latest.get("STOCH_D_3", np.nan)
        st_s = stoch_state(k, d_)
        m3.metric(f"Stoch %K/%D {status_emoji(st_s)}", f"{k:.1f}/{d_:.1f}" if pd.notna(k) and pd.notna(d_) else "NA")

        # ---------- Volatility & bands ----------
        st.subheader("Volatility & bands")
        v1, v2, v3 = st.columns(3)
        v1.metric("Ann vol 30D | 90D",
                f"{fmt_pct(latest.get('ANNUALIZED_VOL_30D'))} | {fmt_pct(latest.get('ANNUALIZED_VOL_90D'))}")
        pct, vp_state = vol_percentile_1y(g)
        v2.metric(f"Vol pct’ile 1Y {status_emoji(vp_state)}", "NA" if pct is None else f"{pct:.0f}th")
        width, bw_state = bb_width_pct(bb_upper, bb_lower, f["close"])  # uses bb_* from earlier overlays
        v3.metric(f"BB width {status_emoji(bw_state)}", fmt_pct(width))

        # ---------- Liquidity ----------
        st.subheader("Liquidity")
        l1, l2 = st.columns(2)
        adv = adv20_dollar(g)
        l1.metric("ADV 20D ($)", abbr(adv))
        vz = vol_zscore_today(g)
        l2.metric("Volume z-score", "NA" if vz is None else f"{vz:.2f}")

        # ---------- Drawdown ----------
        st.subheader("Drawdown")
        d1, d2 = st.columns(2)
        dd, dd_state = curr_dd_1y(g)
        d1.metric(f"From 1Y high {status_emoji(dd_state)}", fmt_pct(dd))
        d2.metric("Max DD 90D", fmt_pct(latest.get("MAX_DRAWDOWN_90D")))

        # ---------- Signals ----------
        st.subheader("Signals")
        sig_cols = [c for c in f.columns if c.startswith("SIG_")]
        if sig_cols:
            latest_sig = f.iloc[-1][sig_cols]
            bull = int((latest_sig > 0).sum()); bear = int((latest_sig < 0).sum())
            net = bull - bear
            s1, s2 = st.columns(2)
            s1.metric("Bull − Bear", f"{net}  ({bull} / {bear})", delta=f"{'+' if net>=0 else ''}{net}")
            sig_df = f[sig_cols].copy()
            chg = sig_df.ne(sig_df.shift()).any(axis=1).to_numpy()
            last_change = np.where(chg)[0][-1] if chg.any() else None
            since = None if last_change is None else (len(sig_df) - 1 - last_change)
            s2.metric("Bars since last change", "NA" if since is None else str(int(since)))
        else:
            st.caption("No `SIG_*` columns found.")

    with guide_tab:
        st.markdown(
            """
### How to read these indicators
Legend: 🟢 good · 🟡 warn · 🔴 bad · ⚪ neutral

**Price snapshot**
- **Close**: last traded price.
- **Day change %**: (last / prev − 1). 🟢 > 0, 🔴 < 0.
- **Range pos 20D**: (Close − 20D min) / (20D max − 20D min). 🟢 ≥ 70th, 🔴 ≤ 30th.

**Returns**
- **1W / 1M / 3M / 6M / YTD**: simple total return. 🟢 ≥ 0%, 🔴 < 0%.

**Trend**
- **Dist to MA x**: Close / MA − 1. 🟢 > 0, 🔴 < 0.
- **Above MAs**: count of MAs with distance > 0. 🟢 ≥ 2, 🟡 1, 🔴 0.
- **MA50 slope (10 bars)**: MA50[t] − MA50[t−10]. 🟢 > 0, 🔴 < 0.

**Momentum**
- **MACD**: 🟢 if MACD > Signal and Hist > 0; else 🔴.
- **RSI(14)**: 🟢 45–70, 🟡 70–80 or 35–45, 🔴 <35 or >80.
- **Stoch %K/%D**: 🟢 if %K > %D and %K > 50.

**Volatility & bands**
- **Ann vol 30D | 90D**: realized vol (informational).
- **Vol percentile 1Y (30D)**: rank of current 30D vol over ~252 bars. 🟢 20–60th, 🟡 <10th or 60–80th, 🔴 >80th.
- **BB width %**: (Upper − Lower) / Close. 🟢 5–15%, 🟡 <3%, 🔴 >25%.

**Liquidity**
- **ADV 20D ($)**: mean 20D dollar volume (set desk threshold).
- **Volume z-score**: today vs prior 20D. 🟢 ≥ +1, 🟡 −1..+1, 🔴 ≤ −1.

**Drawdown**
- **From 1Y high**: Close / 1Y max − 1. 🟢 ≥ −5%, 🟡 −5%..−15%, 🔴 < −15%.
- **Max DD 90D**: largest peak-to-trough over 90D (informational).

**Signals**
- **Bull − Bear**: net count of `SIG_*` > 0 minus < 0. 🟢 > 0, 🟡 0, 🔴 < 0.
- **Bars since last change**: time since any signal flipped.

**Charts (left)**
- **Candlestick**: green up, red down.
- **Bollinger Bands**: mean ± k·σ. Compression often precedes expansion.
- **MACD panel**: histogram above zero supports up momentum.
- **RSI/Stoch/CCI**: use levels and crosses as context, not alone.
            """
        )
