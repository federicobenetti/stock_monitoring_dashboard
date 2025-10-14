import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path

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


@st.cache_data
def load_data():
    main = pd.read_csv("data/raw/main_wide.csv", parse_dates=["date"])
    pre = pd.read_csv("data/processed/precomputed_wide.csv", parse_dates=["date"])
    df = pd.merge(main, pre, on=["date", "ticker"], how="left")
    return df, main, pre

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

df, main, pre = load_data()
init_params()

col1, col2 = st.columns([0.6, 0.4])

# ---------- Controls ----------
st.sidebar.title("Controls")
tickers = sorted(df["ticker"].unique().tolist())
ticker = st.sidebar.selectbox("Ticker", tickers, index=0)
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
f = df[(df["ticker"] == ticker) & (df["date"].between(start, end))].copy()

ma_series, bb_ma, bb_upper, bb_lower = compute_overlays(f, MA_PERIODS, BB_PERIOD, BB_K)



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
