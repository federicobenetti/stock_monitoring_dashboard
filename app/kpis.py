import numpy as np
import pandas as pd

# --------- returns ---------
def ret_n(closes: pd.Series, n: int):
    if len(closes) <= n: return None
    c0, cN = closes.iloc[-1], closes.iloc[-1 - n]
    if pd.isna(c0) or pd.isna(cN) or cN == 0: return None
    return c0 / cN - 1

def ytd_ret(df_tkr: pd.DataFrame):
    if df_tkr.empty: return None
    d = df_tkr.sort_values("date")
    y = d.iloc[-1]["date"].year
    s = d[d["date"].dt.year == y]
    if s.empty: return None
    last, first = d.iloc[-1]["close"], s.iloc[0]["close"]
    if pd.isna(last) or pd.isna(first) or first == 0: return None
    return last / first - 1

def day_change_pct(closes: pd.Series):
    if len(closes) < 2: return None
    a, b = closes.iloc[-2], closes.iloc[-1]
    if pd.isna(a) or pd.isna(b) or a == 0: return None
    return b / a - 1

# --------- states / classifications ---------
def range_pos(series: pd.Series, window: int = 20):
    s = series.dropna()
    if len(s) < 2: return None, "neutral"
    w = s.tail(window)
    lo, hi, last = w.min(), w.max(), w.iloc[-1]
    if hi == lo: return None, "neutral"
    pos = (last - lo) / (hi - lo)
    level = "good" if pos >= 0.7 else ("bad" if pos <= 0.3 else "warn")
    return pos, level

def rsi_state(val):
    if val is None or pd.isna(val): return "neutral"
    if 45 <= val <= 70: return "good"
    if 70 < val <= 80 or 35 <= val < 45: return "warn"
    return "bad"

def stoch_state(k, d):
    if any(pd.isna([k, d])): return "neutral"
    return "good" if (k > d and k > 50) else "bad"

def macd_state(macd, signal, hist):
    if any(pd.isna([macd, signal, hist])): return "neutral"
    return "good" if (macd > signal and hist > 0) else "bad"

def vol_percentile_1y(df_tkr: pd.DataFrame):
    d = df_tkr.sort_values("date").dropna(subset=["ANNUALIZED_VOL_30D"])
    tail = d.tail(252)["ANNUALIZED_VOL_30D"]
    if tail.count() < 10: return None, "neutral"
    pct = (tail.rank(pct=True).iloc[-1]) * 100.0
    level = "good" if 20 <= pct <= 60 else ("warn" if (pct < 10 or 60 < pct <= 80) else "bad")
    return pct, level

def bb_width_pct(upper: pd.Series, lower: pd.Series, close: pd.Series):
    if any(s.empty for s in (upper, lower, close)): return None, "neutral"
    u, l, c = upper.iloc[-1], lower.iloc[-1], close.iloc[-1]
    if any(pd.isna([u, l, c])) or c == 0: return None, "neutral"
    w = (u - l) / c
    level = "good" if 0.05 <= w <= 0.15 else ("warn" if w < 0.03 else "bad")
    return w, level

def curr_dd_1y(df_tkr: pd.DataFrame):
    d = df_tkr.sort_values("date").tail(252)
    if d.empty: return None, "neutral"
    mx, cur = d["close"].max(), d.iloc[-1]["close"]
    if pd.isna(mx) or pd.isna(cur) or mx == 0: return None, "neutral"
    dd = cur / mx - 1
    level = "good" if dd >= -0.05 else ("warn" if dd >= -0.15 else "bad")
    return dd, level

# --------- liquidity ---------
def adv20_dollar(df_tkr: pd.DataFrame):
    if "volume" not in df_tkr.columns: return None
    d = df_tkr.sort_values("date").tail(20)
    if d.empty: return None
    return (d["volume"] * d["close"]).mean()

def vol_zscore_today(df_tkr: pd.DataFrame):
    if "volume" not in df_tkr.columns: return None
    d = df_tkr.sort_values("date").tail(21)
    if len(d) < 5: return None
    last = d.iloc[-1]["volume"]
    base = d.iloc[:-1]["volume"]
    mu, sd = base.mean(), base.std(ddof=1)
    if sd == 0 or pd.isna(sd): return None
    return (last - mu) / sd

# --------- moving averages helpers ---------
def ma_distances(latest_close, ma_series_list):
    out = []
    for name, s in ma_series_list:
        v = None if s.empty or pd.isna(s.iloc[-1]) else s.iloc[-1]
        out.append((name, None if v in (None, 0) or pd.isna(latest_close) else latest_close / v - 1))
    return out

def ma_breadth(dists):
    vals = [v for _, v in dists if v is not None]
    if not vals: return 0, "neutral"
    cnt = sum(v > 0 for v in vals)
    level = "good" if cnt >= 2 else ("warn" if cnt == 1 else "bad")
    return cnt, level

def ma_slope(df_tkr: pd.DataFrame, period: int = 50, lookback: int = 10):
    d = df_tkr.sort_values("date")
    ma = d["close"].rolling(int(period)).mean()
    m = ma.dropna()
    if len(m) <= lookback: return None, "neutral"
    slope = m.iloc[-1] - m.iloc[-1 - lookback]
    return slope, ("good" if slope > 0 else "bad")
