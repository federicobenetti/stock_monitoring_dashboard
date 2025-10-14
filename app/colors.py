from functools import lru_cache
import yaml

CFG_PATH_DEFAULT = "config/colors.yaml"

@lru_cache(maxsize=1)
def _cfg(path: str = CFG_PATH_DEFAULT):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def color_of(name: str, group: str = "traces", path: str = CFG_PATH_DEFAULT) -> str:
    cfg = _cfg(path)
    bucket = (cfg.get(group) or {})
    if name in bucket:
        return bucket[name]
    low = {k.lower(): v for k, v in bucket.items()}
    if name.lower() in low:
        return low[name.lower()]
    d = cfg.get("defaults", {}) or {}
    if group == "traces":
        return d.get("line", "#808080")
    if group == "bars":
        return d.get("bar", "#808080")
    if group == "hline":
        return d.get("hline", "#A0A0A0")
    return d.get("line", "#808080")

def candle_kwargs(path: str = CFG_PATH_DEFAULT) -> dict:
    cfg = _cfg(path)
    d = cfg.get("defaults", {}) or {}
    c = cfg.get("candlestick", {}) or {}
    inc_line = c.get("increasing_line", d.get("candle_increasing", "#10B981"))
    inc_fill = c.get("increasing_fill", inc_line)
    dec_line = c.get("decreasing_line", d.get("candle_decreasing", "#EF4444"))
    dec_fill = c.get("decreasing_fill", dec_line)
    return dict(
        increasing_line_color=inc_line,
        increasing_fillcolor=inc_fill,
        decreasing_line_color=dec_line,
        decreasing_fillcolor=dec_fill,
    )

def ma_color(i: int, path: str = CFG_PATH_DEFAULT) -> str:
    cfg = _cfg(path)
    palette = (cfg.get("palettes", {}) or {}).get("ma") or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    return palette[i % len(palette)]
