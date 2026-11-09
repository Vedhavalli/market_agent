"""Technical indicators + a simple, transparent rule-based score.

The score is a screen, not a prediction: each rule adds or subtracts one point.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    out[(ad == 0) & au.notna()] = 100.0
    return out


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _score(r: dict) -> tuple[int, list[str]]:
    s, why = 0, []
    p = r["last"]

    def rule(ok: bool, good: str, bad: str):
        nonlocal s
        s += 1 if ok else -1
        why.append(good if ok else bad)

    if r["sma50"] is not None:
        rule(p > r["sma50"], "above 50-DMA", "below 50-DMA")
    if r["sma200"] is not None:
        rule(p > r["sma200"], "above 200-DMA", "below 200-DMA")
    if r["sma50"] is not None and r["sma200"] is not None:
        rule(r["sma50"] > r["sma200"], "50>200 DMA", "50<200 DMA")
    if r["macd_hist"] is not None:
        rule(r["macd_hist"] > 0, "MACD positive", "MACD negative")
    rs = r["rsi"]
    if rs is not None:
        if rs > 75:
            s -= 1
            why.append(f"RSI {rs:.0f} overbought")
        elif 50 <= rs <= 70:
            s += 1
            why.append(f"RSI {rs:.0f} healthy")
        elif rs < 40:
            s -= 1
            why.append(f"RSI {rs:.0f} weak")
    return s, why


def analyze(df: pd.DataFrame | None) -> dict | None:
    """Return latest price stats, indicators, levels and a score for one symbol."""
    if df is None or len(df) < 30:
        return None
    c, h, l = df["Close"], df["High"], df["Low"]
    last, prev = float(c.iloc[-1]), float(c.iloc[-2])
    r: dict = {
        "date": df.index[-1].date(),
        "last": last,
        "prev": prev,
        "chg": last - prev,
        "chg_pct": (last / prev - 1) * 100 if prev else 0.0,
    }
    for n in (20, 50, 200):
        r[f"sma{n}"] = _f(sma(c, n).iloc[-1])
    r["rsi"] = _f(rsi(c).iloc[-1])
    line, sig, hist = macd(c)
    r["macd"], r["macd_signal"], r["macd_hist"] = (_f(line.iloc[-1]), _f(sig.iloc[-1]), _f(hist.iloc[-1]))

    H, L = float(h.iloc[-1]), float(l.iloc[-1])
    p = (H + L + last) / 3
    r.update(pivot=p, r1=2 * p - L, s1=2 * p - H, r2=p + (H - L), s2=p - (H - L))

    w, m = df.iloc[-252:], df.iloc[-20:]
    r["hi52"], r["lo52"] = float(w["High"].max()), float(w["Low"].min())
    r["hi20"], r["lo20"] = float(m["High"].max()), float(m["Low"].min())
    r["dist_52h_pct"] = (last / r["hi52"] - 1) * 100

    r["score"], r["why"] = _score(r)
    sc = r["score"]
    r["trend"] = (
        "Bullish" if sc >= 3 else "Mild bullish" if sc >= 1 else "Neutral" if sc == 0
        else "Mild bearish" if sc >= -2 else "Bearish"
    )
    r["signal"] = "BUY-WATCH" if sc >= 3 else "SELL-WATCH" if sc <= -3 else "HOLD"
    return r
