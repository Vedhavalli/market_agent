"""Swing-trading engine: BUY / SELL signals with a full trade plan.

Every run recomputes the complete technical picture from fresh daily data (daily + weekly timeframe), then
applies transparent rules:

  * 9 trend / momentum / volume CONDITIONS, each Bullish, Bearish or Neutral -> score out of 9
  * a fresh TRIGGER in the last 3 sessions (MACD cross, Supertrend flip, 20-day breakout, 20-EMA reclaim)
  * filters: RSI not stretched, price not extended more than N x ATR from the 20-EMA

  BUY  = bullish score >= min_score  + bullish trigger + filters
  SELL = bearish score >= min_score  + bearish trigger + filters

Each signal gets a trade plan (entry, ATR / swing-low based stop-loss, two targets at fixed reward:risk,
profit %, estimated holding time), and the *same* rules are backtested on the stock's own history so the
historical win rate, average return and holding days are visible before acting.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import advice
import data
import fundamentals
import horizons
import macro
import report as rpt
import strategy
from indicators import ema, macd, rsi, sma
from util import DASH, num, pct

DEFAULTS = {
    "min_score": 7,            # of 9 conditions
    "atr_stop_min": 1.0,       # stop-loss at least this many ATRs from entry
    "atr_stop_max": 2.5,       # ...and at most this many
    "t1_rr": 1.5,              # target 1 = entry + 1.5 x risk
    "t2_rr": 3.0,              # target 2 = entry + 3 x risk
    "max_hold_days": 20,       # time stop used by the backtest
    "max_extension_atr": 2.5,  # skip if price is already this many ATRs beyond the 20-EMA (don't chase)
    "history": "10y",          # data period downloaded for analysis + backtest (long-term view needs years)
}
MIN_BARS = 260

SELL_NOTE = (
    "SELL = exit / avoid longs. In India, overnight short selling of shares in the cash segment is not "
    "allowed; a short swing trade needs F&O (stock futures / puts). Profit % shown is for a short position."
)
PLAN_NOTE = (
    "Entry = latest close. Stop-loss = below the 10-day swing low (above the swing high for SELL), kept within "
    "{lo:g}-{hi:g} x ATR. T1 = {t1:g}R, T2 = {t2:g}R. Timeline = trading days, estimated from the stock's ATR "
    "(fastest-slowest) and its own past signals. Suggested exit: book ~50% at T1, move stop to entry, trail "
    "the rest with Supertrend / 20-EMA; exit if not at T1 within {hold} sessions. Profit % is before brokerage, "
    "STT and tax."
)


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **cfg.get("swing", {})}


# ---------------------------------------------------------------- indicators
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14):
    up, dn = df["High"].diff(), -df["Low"].diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(df, n).replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), pdi, mdi


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    a = atr(df, n).to_numpy()
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("High", "Low", "Close"))
    mid = (h + l) / 2
    ub, lb = mid + mult * a, mid - mult * a
    fu, fl = ub.copy(), lb.copy()
    d = np.ones(len(c))
    line = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        if np.isnan(a[i]):
            continue
        fu[i] = ub[i] if np.isnan(fu[i - 1]) or ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1] else fu[i - 1]
        fl[i] = lb[i] if np.isnan(fl[i - 1]) or lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1] else fl[i - 1]
        d[i] = (-1 if c[i] < fl[i] else 1) if d[i - 1] == 1 else (1 if c[i] > fu[i] else -1)
        line[i] = fl[i] if d[i] == 1 else fu[i]
    return pd.Series(d, df.index), pd.Series(line, df.index)


def stochastic(df: pd.DataFrame, k: int = 14, smooth: int = 3):
    lo, hi = df["Low"].rolling(k).min(), df["High"].rolling(k).max()
    kk = (100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)).rolling(smooth).mean()
    return kk, kk.rolling(smooth).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def weekly_trend(close: pd.Series, n: int = 20) -> pd.Series:
    """1.0 if the last *completed* week closed above its 20-week EMA, 0.0 if below (no look-ahead)."""
    wk = close.resample("W-FRI").last().dropna()
    e = ema(wk, n).where(np.arange(len(wk)) >= n - 1)
    up = (wk > e).astype(float).where(e.notna())
    up.index = up.index + pd.Timedelta(days=1)  # a week's value becomes usable the day after it closes
    return up.reindex(close.index, method="ffill")


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Volume"] = df["Volume"].fillna(0)
    df["Open"] = df["Open"].fillna(df["Close"])
    c = df["Close"]
    x = pd.DataFrame({"open": df["Open"], "high": df["High"], "low": df["Low"], "close": c,
                      "volume": df["Volume"]}, index=df.index)
    bar = np.arange(len(df))
    for n in (20, 50, 200):
        x[f"ema{n}"] = ema(c, n).where(bar >= n - 1)
    x["rsi"] = rsi(c)
    x["macd"], x["macd_sig"], x["macd_hist"] = macd(c)
    x["atr"] = atr(df)
    x["adx"], x["pdi"], x["mdi"] = adx(df)
    x["st_dir"], x["st_line"] = supertrend(df)
    x["stoch_k"], x["stoch_d"] = stochastic(df)
    mid, sd = sma(c, 20), c.rolling(20).std()
    x["bb_mid"], x["bb_up"], x["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    x["obv"] = obv(df)
    x["obv_ema"] = ema(x["obv"], 20)
    x["vol20"] = df["Volume"].rolling(20).mean()
    x["hi20"], x["lo20"] = df["High"].rolling(20).max(), df["Low"].rolling(20).min()
    x["hi10"], x["lo10"] = df["High"].rolling(10).max(), df["Low"].rolling(10).min()
    x["wk_up"] = weekly_trend(c)
    return x


# ---------------------------------------------------------------- rules
C_TREND, C_ALIGN, C_LONG = "Trend: price vs 50-EMA", "EMA alignment: 20 vs 50", "Long-term: price vs 200-EMA"
C_MACD, C_RSI, C_ADX = "MACD histogram", "RSI(14) momentum zone", "ADX(14) strength + direction"
C_ST, C_OBV, C_WK = "Supertrend(10,3)", "OBV volume flow", "Weekly trend: 20-week EMA"


def conditions(x: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """label -> (bullish, bearish) boolean series. NaN compares False, so missing data is Neutral."""
    c, vol_ok = x["close"], x["vol20"] > 0
    return {
        C_TREND: (c > x["ema50"], c < x["ema50"]),
        C_ALIGN: (x["ema20"] > x["ema50"], x["ema20"] < x["ema50"]),
        C_LONG: (c > x["ema200"], c < x["ema200"]),
        C_MACD: (x["macd_hist"] > 0, x["macd_hist"] < 0),
        C_RSI: (x["rsi"].between(50, 70), x["rsi"].between(30, 50, inclusive="left")),
        C_ADX: ((x["adx"] > 20) & (x["pdi"] > x["mdi"]), (x["adx"] > 20) & (x["mdi"] > x["pdi"])),
        C_ST: (x["st_dir"] == 1, x["st_dir"] == -1),
        C_OBV: ((x["obv"] > x["obv_ema"]) & vol_ok, (x["obv"] < x["obv_ema"]) & vol_ok),
        C_WK: (x["wk_up"] == 1, x["wk_up"] == 0),
    }


def triggers(x: pd.DataFrame):
    """Fresh events within the last 3 sessions."""
    def recent(s: pd.Series) -> pd.Series:
        return s.fillna(False).astype(int).rolling(3, min_periods=1).max().astype(bool)

    c, h, st, e20 = x["close"], x["macd_hist"], x["st_dir"], x["ema20"]
    buy = {
        "MACD bullish crossover": recent((h > 0) & (h.shift() <= 0)),
        "Supertrend flipped up": recent((st == 1) & (st.shift() == -1)),
        "20-day high breakout": recent(c > x["hi20"].shift()),
        "Reclaimed 20-EMA": recent((c > e20) & (c.shift() <= e20.shift())),
    }
    sell = {
        "MACD bearish crossover": recent((h < 0) & (h.shift() >= 0)),
        "Supertrend flipped down": recent((st == -1) & (st.shift() == 1)),
        "20-day low breakdown": recent(c < x["lo20"].shift()),
        "Lost 20-EMA": recent((c < e20) & (c.shift() >= e20.shift())),
    }
    return buy, sell


def signals(x: pd.DataFrame, s: dict):
    cond = conditions(x)
    bull = sum(b.astype(int) for b, _ in cond.values())
    bear = sum(r.astype(int) for _, r in cond.values())
    tb, ts = triggers(x)
    any_b = pd.concat(tb.values(), axis=1).any(axis=1)
    any_s = pd.concat(ts.values(), axis=1).any(axis=1)
    ext = s["max_extension_atr"] * x["atr"]
    ok = x["atr"] > 0
    buy = ok & (bull >= s["min_score"]) & any_b & (x["rsi"] < 72) & (x["close"] - x["ema20"] <= ext)
    sell = ok & (bear >= s["min_score"]) & any_s & (x["rsi"] > 28) & (x["ema20"] - x["close"] <= ext)
    return bull, bear, buy, sell, cond, tb, ts


def no_trade_reason(r: dict, s: dict) -> str:
    """Plain-English reason why analyze() produced no BUY/SELL."""
    L = r["L"]
    bull = r["bull"] >= r["bear"]
    score, trig = (r["bull"], r["trig_buy"]) if bull else (r["bear"], r["trig_sell"])
    word = "bullish" if bull else "bearish"
    if score < s["min_score"]:
        return f"Only {score}/9 {word} conditions met (need {s['min_score']}): trend is not clear enough."
    if not trig:
        return f"Setup is {word} ({score}/9) but no fresh entry trigger in the last 3 sessions - wait for one."
    if bull and L["rsi"] >= 72:
        return f"Bullish ({score}/9) but RSI {L['rsi']:.0f} is overbought - wait for a dip."
    if not bull and L["rsi"] <= 28:
        return f"Bearish ({score}/9) but RSI {L['rsi']:.0f} is oversold - wait for a bounce."
    return (f"{word.capitalize()} ({score}/9) but price is more than {s['max_extension_atr']:g} ATR away from "
            f"the 20-EMA - too extended to chase; wait for a pullback toward {num(L['ema20'])}.")


def plan(side: str, entry: float, a: float, swing: float, s: dict) -> dict:
    """Stop beyond the 10-day swing extreme (+0.25 ATR buffer), clamped to [atr_stop_min, atr_stop_max] ATR."""
    lo, hi = s["atr_stop_min"] * a, s["atr_stop_max"] * a
    if side == "BUY":
        stop = min(max(swing - 0.25 * a, entry - hi), entry - lo)
        risk = entry - stop
        t1, t2 = entry + s["t1_rr"] * risk, entry + s["t2_rr"] * risk
    else:
        stop = max(min(swing + 0.25 * a, entry + hi), entry + lo)
        risk = stop - entry
        t1, t2 = entry - s["t1_rr"] * risk, entry - s["t2_rr"] * risk
    return {"side": side, "entry": entry, "stop": stop, "t1": t1, "t2": t2, "risk": risk,
            "risk_pct": risk / entry * 100, "t1_pct": abs(t1 - entry) / entry * 100,
            "t2_pct": abs(t2 - entry) / entry * 100}


def backtest(x: pd.DataFrame, buy: pd.Series, sell: pd.Series, s: dict) -> dict:
    """Replay the signals: enter next open, exit at stop / T1 / time stop. Stop is checked first (conservative)."""
    o, h, l, c, a = (x[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "atr"))
    lo10, hi10 = x["lo10"].to_numpy(dtype=float), x["hi10"].to_numpy(dtype=float)
    b, sl = buy.to_numpy(), sell.to_numpy()
    n, hold = len(x), int(s["max_hold_days"])
    trades, i = [], 0
    while i < n - 1:
        side = "BUY" if b[i] else "SELL" if sl[i] else None
        if side is None or not a[i] > 0:
            i += 1
            continue
        entry = o[i + 1]
        p = plan(side, entry, a[i], lo10[i] if side == "BUY" else hi10[i], s)
        end, exit_px, hit = min(i + 1 + hold, n), None, "time"
        for j in range(i + 1, end):
            if side == "BUY":
                if l[j] <= p["stop"]:
                    exit_px, hit = min(o[j], p["stop"]), "stop"
                    break
                if h[j] >= p["t1"]:
                    exit_px, hit = p["t1"], "target"
                    break
            else:
                if h[j] >= p["stop"]:
                    exit_px, hit = max(o[j], p["stop"]), "stop"
                    break
                if l[j] <= p["t1"]:
                    exit_px, hit = p["t1"], "target"
                    break
        else:
            j = end - 1
            if end == n and end - i - 1 < hold:  # trade still open at the end of the data
                break
            exit_px = c[j]
        ret = (exit_px / entry - 1) * 100 * (1 if side == "BUY" else -1)
        trades.append((side, ret, j - i, hit))
        i = j + 1

    out = {}
    for side in ("BUY", "SELL"):
        t = [tr for tr in trades if tr[0] == side]
        if not t:
            continue
        rets = [tr[1] for tr in t]
        tgt_days = [tr[2] for tr in t if tr[3] == "target"]
        out[side] = {
            "n": len(t), "win_rate": 100 * sum(r > 0 for r in rets) / len(t),
            "target_rate": 100 * len(tgt_days) / len(t), "avg_ret": float(np.mean(rets)),
            "best": max(rets), "worst": min(rets), "avg_days": float(np.mean([tr[2] for tr in t])),
            "days_to_target": float(np.mean(tgt_days)) if tgt_days else None,
        }
    return out


def eta(dist: float, a: float, start, hist_days: float | None = None) -> dict | None:
    """Trading-day range to cover `dist`: fastest ~0.8 ATR/day of net progress, slowest ~0.3 ATR/day."""
    if not a or a <= 0 or math.isnan(a):
        return None
    fast = max(1, math.ceil(dist / (0.8 * a)))
    slow = max(fast + 1, math.ceil(dist / (0.3 * a)))
    by = pd.Timestamp(np.busday_offset(np.datetime64(start), slow, roll="forward")).date()
    return {"fast": fast, "slow": slow, "by": by, "hist": hist_days}


def eta_str(e: dict | None) -> str:
    if not e:
        return DASH
    out = f"{e['fast']}-{e['slow']} days (by {e['by']:%d %b})"
    return out + (f"; past avg {e['hist']:.0f}d" if e.get("hist") else "")


def sr_levels(df: pd.DataFrame, price: float, lookback: int = 120, k: int = 3):
    """Nearest two fractal swing highs above and swing lows below the price."""
    d, win = df.iloc[-lookback:], 2 * k + 1
    hi = d["High"][d["High"] == d["High"].rolling(win, center=True).max()]
    lo = d["Low"][d["Low"] == d["Low"].rolling(win, center=True).min()]
    res = sorted({round(float(v), 2) for v in hi if v > price})[:2]
    sup = sorted({round(float(v), 2) for v in lo if v < price}, reverse=True)[:2]
    return res, sup


def candle(x: pd.DataFrame) -> str:
    o, h, l, c = (float(x[k].iloc[-1]) for k in ("open", "high", "low", "close"))
    po, pc = float(x["open"].iloc[-2]), float(x["close"].iloc[-2])
    rng, body = h - l, abs(c - o)
    if rng <= 0:
        return "-"
    if pc < po and c > o and c >= po and o <= pc:
        return "Bullish engulfing"
    if pc > po and c < o and o >= pc and c <= po:
        return "Bearish engulfing"
    if body <= 0.1 * rng:
        return "Doji (indecision)"
    upper, lower = h - max(o, c), min(o, c) - l
    if body <= 0.35 * rng and lower >= 2 * body and upper <= 0.25 * rng:
        return "Hammer (bullish reversal)"
    if body <= 0.35 * rng and upper >= 2 * body and lower <= 0.25 * rng:
        return "Shooting star (bearish reversal)"
    return "Bullish candle" if c > o else "Bearish candle"


def market_regime(df: pd.DataFrame | None) -> str:
    if df is None or len(df) < 60:
        return "Unknown"
    L = compute(df).iloc[-1]
    up = [L["close"] > L["ema50"], L["ema20"] > L["ema50"], L["st_dir"] == 1]
    return "Bullish" if all(up) else "Bearish" if not any(up) else "Sideways"


def confidence(score: int, side: str, regime: str, bt: dict | None) -> str:
    aligned = regime == ("Bullish" if side == "BUY" else "Bearish")
    opposed = regime == ("Bearish" if side == "BUY" else "Bullish")
    tested = bt is not None and bt["n"] >= 5
    if opposed or (tested and bt["win_rate"] < 40):
        return "Low"
    if score >= 8 and aligned and (not tested or bt["win_rate"] >= 50):
        return "High"
    return "Medium"


# ---------------------------------------------------------------- trend + outlook (beyond today's bar)
def trend_view(x: pd.DataFrame) -> dict:
    """Trend on four horizons (short / medium / long / weekly) and an overall label."""
    L, c = x.iloc[-1], x["close"]

    def ret(n: int) -> float:
        return (float(c.iloc[-1]) / float(c.iloc[-1 - n]) - 1) * 100 if len(c) > n else float("nan")

    def state(col: str, n: int) -> str:
        e = x[col]
        if pd.isna(L[col]) or pd.isna(e.iloc[-1 - n]):
            return "n/a"
        above, rising = L["close"] > L[col], e.iloc[-1] > e.iloc[-1 - n]
        return "Up" if above and rising else "Down" if not above and not rising else "Sideways"

    def where(col: str) -> str:
        return f"price {'above' if L['close'] > L[col] else 'below'} {col[3:]}-EMA {num(L[col])}"

    wk = {1.0: "Up", 0.0: "Down"}.get(L["wk_up"], "n/a")
    rows = [
        ["Short term (2-4 weeks)", state("ema20", 5), ret(21), "1 month", where("ema20")],
        ["Medium term (1-3 months)", state("ema50", 10), ret(63), "3 months", where("ema50")],
        ["Long term (6-12 months)", state("ema200", 20), ret(252), "1 year", where("ema200")],
        ["Weekly chart", wk, ret(5), "1 week", "last weekly close vs 20-week EMA"],
    ]
    net = sum(r[1] == "Up" for r in rows) - sum(r[1] == "Down" for r in rows)
    overall = ("Strong uptrend" if net >= 4 else "Uptrend" if net >= 2 else "Strong downtrend" if net <= -4
               else "Downtrend" if net <= -2 else "Sideways / mixed")
    ad = float(L["adx"])
    strength = (f"strong trend (ADX {ad:.0f})" if ad >= 25 else f"trend forming (ADX {ad:.0f})" if ad >= 20
                else f"weak / range-bound (ADX {ad:.0f})")
    return {"rows": rows, "overall": overall, "strength": strength}


def signal_history(x: pd.DataFrame, buy: pd.Series, sell: pd.Series) -> dict:
    """Last signal of each side and how often this stock produces one (a new setup = no signal in the prior 10 days)."""
    c, years = x["close"].to_numpy(dtype=float), max(len(x) / 252, 0.5)
    out = {}
    for side, sig in (("BUY", buy), ("SELL", sell)):
        sig = sig.fillna(False).astype(bool)
        prior = sig.shift(1, fill_value=False).astype(int).rolling(10, min_periods=1).max().astype(bool)
        idx = np.flatnonzero((sig & ~prior).to_numpy())
        if not len(idx):
            continue
        i = int(np.flatnonzero(sig.to_numpy())[-1])  # most recent signal day
        out[side] = {"n": len(idx), "per_year": len(idx) / years, "date": x.index[i].date(), "price": c[i],
                     "ago": len(x) - 1 - i, "since_pct": (c[-1] / c[i] - 1) * 100}
    return out


def _cooloff(rsi_s: pd.Series, overbought: bool) -> int | None:
    """Median sessions this stock's RSI took to leave an extreme (>=72 -> <65, or <=28 -> >35)."""
    v = rsi_s.to_numpy(dtype=float)
    days, i = [], 0
    while i < len(v):
        if (v[i] >= 72) if overbought else (v[i] <= 28):
            j = i
            while j < len(v) and not ((v[j] < 65) if overbought else (v[j] > 35)):
                j += 1
            if j < len(v):
                days.append(j - i)
            i = j + 1
        else:
            i += 1
    return int(np.median(days)) if len(days) >= 3 else None


def outlook(x: pd.DataFrame, r: dict, s: dict) -> dict:
    """When no signal fires today: what has to happen, the price levels, and a rough time window."""
    L, c, d, a = r["L"], r["last"], r["date"], float(r["L"]["atr"])
    e20, e50, e200, hi20, lo20 = (float(L[k]) for k in ("ema20", "ema50", "ema200", "hi20", "lo20"))
    st = dict(r["conditions"])

    def pc(level: float) -> str:
        return f"{(level / c - 1) * 100:+.1f}%"

    def win(dist: float) -> str:
        e = eta(abs(dist), a, d)
        return f"~{e['fast']}-{e['slow']} trading sessions (by ~{e['by']:%d %b})" if e else "unknown"

    def buy_needs() -> tuple[list[str], float | None]:
        lv = []
        if st[C_TREND] != "Bullish":
            lv.append((e50, f"a close above the 50-EMA Rs {num(e50)} ({pc(e50)})"))
        if st[C_ST] != "Bullish" and L["st_line"] > c:
            lv.append((float(L["st_line"]), f"a close above the Supertrend line Rs {num(L['st_line'])} ({pc(L['st_line'])})"))
        if st[C_LONG] != "Bullish" and e200 > 0:
            lv.append((e200, f"a close above the 200-EMA Rs {num(e200)} ({pc(e200)})"))
        other = {C_ALIGN: f"the 20-EMA (Rs {num(e20)}) to cross above the 50-EMA (Rs {num(e50)})",
                 C_MACD: "MACD to turn positive", C_RSI: f"RSI to move into 50-70 (now {L['rsi']:.0f})",
                 C_ADX: "ADX above 20 with buyers in control", C_OBV: "buying volume (OBV above its average)",
                 C_WK: "a weekly close above the 20-week EMA"}
        items = [t for _, t in lv] + [t for k, t in other.items() if st[k] != "Bullish"]
        key = max((lvl for lvl, _ in lv if lvl > c), default=None)
        return items, key

    bull_bias = r["bull"] >= r["bear"]
    steps, when = [], ""
    if r["side"]:
        pass
    elif bull_bias and r["bull"] >= s["min_score"] and not r["trig_buy"]:
        steps = [f"Breakout entry: a daily close above the 20-day high Rs {num(hi20)} ({pc(hi20)}).",
                 f"Pullback entry: a dip toward the 20-EMA Rs {num(e20)} ({pc(e20)}) and a close back above it."]
        when = f"The trend is up, so a trigger is likely soon. Breakout level is {win(hi20 - c)} away at a normal pace."
    elif bull_bias and r["bull"] >= s["min_score"] and L["rsi"] >= 72:
        steps = [f"RSI {L['rsi']:.0f} is overbought: buying now risks a pullback.",
                 f"Better entry: a dip toward the 20-EMA Rs {num(e20)} ({pc(e20)})."]
        n = _cooloff(x["rsi"], True)
        when = (f"In this stock RSI has usually cooled below 65 within ~{n} sessions." if n else
                f"A dip to the 20-EMA typically takes {win(c - e20)}.")
    elif bull_bias and r["bull"] >= s["min_score"]:
        lvl = e20 + s["max_extension_atr"] * a
        steps = [f"Price is {(c - e20) / a:.1f} ATR above the 20-EMA (limit {s['max_extension_atr']:g}): too far to chase.",
                 f"It becomes buyable on a close at or below Rs {num(lvl)} ({pc(lvl)}), "
                 "or after a few sideways days while the rising 20-EMA catches up."]
        when = f"A pullback of that size usually takes {win(c - lvl)}."
    else:
        items, key = buy_needs()
        need = max(s["min_score"] - r["bull"], 0)
        if not bull_bias and r["bear"] >= s["min_score"]:
            steps.append(f"Not a buy: the trend is down ({r['bear']}/9 bearish). Avoid fresh buying.")
            if not r["trig_sell"]:
                steps.append(f"If you hold it: a close below the 20-day low Rs {num(lo20)} ({pc(lo20)}) confirms an exit/SELL signal.")
            elif L["rsi"] <= 28:
                n = _cooloff(x["rsi"], False)
                steps.append(f"RSI {L['rsi']:.0f} is oversold: a short bounce is likely"
                             + (f" (usually within ~{n} sessions here)" if n else "") + "; holders can use it to exit.")
            else:
                steps.append(f"Price has fallen far below the 20-EMA Rs {num(e20)}: a bounce toward it may come first.")
        if need:
            steps.append(f"To turn into a BUY it needs {need} more bullish condition(s) from: " + "; ".join(items) + ".")
        else:
            steps.append("Enough conditions are bullish; it now needs a fresh trigger.")
        if key:
            when = (f"Key level Rs {num(key)} is {pc(key)} away: at a steady pace that is {win(key - c)} at the "
                    "earliest, and an entry trigger is still needed after that.")
        else:
            when = ("Price is already above its key averages; momentum indicators usually catch up in 1-3 weeks "
                    "if the price holds. Re-check daily.")
    return {"steps": steps, "when": when}


# ---------------------------------------------------------------- per-stock analysis
def analyze(df: pd.DataFrame | None, s: dict, regime: str = "Unknown") -> dict | None:
    if df is None or len(df) < MIN_BARS:
        return None
    x = compute(df)
    bull, bear, buy, sell, cond, tb, ts = signals(x, s)
    L = x.iloc[-1]
    r = {
        "date": x.index[-1].date(), "last": float(L["close"]),
        "chg_pct": (float(L["close"]) / float(x["close"].iloc[-2]) - 1) * 100,
        "bull": int(bull.iloc[-1]), "bear": int(bear.iloc[-1]), "L": L, "regime": regime,
        "macd_hist_prev": float(x["macd_hist"].iloc[-2]),
        "side": "BUY" if buy.iloc[-1] else "SELL" if sell.iloc[-1] else None,
        "conditions": [(k, "Bullish" if b.iloc[-1] else "Bearish" if br.iloc[-1] else "Neutral")
                       for k, (b, br) in cond.items()],
        "trig_buy": [k for k, v in tb.items() if v.iloc[-1]],
        "trig_sell": [k for k, v in ts.items() if v.iloc[-1]],
        "bt": backtest(x, buy, sell, s), "candle": candle(x), "notes": [],
    }
    r["res"], r["sup"] = sr_levels(df, r["last"])
    side = r["side"]
    if side:
        p = plan(side, r["last"], float(L["atr"]), float(L["lo10"] if side == "BUY" else L["hi10"]), s)
        bt = r["bt"].get(side)
        p["t1_eta"] = eta(p["t1_pct"] / 100 * p["entry"], float(L["atr"]), r["date"], bt and bt["days_to_target"])
        p["t2_eta"] = eta(p["t2_pct"] / 100 * p["entry"], float(L["atr"]), r["date"])
        r["plan"], r["score"] = p, r["bull"] if side == "BUY" else r["bear"]
        r["confidence"] = confidence(r["score"], side, regime, bt)
        if side == "BUY" and r["res"] and r["res"][0] < p["t1"]:
            r["notes"].append(f"resistance {num(r['res'][0])} before T1")
        if side == "SELL" and r["sup"] and r["sup"][0] > p["t1"]:
            r["notes"].append(f"support {num(r['sup'][0])} before T1")
        if regime == ("Bearish" if side == "BUY" else "Bullish"):
            r["notes"].append("against market trend")
    else:
        r["why_none"] = no_trade_reason(r, s)
    r["trend"] = trend_view(x)
    r["history"] = signal_history(x, buy, sell)
    r["outlook"] = outlook(x, r, s)
    return r


def _bt_str(bt: dict | None) -> str:
    if not bt:
        return "no past signals"
    return f"{bt['win_rate']:.0f}% win of {bt['n']}, avg {bt['avg_ret']:+.1f}%"


def swing_verdict(r: dict) -> str:
    if r["side"]:
        return "BUY" if r["side"] == "BUY" else "SELL / AVOID"
    return "WAIT" if r["bull"] >= r["bear"] else "AVOID"


def holding_rows(r: dict) -> list[list[str]]:
    """One row per holding style: Style, Hold for, Verdict, Buy at, Sell targets, Stop-loss, Past record."""
    side, p = r["side"], r.get("plan")
    sg = "+" if side == "BUY" else "-"
    rows = [["Swing", "days to weeks", swing_verdict(r), num(p["entry"]) if p else DASH,
             f"{num(p['t1'])} ({sg}{p['t1_pct']:.1f}%) / {num(p['t2'])} ({sg}{p['t2_pct']:.1f}%)" if p else DASH,
             f"{num(p['stop'])} ({p['risk_pct']:.1f}% risk)" if p else DASH,
             "backtest " + _bt_str(r["bt"].get(side or "BUY"))]]
    for h in r.get("horizons", {}).values():
        hp, s1 = h["plan"], h["stats"][0]
        buy = (num(hp["entry"]) if h["verdict"] == "BUY" else f"on dip to {num(hp['entry'])}") if hp else DASH
        rec = (f"up {s1['win']:.0f}% of the time {h['names'][0]} later (median {pct(s1['median'], 1)})"
               if s1 else "not enough history")
        rows.append([h["label"], h["period"], h["verdict"], buy,
                     f"{num(hp['t1'])} ({pct(hp['t1_pct'], 1)}) / {num(hp['t2'])} ({pct(hp['t2_pct'], 1)})" if hp else DASH,
                     f"{num(hp['stop'])} (weekly close, {hp['risk_pct']:.1f}% risk)" if hp else DASH, rec])
    return rows


def horizon_sections(h: dict) -> list[rpt.Section]:
    rows = [["Why", h["why"]]] + [[f"Step {i + 1}", t] for i, t in enumerate(h["steps"])]
    if h["when"]:
        rows.append(["When (estimate)", h["when"]])
    p = h["plan"]
    if p:
        rows += [
            ["Buy at" if h["verdict"] == "BUY" else "Buy at (dip zone)", f"Rs {num(p['entry'])}"],
            [f"Target 1 (~{h['names'][0]}, by {p['by1']:%b %Y})", f"Rs {num(p['t1'])} ({pct(p['t1_pct'], 1)})"],
            [f"Target 2 (~{h['names'][1]}, by {p['by2']:%b %Y})", f"Rs {num(p['t2'])} ({pct(p['t2_pct'], 1)})"],
            ["Stop-loss (weekly close below)", f"Rs {num(p['stop'])} (-{p['risk_pct']:.1f}%)"],
            ["Targets based on", "this stock's past setups" if p["source"] == "history" else "reward:risk rule"],
            ["Confidence", h["confidence"]], ["Review", h["review"]],
        ]
    rows += [["Past record", h["record"]], ["If you hold it", h["holder"]]]
    ep = h["episode"]
    if ep:
        rows.append(["This setup on this stock", ("active since " if ep["active"] else "last seen ")
                     + f"{ep['start']:%d %b %Y} at Rs {num(ep['start_price'])} ({pct(ep['since_pct'], 1)} since); "
                     f"~{ep['per_year']:.1f} separate setups a year"])
    return [
        rpt.Section(f"{h['label']} ({h['period']}): {h['verdict']}", ["Item", "Detail"], rows, align="ll",
                    note=horizons.NOTE),
        rpt.Section(f"{h['label']} checklist ({h['bull']}/{h['total']} bullish)", ["Condition", "Reading", "Status"],
                    [[k, rd, s] for k, s, rd in h["conditions"]], align="lll"),
    ]


def history_str(side: str, h: dict) -> str:
    return (f"{h['date']:%d %b %Y} at Rs {num(h['price'])} ({h['ago']} sessions ago, price since {pct(h['since_pct'])}); "
            f"this stock gives ~{h['per_year']:.1f} separate {side} setups a year")


def holistic(cfg: dict, symbol: str, r: dict, idx_df: pd.DataFrame | None):
    """Fundamentals, macro and the portfolio-strategy final call for every holding style."""
    sym = symbol.upper().removesuffix(".NS")
    fund = fundamentals.analyze(data.financials(symbol), r["last"])
    prof = fund or data.profile(symbol)
    sector, industry = prof.get("sector") or "", prof.get("industry") or ""
    mac = macro.fetch(idx_df, sector, industry)
    p = r.get("plan") if r["side"] == "BUY" else None
    specs = [{"style": "swing", "label": "Swing", "period": "days to weeks", "technical": swing_verdict(r),
              "risk_pct": p and p["risk_pct"], "confidence": r.get("confidence"), "entry": p and p["entry"]}]
    for k, h in r["horizons"].items():
        hp = h["plan"]
        specs.append({"style": k, "label": h["label"], "period": h["period"], "technical": h["verdict"],
                      "risk_pct": hp and hp["risk_pct"], "confidence": h.get("confidence"),
                      "entry": hp["entry"] if hp else r["last"]})
    return fund, mac, strategy.build(specs, fund, mac, sym, strategy.context(cfg, sym), sector)


def holistic_sections(sym: str, fund: dict | None, mac: dict | None, stg: dict) -> tuple[list, list]:
    """(summary sections shown near the top, detail sections)."""
    rows = [[d["label"] + f" ({d['period']})", d["technical"], d.get("investment_call", d["final"]),
             d["final"] + (" (starter)" if d["starter"] else ""), f"{d['size_pct']:g}%" if d["size_pct"] else DASH,
             f"{d['parts']} parts" if d["parts"] > 1 else "one go" if d["size_pct"] else DASH,
             " ".join(d["why"])] for d in stg["styles"].values()]
    head = [rpt.Section(
        "Final verdict: value + quality + timing", ["Style", "Chart timing", "Investor view", "Final call",
                                                    "Size (% of capital)", "Buy in", "Why"], rows, align="lllllll",
        note=(fund["summary"] if fund else "Fundamentals unavailable.") + " " + (mac["summary"] if mac else
              "Macro data unavailable.") + " Size = share of your total trading / investing capital.")]
    det = []
    if fund:
        g = f"grade {fund['grade']} ({fund['score']}/100)" if fund["grade"] else "not graded (limited data)"
        assumptions = fund["value_assumptions"]
        model_note = (f"Sector profile: {assumptions['sector_group']}; fair P/E {num(fund['fair_pe'], 1)}x. "
                      + (f"DCF assumes {assumptions['DCF WACC']:.0%} discount rate and "
                         f"{assumptions['terminal growth']:.0f}% terminal growth. "
                         if assumptions["DCF WACC"] else "DCF is not applied to lenders. ")
                      + "Intrinsic values are approximate estimates, not guaranteed targets.")
        det.append(rpt.Section(f"Business & fundamentals: {g}, valuation {fund['valuation'].lower()}",
                               ["Area", "Check", "Reading", "Status"], fund["checks"], align="llll",
                               note="From Yahoo Finance annual / quarterly statements (consolidated where available). "
                                                "Management quality, competition and news are not assessed. " + model_note))
        det.append(rpt.Section("Key numbers", ["Metric", "Value"], fund["metrics"], align="lr"))
        if fund["history"]:
            det.append(rpt.Section("Financial history (Rs crore)", ["Year", "Sales", "Net profit", "Net margin",
                                                                  "EPS (Rs)", "ROE"], fund["history"], align="lrrrrr"))
    else:
        det.append(rpt.Section("Business & fundamentals: not available", ["Item", "Detail"],
                               [["Data", "Yahoo did not return financial statements for " + sym]], align="ll"))
    if mac:
        mrows = [list(x) for x in mac["rows"]] + [["For this stock", t, "", ""] for t in mac["impacts"]]
        det.append(rpt.Section(f"Macro: {mac['regime']} market, {mac['tilt'].lower()} for this stock",
                               ["Factor", "Reading", "Signal", "Meaning"], mrows, align="llll", note=mac["advice"]))
    prow = [[k, v] for k, v in stg["fit"]["rows"]] + [["Warning", w] for w in stg["fit"]["warnings"]]
    prow += [[k, v] for k, v in stg["guide"]]
    det.append(rpt.Section("Portfolio strategy", ["Item", "Detail"], prow, align="ll",
                           note="Shares to buy = capital x size% / buy price. Caps: business grade, a fixed % of "
                                "capital at risk to the stop-loss, macro regime and your current exposure."))
    return head, det


def _signal_row(sym: str, r: dict) -> list[str]:
    p, side = r["plan"], r["side"]
    why = r["trig_buy"] if side == "BUY" else r["trig_sell"]
    return [
        sym, num(p["entry"]), f"{num(p['stop'])} (risk {p['risk_pct']:.1f}%)",
        f"{num(p['t1'])} (+{p['t1_pct']:.1f}%)", eta_str(p["t1_eta"]),
        f"{num(p['t2'])} (+{p['t2_pct']:.1f}%)", eta_str(p["t2_eta"]),
        f"{r['score']}/9", r["confidence"], _bt_str(r["bt"].get(side)), ", ".join(why + r["notes"]),
    ]


REGIME_MEANING = {
    "Bullish": "Nifty above 50-EMA with Supertrend up: favour BUY setups",
    "Bearish": "Nifty below 50-EMA with Supertrend down: avoid fresh longs, protect profits",
    "Sideways": "Mixed trend: trade smaller, book profits at T1",
    "Unknown": "Index data unavailable",
}


def _fetch(cfg: dict, symbols: list[str], s: dict):
    idx_name, idx_sym = next(iter(cfg["indices"].items()))
    hist = data.fetch_many([idx_sym] + list(symbols), period=s["history"])
    return idx_name, market_regime(hist.get(data.yf_symbol(idx_sym))), hist


# ---------------------------------------------------------------- reports
def scan(cfg: dict, symbols: list[str]) -> rpt.Report:
    s = settings(cfg)
    idx_name, regime, hist = _fetch(cfg, symbols, s)
    res = {sym.upper(): analyze(hist.get(data.yf_symbol(sym)), s, regime) for sym in symbols}
    ok = {k: v for k, v in res.items() if v}
    date = max((v["date"] for v in ok.values()), default=None)
    rep = rpt.Report(title=f"Swing-trading signals - session of {date or 'unknown date'}", session_date=date)
    rep.sections.append(rpt.Section("Market regime", ["Index", "Regime", "Meaning"],
                                    [[idx_name, regime, REGIME_MEANING[regime]]], align="lll"))

    cols = ["Symbol", "Entry", "Stop-loss", "Target 1", "T1 timeline", "Target 2", "T2 timeline",
            "Score", "Confidence", "Backtest (T1 exit)", "Why"]
    al = "lrrrlrlrlll"
    for side, title, note in (
        ("BUY", "BUY signals (swing long)", PLAN_NOTE.format(lo=s["atr_stop_min"], hi=s["atr_stop_max"],
                                                              t1=s["t1_rr"], t2=s["t2_rr"], hold=s["max_hold_days"])),
        ("SELL", "SELL signals (exit longs / short via F&O)", SELL_NOTE),
    ):
        picks = sorted(((k, v) for k, v in ok.items() if v["side"] == side),
                       key=lambda t: ({"High": 0, "Medium": 1, "Low": 2}[t[1]["confidence"]], -t[1]["score"]))
        rows = [_signal_row(k, v) for k, v in picks] or [[f"no {side} signal today"] + [DASH] * (len(cols) - 1)]
        rep.sections.append(rpt.Section(title, cols, rows, note=note, align=al))

    forming = []
    for k, v in ok.items():
        if v["side"]:
            continue
        L = v["L"]
        if v["bull"] >= s["min_score"] - 1 and v["bull"] > v["bear"]:
            miss = [c for c, st in v["conditions"] if st != "Bullish"]
            forming.append((v["bull"], [k, num(v["last"]), "BUY", f"{v['bull']}/9",
                                        f"close above {num(L['hi20'])}", ", ".join(miss) or "fresh trigger"]))
        elif v["bear"] >= s["min_score"] - 1 and v["bear"] > v["bull"]:
            miss = [c for c, st in v["conditions"] if st != "Bearish"]
            forming.append((v["bear"], [k, num(v["last"]), "SELL", f"{v['bear']}/9",
                                        f"close below {num(L['lo20'])}", ", ".join(miss) or "fresh trigger"]))
    forming.sort(key=lambda t: -t[0])
    rep.sections.append(rpt.Section(
        "Setups forming (watch next sessions)", ["Symbol", "LTP", "Bias", "Score", "Trigger level", "Still missing"],
        [row for _, row in forming[:10]] or [["none"] + [DASH] * 5], align="lrlrll",
        note="Not yet a signal: the trend score is close, but a fresh trigger or a condition is missing."))

    missing = [k for k, v in res.items() if not v]
    if missing:
        rep.sections.append(rpt.Section("No / insufficient data", ["Symbols"], [[", ".join(missing)]], align="l"))
    return rep


def evaluate(cfg: dict, symbol: str, df: pd.DataFrame | None, idx_df: pd.DataFrame | None, regime: str,
             s: dict | None = None) -> dict | None:
    """All layers for one stock: chart (swing + holding periods), business, macro, final call and advice."""
    r = analyze(df, s or settings(cfg), regime)
    if not r:
        return None
    r["horizons"] = horizons.analyze_all(df, idx_df)
    r["fund"], r["macro"], r["strategy"] = holistic(cfg, symbol, r, idx_df)
    sym = symbol.upper().removesuffix(".NS")
    r["advice"] = advice.build(r, (r["fund"] or {}).get("name") or data.profile(symbol).get("name") or sym)
    return r


def detail(cfg: dict, symbol: str) -> tuple[rpt.Report, dict, pd.DataFrame] | None:
    """Full single-stock analysis: (report, analysis dict, daily price history)."""
    s = settings(cfg)
    idx_name, regime, hist = _fetch(cfg, [symbol], s)
    df = hist.get(data.yf_symbol(symbol))
    idx_df = hist.get(data.yf_symbol(next(iter(cfg["indices"].values()))))
    r = evaluate(cfg, symbol, df, idx_df, regime, s)
    if not r:
        return None
    L, sym, side = r["L"], symbol.upper(), r["side"]
    head, det = holistic_sections(sym, r["fund"], r["macro"], r["strategy"])
    c = r["last"]
    rep = rpt.Report(title=f"{sym} - analysis: chart, business, macro & strategy ({r['date']})",
                     session_date=r["date"])
    if r["advice"]:
        rep.sections.append(rpt.Section(f"Investment advice: {r['advice']['stance']}", ["Item", "Detail"],
                                        advice.rows(r["advice"]), align="ll",
                                        note="Combines the business analysis, valuation, price trend, macro backdrop "
                                             "and your portfolio. Rule-based, not personalised financial advice."))

    verdict = [
        ["Signal", side or "NO TRADE"],
        ["Close / day change", f"{num(c)}  ({pct(r['chg_pct'])})"],
        ["Bullish score", f"{r['bull']}/9"], ["Bearish score", f"{r['bear']}/9"],
        [f"Market ({idx_name})", regime],
        ["Active triggers", ", ".join(r["trig_buy"] + r["trig_sell"]) or "none"],
        ["Last candle", r["candle"]],
    ]
    if side:
        verdict.insert(1, ["Confidence", r["confidence"]])
        if r["notes"]:
            verdict.append(["Caution", ", ".join(r["notes"])])
    else:
        verdict.insert(1, ["Why no trade", r["why_none"]])
    rep.sections.append(rpt.Section("Verdict", ["Item", "Value"], verdict, align="ll"))

    tv = r["trend"]
    rep.sections.append(rpt.Section(
        f"Trend: {tv['overall']}, {tv['strength']}", ["Horizon", "Trend", "Return", "Reading"],
        [[h, t, f"{pct(rt)} ({per})", rd] for h, t, rt, per, rd in tv["rows"]], align="llrl"))
    out_rows = [["Why no trade today", r["why_none"]]] if not side else []
    out_rows += [[f"Step {i + 1}", t] for i, t in enumerate(r["outlook"]["steps"])]
    if r["outlook"]["when"]:
        out_rows.append(["When (estimate)", r["outlook"]["when"]])
    out_rows += [[f"Last {sd} signal", history_str(sd, h)] for sd, h in r["history"].items()]
    if out_rows:
        rep.sections.append(rpt.Section("Outlook: what to wait for", ["Item", "Detail"], out_rows, align="ll",
                                        note="Time windows are rough ATR-based estimates, not predictions."))

    rep.sections.append(rpt.Section(
        "Holding period: swing vs positional vs long-term",
        ["Style", "Hold for", "Verdict", "Buy at", "Sell targets", "Stop-loss", "Past record"],
        holding_rows(r), align="lllllll",
        note="Each style uses its own timeframe: swing = daily chart, positional = weekly chart, long-term = "
             "primary trend (40-week / 200-day averages, yearly returns)."))
    rep.sections += head
    for h in r["horizons"].values():
        rep.sections.append(horizon_sections(h)[0])
    rep.sections += det

    if side:
        p = r["plan"]
        rows = [
            ["Entry (latest close)", num(p["entry"]), ""],
            ["Stop-loss", num(p["stop"]), f"risk {p['risk_pct']:.2f}% ({num(p['risk'])}/share)"],
            ["Target 1", num(p["t1"]), f"+{p['t1_pct']:.2f}% in {eta_str(p['t1_eta'])}"],
            ["Target 2", num(p["t2"]), f"+{p['t2_pct']:.2f}% in {eta_str(p['t2_eta'])}"],
            ["Reward : risk", f"{s['t1_rr']:g} / {s['t2_rr']:g}", "T1 / T2"],
        ]
        note = PLAN_NOTE.format(lo=s["atr_stop_min"], hi=s["atr_stop_max"], t1=s["t1_rr"], t2=s["t2_rr"],
                                hold=s["max_hold_days"]) + (" " + SELL_NOTE if side == "SELL" else "")
        rep.sections.append(rpt.Section(f"Trade plan ({side})", ["Level", "Price", "Detail"], rows, note=note,
                                        align="lrl"))
    else:
        rep.sections.append(rpt.Section("Watch levels (no trade today)", ["Event", "Level"], [
            ["Bullish trigger: close above 20-day high", num(L["hi20"])],
            ["Bearish trigger: close below 20-day low", num(L["lo20"])],
            ["Supertrend line", num(L["st_line"])],
            ["20-EMA (reclaim / loss)", num(L["ema20"])],
        ], align="lr"))

    wk = {1.0: "last week closed above 20-week EMA", 0.0: "last week closed below 20-week EMA"}
    reading = {
        C_TREND: f"{num(c)} vs {num(L['ema50'])}", C_ALIGN: f"{num(L['ema20'])} vs {num(L['ema50'])}",
        C_LONG: f"{num(c)} vs {num(L['ema200'])}", C_MACD: num(L["macd_hist"]), C_RSI: num(L["rsi"], 1),
        C_ADX: f"ADX {num(L['adx'], 1)}, +DI {num(L['pdi'], 1)}, -DI {num(L['mdi'], 1)}",
        C_ST: f"{'up' if L['st_dir'] == 1 else 'down'}, line {num(L['st_line'])}",
        C_OBV: "OBV above its 20-EMA" if L["obv"] > L["obv_ema"] else "OBV below its 20-EMA",
        C_WK: wk.get(L["wk_up"], "n/a"),
    }
    rep.sections.append(rpt.Section("Technical checklist (score inputs)", ["Condition", "Reading", "Status"],
                                    [[k, reading[k], st] for k, st in r["conditions"]], align="lll"))
    for h in r["horizons"].values():
        rep.sections.append(horizon_sections(h)[1])

    rs, k, d, ad = L["rsi"], L["stoch_k"], L["stoch_d"], L["adx"]
    band = L["bb_up"] - L["bb_lo"]
    pb = (c - L["bb_lo"]) / band if band > 0 else float("nan")
    vr = L["volume"] / L["vol20"] if L["vol20"] > 0 else float("nan")
    hist_rising = L["macd_hist"] > r["macd_hist_prev"]
    rep.sections.append(rpt.Section("Indicators", ["Indicator", "Value", "Reading"], [
        ["EMA 20 / 50 / 200", f"{num(L['ema20'])} / {num(L['ema50'])} / {num(L['ema200'])}",
         "Bullish" if c > L["ema20"] > L["ema50"] > L["ema200"] else
         "Bearish" if c < L["ema20"] < L["ema50"] < L["ema200"] else "Mixed"],
        ["RSI (14)", num(rs, 1), "Overbought (>70)" if rs > 70 else "Oversold (<30)" if rs < 30
         else "Bullish" if rs >= 50 else "Bearish"],
        ["MACD (12,26,9)", f"{num(L['macd'])} / signal {num(L['macd_sig'])} / hist {num(L['macd_hist'])}",
         ("Bullish" if L["macd_hist"] > 0 else "Bearish") + (", momentum rising" if hist_rising else ", momentum fading")],
        ["ADX (14)", f"{num(ad, 1)}  (+DI {num(L['pdi'], 1)} / -DI {num(L['mdi'], 1)})",
         "Strong trend" if ad >= 25 else "Trending" if ad >= 20 else "Range-bound / weak trend"],
        ["Stochastic (14,3,3)", f"%K {num(k, 1)} / %D {num(d, 1)}",
         "Overbought (>80)" if k > 80 else "Oversold (<20)" if k < 20 else "Bullish" if k > d else "Bearish"],
        ["Bollinger (20,2)", f"{num(L['bb_lo'])} / {num(L['bb_mid'])} / {num(L['bb_up'])}",
         f"%B {num(pb)}" + (" - above upper band" if pb > 1 else " - below lower band" if pb < 0 else "")],
        ["ATR (14)", num(L["atr"]), f"{L['atr'] / c * 100:.2f}% average daily range"],
        ["Volume vs 20-day avg", f"{num(vr)}x", "High participation" if vr > 1.5 else "Normal" if vr >= 0.7 else "Low"],
        ["Supertrend (10,3)", num(L["st_line"]), "Bullish" if L["st_dir"] == 1 else "Bearish"],
    ], align="lll"))

    h, lo = float(df["High"].iloc[-1]), float(df["Low"].iloc[-1])
    pv = (h + lo + c) / 3
    w = df.iloc[-252:]
    lv = [[f"Resistance {i + 1} (swing high)", num(v)] for i, v in reversed(list(enumerate(r["res"])))]
    lv += [["Pivot R1 (next session)", num(2 * pv - lo)], ["Pivot", num(pv)], ["Pivot S1 (next session)", num(2 * pv - h)]]
    lv += [[f"Support {i + 1} (swing low)", num(v)] for i, v in enumerate(r["sup"])]
    lv += [["20-day high / low", f"{num(L['hi20'])} / {num(L['lo20'])}"],
           ["52-week high / low", f"{num(w['High'].max())} / {num(w['Low'].min())}"]]
    rep.sections.append(rpt.Section("Support / resistance", ["Level", "Price"], lv, align="lr",
                                    note="Swing levels are fractal highs/lows of the last 120 sessions."))

    bt_rows = []
    for sd in ("BUY", "SELL"):
        b = r["bt"].get(sd)
        bt_rows.append([sd] + ([str(b["n"]), f"{b['win_rate']:.0f}%", f"{b['target_rate']:.0f}%", pct(b["avg_ret"]),
                                f"{pct(b['best'])} / {pct(b['worst'])}", f"{b['avg_days']:.1f}",
                                f"{b['days_to_target']:.1f}" if b["days_to_target"] else DASH]
                               if b else ["0"] + [DASH] * 6))
    rep.sections.append(rpt.Section(
        f"Backtest of these rules on {sym} ({df.index[0].date()} to {r['date']})",
        ["Side", "Trades", "Win rate", "Hit T1", "Avg return", "Best / worst", "Avg days held", "Avg days to T1"],
        bt_rows, align="lrrrrrrr",
        note=f"Entry next open, exit at stop, T1 or after {s['max_hold_days']} sessions. Gross of costs; "
             "past behaviour does not guarantee future results."))
    return rep, r, df
