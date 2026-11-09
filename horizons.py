"""Longer holding periods: Positional (3-6 months) and Long-term (1-2 years).

Swing trading (swing.py) looks for one move over days to a few weeks. Longer holds need slower evidence, so each
style uses its own timeframe:

  * Positional (3-6 months): the WEEKLY chart - 10/30-week EMAs and slope, weekly MACD and RSI, 6-month strength
    vs Nifty, distance from the 52-week high. 7 conditions.
  * Long-term (1-2 years): the primary trend - 40-week EMA (~200-DMA) and its slope, 50/200-DMA golden cross,
    12-month return and strength vs Nifty, position in the 52-week range, higher yearly lows. 8 conditions.

Verdict: BUY (setup on, price not stretched), BUY ON DIP (setup on but stretched above the weekly average),
WAIT (mixed), AVOID (more bearish than bullish). Targets come from what this stock actually did after the same
setup in the past (typical highest price reached within 3/6 months or 1/2 years); with too little history a
reward:risk rule is used. Weekly values use completed weeks only (no look-ahead).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from indicators import ema, macd, rsi
from util import num, pct

STYLES = {
    "positional": {"label": "Positional", "period": "3-6 months", "adj": "3-6 month", "windows": (63, 126),
                   "names": ("3 months", "6 months"), "min_score": 6, "total": 7, "rr": (2.0, 3.5),
                   "stretch_atr": 2.0, "review": "every Friday (weekly close)", "avg": "10-week EMA",
                   "exit_avg": "30-week EMA"},
    "longterm": {"label": "Long-term", "period": "1-2 years", "adj": "1-2 year", "windows": (252, 504),
                 "names": ("1 year", "2 years"), "min_score": 7, "total": 8, "rr": (1.5, 3.0),
                 "stretch_atr": 5.0, "review": "once a month", "avg": "40-week EMA", "exit_avg": "40-week EMA"},
}
NOTE = ("Targets = the typical highest price this stock reached within 3/6 months (positional) or 1/2 years "
        "(long-term) after the same setup in the past, or a reward:risk rule when history is short. Stop-loss is "
        "checked on the weekly (Friday) close. This is the chart view; the final call also weighs the business, "
        "macro backdrop and your portfolio (see Final verdict).")


# ---------------------------------------------------------------- data
def _weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly indicators, labelled by the week's Friday (the running week only appears once it is complete)."""
    w = df.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Close"])
    c, n = w["Close"], np.arange(len(w))
    out = pd.DataFrame(index=w.index)
    out["wc"] = c
    for k in (10, 30, 40):
        out[f"e{k}"] = ema(c, k).where(n >= k - 1)
    out["e30_prev"], out["e40_prev"] = out["e30"].shift(4), out["e40"].shift(13)
    out["wrsi"] = rsi(c)
    out["wmacd"] = macd(c)[2].where(n >= 34)
    pc = c.shift()
    tr = pd.concat([w["High"] - w["Low"], (w["High"] - pc).abs(), (w["Low"] - pc).abs()], axis=1).max(axis=1)
    out["watr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["wlow10"] = w["Low"].rolling(10).min()
    return out


def frame(df: pd.DataFrame, idx_close: pd.Series | None = None) -> pd.DataFrame:
    c = df["Close"]
    x = _weekly(df).reindex(df.index, method="ffill")
    x["close"] = c
    x["sma50"], x["sma200"] = c.rolling(50).mean(), c.rolling(200).mean()
    x["hi252"], x["lo252"] = df["High"].rolling(252).max(), df["Low"].rolling(252).min()
    x["lo252_prev"] = x["lo252"].shift(252)
    ic = idx_close.reindex(df.index, method="ffill") if idx_close is not None else None
    for n in (126, 252):
        x[f"ret{n}"] = c / c.shift(n) - 1
        x[f"iret{n}"] = ic / ic.shift(n) - 1 if ic is not None else np.nan
    return x


def conditions(x: pd.DataFrame, key: str) -> dict:
    """label -> (bullish, bearish, reading of the last bar). NaN compares False, so missing data is Neutral."""
    L, c = x.iloc[-1], x["close"]

    def vs(a: str, b: str) -> str:
        return f"{num(L[a])} vs {num(L[b])}"

    def rel(n: int) -> tuple[pd.Series, str]:
        has_idx = x[f"iret{n}"].notna().any()
        diff = x[f"ret{n}"] - (x[f"iret{n}"] if has_idx else 0)
        txt = f"stock {pct(L[f'ret{n}'] * 100, 1)}" + (f" vs Nifty {pct(L[f'iret{n}'] * 100, 1)}" if has_idx else "")
        return diff, txt

    from_hi = f"{pct((L['close'] / L['hi252'] - 1) * 100, 1)} from 52-week high {num(L['hi252'])}"
    if key == "positional":
        rs, rs_txt = rel(126)
        return {
            "Weekly close above 10-week EMA": (x["wc"] > x["e10"], x["wc"] < x["e10"], vs("wc", "e10")),
            "10-week EMA above 30-week EMA": (x["e10"] > x["e30"], x["e10"] < x["e30"], vs("e10", "e30")),
            "30-week EMA rising (vs 4 weeks ago)": (x["e30"] > x["e30_prev"], x["e30"] < x["e30_prev"],
                                                    vs("e30", "e30_prev")),
            "Weekly MACD histogram positive": (x["wmacd"] > 0, x["wmacd"] < 0, num(L["wmacd"])),
            "Weekly RSI(14) in 50-75": (x["wrsi"].between(50, 75), x["wrsi"] < 45, num(L["wrsi"], 1)),
            "6-month return beats Nifty": (rs > 0, rs < 0, rs_txt),
            "Within 20% of 52-week high": (c >= 0.8 * x["hi252"], c < 0.65 * x["hi252"], from_hi),
        }
    rs, rs_txt = rel(252)
    return {
        "Weekly close above 40-week EMA (~200-DMA)": (x["wc"] > x["e40"], x["wc"] < x["e40"], vs("wc", "e40")),
        "40-week EMA rising (vs 13 weeks ago)": (x["e40"] > x["e40_prev"], x["e40"] < x["e40_prev"],
                                                 vs("e40", "e40_prev")),
        "50-DMA above 200-DMA (golden cross)": (x["sma50"] > x["sma200"], x["sma50"] < x["sma200"],
                                                vs("sma50", "sma200")),
        "12-month return positive": (x["ret252"] > 0, x["ret252"] < 0, pct(L["ret252"] * 100, 1)),
        "12-month return beats Nifty": (rs > 0, rs < 0, rs_txt),
        "Within 25% of 52-week high": (c >= 0.75 * x["hi252"], c < 0.6 * x["hi252"], from_hi),
        "At least 30% above 52-week low": (c >= 1.3 * x["lo252"], c < 1.1 * x["lo252"],
                                           f"{pct((L['close'] / L['lo252'] - 1) * 100, 1)} above low {num(L['lo252'])}"),
        "Higher 52-week low than last year": (x["lo252"] > x["lo252_prev"], x["lo252"] < x["lo252_prev"],
                                              vs("lo252", "lo252_prev")),
    }


# ---------------------------------------------------------------- helpers
def _forward(c: np.ndarray, hi: np.ndarray, setup: np.ndarray, w: int) -> dict | None:
    """Outcome `w` sessions after every past setup day (needs >= 20 days from >= 2 separate setups).
    `median` = close-to-close return; `peak` = median of the highest price reached within the window."""
    idx = np.flatnonzero(setup[: max(len(c) - w, 0)])
    if len(idx) < 20:
        return None
    episodes = int(1 + np.sum(np.diff(idx) > 20))
    if episodes < 2:
        return None
    rets = (c[idx + w] / c[idx] - 1) * 100
    peaks = (np.array([hi[i + 1: i + w + 1].max() for i in idx]) / c[idx] - 1) * 100
    return {"days": len(idx), "episodes": episodes, "median": float(np.median(rets)),
            "p25": float(np.percentile(rets, 25)), "p75": float(np.percentile(rets, 75)),
            "win": float(np.mean(rets > 0) * 100), "peak": float(np.median(peaks))}


def _typical_wait(score: np.ndarray, setup: np.ndarray) -> tuple[int, float] | None:
    """From past days with today's score (and no setup): median sessions until the next setup, and how often one came."""
    sidx = np.flatnonzero(setup)
    cand = np.flatnonzero((score == score[-1]) & ~setup)[:-1]
    if len(cand) < 20 or not len(sidx):
        return None
    pos = np.searchsorted(sidx, cand)
    reached = pos < len(sidx)
    if reached.sum() < 10:
        return None
    return int(np.median(sidx[pos[reached]] - cand[reached])), float(reached.mean() * 100)


def _span(sessions: int) -> str:
    return (f"~{sessions} sessions" if sessions <= 10 else f"~{round(sessions / 5)} weeks" if sessions <= 60
            else f"~{sessions / 21:.0f} months")


def _by(start, sessions: int):
    return pd.Timestamp(np.busday_offset(np.datetime64(start), sessions, roll="forward")).date()


def _weeks(dist: float, watr: float) -> str:
    if not watr or watr <= 0 or math.isnan(watr):
        return "unknown"
    fast = max(1, math.ceil(abs(dist) / (0.8 * watr)))
    return f"~{fast}-{max(fast + 1, math.ceil(abs(dist) / (0.3 * watr)))} weeks"


def _last_episode(dates, c: np.ndarray, setup: np.ndarray) -> dict | None:
    idx = np.flatnonzero(setup)
    if not len(idx):
        return None
    gaps = np.flatnonzero(np.diff(idx) > 20)
    start = int(idx[gaps[-1] + 1] if len(gaps) else idx[0])
    end = int(idx[-1])
    return {"start": dates[start].date(), "start_price": float(c[start]), "end": dates[end].date(),
            "since_pct": (c[-1] / c[start] - 1) * 100, "active": end == len(c) - 1,
            "per_year": (1 + len(gaps)) / max(len(c) / 252, 0.5)}


# ---------------------------------------------------------------- analysis
def analyze(df: pd.DataFrame, key: str, idx_close: pd.Series | None = None) -> dict | None:
    st = STYLES[key]
    if df is None or len(df) < 300:
        return None
    x = frame(df, idx_close)
    cond = conditions(x, key)
    bull = sum(b.astype(int) for b, _, _ in cond.values())
    bear = sum(r.astype(int) for _, r, _ in cond.values())
    ref = "e10" if key == "positional" else "e40"
    limit = x[ref] + st["stretch_atr"] * x["watr"]
    stretched = x["close"] > limit
    setup = ((bull >= st["min_score"]) & ~stretched & (x["watr"] > 0)).to_numpy()
    good = (bull >= st["min_score"]).to_numpy()

    L, c = x.iloc[-1], x["close"].to_numpy(dtype=float)
    last, date, watr = float(c[-1]), x.index[-1].date(), float(L["watr"])
    nb, nr, tot = int(bull.iloc[-1]), int(bear.iloc[-1]), st["total"]
    verdict = ("BUY" if setup[-1] else "BUY ON DIP" if good[-1] else "AVOID" if nr > nb else "WAIT")
    stats = [_forward(c, df["High"].to_numpy(dtype=float), setup, w) for w in st["windows"]]
    h = {
        "key": key, **{k: st[k] for k in ("label", "period", "adj", "names", "review", "total", "min_score")},
        "verdict": verdict, "bull": nb, "bear": nr, "date": date, "last": last, "stats": stats,
        "conditions": [(k, "Bullish" if b.iloc[-1] else "Bearish" if r.iloc[-1] else "Neutral", rd)
                       for k, (b, r, rd) in cond.items()],
        "episode": _last_episode(x.index, c, setup), "plan": None, "steps": [], "when": "",
    }

    def pc(level: float) -> str:
        return f"{(level / last - 1) * 100:+.1f}%"

    # trade plan (also for BUY ON DIP, from the top of the buy zone)
    if verdict in ("BUY", "BUY ON DIP") and watr > 0:
        entry = last if verdict == "BUY" else float(limit.iloc[-1])
        if key == "positional":
            stop = min(max(float(L["wlow10"]) - 0.25 * watr, entry - 2.5 * watr), entry - 1.5 * watr)
        else:
            stop = min(max(float(L["e40"]) - watr, entry * 0.70), entry * 0.88)
        risk = entry - stop
        m1, m2 = (s["peak"] if s else None for s in stats)
        t1 = entry * (1 + m1 / 100) if m1 and m1 > 0 else entry + st["rr"][0] * risk
        t2 = (entry * (1 + m2 / 100) if m2 and m1 and m2 > m1 > 0
              else max(entry + st["rr"][1] * risk, t1 + 0.5 * risk))
        h["plan"] = {"entry": entry, "stop": stop, "risk_pct": risk / entry * 100, "t1": t1, "t2": t2,
                     "t1_pct": (t1 / entry - 1) * 100, "t2_pct": (t2 / entry - 1) * 100,
                     "by1": _by(date, st["windows"][0]), "by2": _by(date, st["windows"][1]),
                     "source": "history" if m1 and m1 > 0 else "rule"}
        s1 = stats[0]
        h["confidence"] = ("Low" if not s1 or s1["win"] < 50 or s1["median"] <= 0 or t1 - entry < 0.75 * risk else
                           "High" if s1["win"] >= 65 and nb == tot and t1 - entry >= 1.5 * risk
                           else "Medium")

    # why / what to wait for / when
    avg_lv = float(L[ref])
    if verdict == "BUY":
        h["why"] = f"{nb}/{tot} {st['adj']} conditions are bullish and the price is not stretched."
    elif verdict == "BUY ON DIP":
        lv = float(limit.iloc[-1])
        h["why"] = (f"Strong {st['adj']} trend ({nb}/{tot}) but the price is "
                    f"{(last - avg_lv) / watr:.1f} weekly ATR above the {st['avg']}: buying here risks a pullback.")
        h["steps"] = [f"Buy zone: Rs {num(avg_lv)} - Rs {num(lv)} ({pc(avg_lv)} to {pc(lv)}), near the {st['avg']}.",
                      "Or buy in 2-3 parts: a small part now, the rest on dips into the zone."]
        h["when"] = (f"A dip of that size usually takes {_weeks(last - lv, watr)}, or the {st['avg']} catches up "
                     "after a few sideways weeks.")
    else:
        if verdict == "AVOID":
            h["why"] = f"{nr}/{tot} conditions are bearish: the {st['adj']} trend is down. Avoid fresh buying."
        else:
            h["why"] = f"Only {nb}/{tot} conditions are bullish (need {st['min_score']}): the {st['adj']} trend is not clear."
        lv_need = {
            "Weekly close above 10-week EMA": (float(L["e10"]), f"a weekly close above the 10-week EMA Rs {num(L['e10'])} ({pc(L['e10'])})"),
            "Weekly close above 40-week EMA (~200-DMA)": (float(L["e40"]), f"a weekly close above the 40-week EMA Rs {num(L['e40'])} ({pc(L['e40'])})"),
            "Within 20% of 52-week high": (0.8 * float(L["hi252"]), f"price back above Rs {num(0.8 * L['hi252'])} (within 20% of the 52-week high)"),
            "Within 25% of 52-week high": (0.75 * float(L["hi252"]), f"price back above Rs {num(0.75 * L['hi252'])} (within 25% of the 52-week high)"),
        }
        other = {
            "10-week EMA above 30-week EMA": f"the 10-week EMA (Rs {num(L['e10'])}) to cross above the 30-week EMA (Rs {num(L['e30'])})",
            "30-week EMA rising (vs 4 weeks ago)": "the 30-week EMA to turn up",
            "Weekly MACD histogram positive": "weekly MACD to turn positive",
            "Weekly RSI(14) in 50-75": f"weekly RSI back into 50-75 (now {L['wrsi']:.0f})",
            "6-month return beats Nifty": "the stock to outperform Nifty over 6 months",
            "40-week EMA rising (vs 13 weeks ago)": "the 40-week EMA to turn up",
            "50-DMA above 200-DMA (golden cross)": f"a golden cross: 50-DMA (Rs {num(L['sma50'])}) above 200-DMA (Rs {num(L['sma200'])})",
            "12-month return positive": "a positive 12-month return",
            "12-month return beats Nifty": "the stock to outperform Nifty over 12 months",
            "At least 30% above 52-week low": f"price at least 30% above the 52-week low (Rs {num(1.3 * L['lo252'])})",
            "Higher 52-week low than last year": "a higher yearly low (no new lows below last year's)",
        }
        miss = [k for k, s, _ in h["conditions"] if s != "Bullish"]
        items = [lv_need[k][1] if k in lv_need else other[k] for k in miss]
        need = max(st["min_score"] - nb, 0)
        if need:
            h["steps"].append(f"To become a {st['adj']} BUY it needs {need} more of: " + "; ".join(items) + ".")
        key_lv = min((lv_need[k][0] for k in miss if k in lv_need and lv_need[k][0] > last), default=None)
        tw = _typical_wait(bull.to_numpy(), setup)
        parts = []
        if key_lv:
            parts.append(f"Next price level to watch: Rs {num(key_lv)} ({pc(key_lv)}), {_weeks(key_lv - last, watr)} "
                         "away at a normal pace.")
        if tw:
            parts.append(f"From a similar state ({nb}/{tot} bullish) this stock took a median {_span(tw[0])} to turn "
                         f"into a {st['adj']} BUY setup (it happened in {tw[1]:.0f}% of past cases).")
        elif need >= 3:
            parts.append("Slow conditions (average crossovers, yearly returns) are missing, so a full setup usually "
                         "needs a few months to rebuild.")
        else:
            parts.append("The slower conditions usually need several weeks to catch up if the price holds.")
        h["when"] = " ".join(parts) + " Re-check weekly."

    exit_col = "e30" if key == "positional" else "e40"
    ex = float(L[exit_col])
    h["holder"] = (f"Already holding? Hold while the weekly close stays above the {st['exit_avg']} Rs {num(ex)} ({pc(ex)})."
                   if L["wc"] > ex else
                   f"Already holding? The weekly close is below the {st['exit_avg']} Rs {num(ex)}: the {st['adj']} "
                   "trend is broken - consider exiting or reducing on bounces.")
    h["record"] = record_str(h)
    return h


def record_str(h: dict) -> str:
    s1, s2 = h["stats"]
    if not s1:
        return "Past record: not enough history of this setup on this stock to measure."
    out = (f"Past record: after {s1['episodes']} earlier setups, {h['names'][0]} later the stock was up "
           f"{s1['win']:.0f}% of the time (median {pct(s1['median'], 1)}, range {pct(s1['p25'], 0)} to {pct(s1['p75'], 0)})")
    if s2:
        out += f"; {h['names'][1]} later up {s2['win']:.0f}% (median {pct(s2['median'], 1)})"
    return out + "."


def analyze_all(df: pd.DataFrame, idx_df: pd.DataFrame | None) -> dict:
    ic = idx_df["Close"] if idx_df is not None and len(idx_df) else None
    return {k: v for k in STYLES if (v := analyze(df, k, ic))}


def text(h: dict, sym: str) -> str:
    """Markdown explanation for the chat."""
    out = f"**{h['label']} ({h['period']}) view for {sym}: {h['verdict']}**\n\n{h['why']}"
    p = h.get("plan")
    if p:
        at = "Buy around" if h["verdict"] == "BUY" else "If bought in the dip zone at"
        out += (f"\n\n{at} **Rs {num(p['entry'])}**: sell at **Rs {num(p['t1'])} ({pct(p['t1_pct'], 1)})** around "
                f"{p['by1']:%b %Y} or **Rs {num(p['t2'])} ({pct(p['t2_pct'], 1)})** around {p['by2']:%b %Y}. "
                f"Stop-loss: weekly close below **Rs {num(p['stop'])}** (-{p['risk_pct']:.1f}%). "
                f"Confidence {h['confidence']}. Review {h['review']}.")
    if h["steps"]:
        out += "\n\n**What to wait for:**\n" + "\n".join(f"- {t}" for t in h["steps"])
    if h["when"]:
        out += f"\n\n**When (estimate):** {h['when']}"
    return out + f"\n\n{h['record']}\n\n{h['holder']}"
