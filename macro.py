"""Macro backdrop for Indian shares and for the stock's own sector.

Market-wide factors (each Good / Neutral / Bad):
  * Nifty 50 primary trend: close vs 200-DMA and 50-DMA vs 200-DMA (bull market / correction / bear market).
  * India VIX: expected volatility (calm < 15, fear > 22).
  * Foreign flows proxy: USD/INR and the US 10-year yield over 3 months. A weaker rupee or sharply higher US
    yields usually mean foreign investors (FIIs) sell Indian shares.
  * Brent crude over 3 months: India imports most of its oil, so a spike hurts inflation and margins.
They add up to a regime: Risk-on (full size), Neutral (~3/4 size), Risk-off (half size, buy in parts).

Stock-specific: the sector index trend vs Nifty, and known sensitivities (IT / pharma gain from a weak rupee,
paints / tyres / airlines lose from costly crude, banks / realty are rate-sensitive, ...).
"""
from __future__ import annotations

import math

import pandas as pd

import data
from util import num, pct

NAN = float("nan")
SYMBOLS = {"vix": "^INDIAVIX", "inr": "INR=X", "crude": "BZ=F", "us10y": "^TNX"}
SECTOR_INDEX = [
    (("bank",), "Bank Nifty", "^NSEBANK"),
    (("credit services", "insurance", "capital markets", "asset management", "mortgage", "financial"),
     "Nifty Financial Services", "NIFTY_FIN_SERVICE.NS"),
    (("software", "information technology", "it services", "technology"), "Nifty IT", "^CNXIT"),
    (("drug", "pharma", "biotech", "medical", "healthcare", "diagnostic"), "Nifty Pharma", "^CNXPHARMA"),
    (("auto",), "Nifty Auto", "^CNXAUTO"),
    (("steel", "aluminum", "metal", "mining", "copper"), "Nifty Metal", "^CNXMETAL"),
    (("real estate",), "Nifty Realty", "^CNXREALTY"),
    (("oil", "gas", "energy", "utilities", "power", "coal", "solar"), "Nifty Energy", "^CNXENERGY"),
    (("consumer defensive", "household", "packaged food", "tobacco", "beverage", "personal product",
      "confectioner"), "Nifty FMCG", "^CNXFMCG"),
    (("engineering", "construction", "infrastructure", "industrials"), "Nifty Infra", "^CNXINFRA"),
]
# (keywords in industry/sector, factor, +1 = benefits when the factor rises, reason)
SENSITIVITY = [
    (("software", "information technology", "it services", "drug", "pharma", "biotech"), "inr", 1,
     "earns much of its revenue in dollars, so a weaker rupee lifts profits"),
    (("steel", "aluminum", "metal", "mining"), "inr", 1, "sells at dollar-linked prices, so a weaker rupee helps"),
    (("oil & gas e&p", "oil & gas drilling"), "crude", 1, "sells oil and gas, so higher crude prices lift profits"),
    (("refining", "specialty chemical", "chemicals", "paint", "airline", "tire", "tyre", "auto parts",
      "auto manufacturers", "packaging", "fertilizer"), "crude", -1, "crude oil is a key input cost"),
    (("airline",), "inr", -1, "pays fuel and leases in dollars, so a weaker rupee hurts"),
    (("bank", "credit", "insurance", "capital markets", "real estate", "utilities", "mortgage"), "yield", -1,
     "is rate-sensitive: higher yields and foreign selling weigh on its valuation"),
]
DEFENSIVE = ("consumer defensive", "household", "packaged food", "tobacco", "drug", "pharma", "utilities")
FACTOR_TXT = {"inr": ("the rupee is weakening", "the rupee is strengthening"),
              "crude": ("crude oil is rising", "crude oil is falling"),
              "yield": ("US bond yields are rising", "US bond yields are falling")}
ADVICE = {
    "Risk-on": "Supportive market: normal position sizes, stay invested, buy dips in strong stocks.",
    "Neutral": "Mixed market: use about 3/4 of your normal size and buy in 2 parts.",
    "Risk-off": "Weak / nervous market: half size, buy in 3 parts, keep 20-30% cash and prefer strong businesses.",
}


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _close(df) -> pd.Series | None:
    return df["Close"].dropna() if df is not None and len(df) else None


def _chg(df, n: int = 63) -> float:
    c = _close(df)
    return (float(c.iloc[-1]) / float(c.iloc[-1 - n]) - 1) * 100 if c is not None and len(c) > n else NAN


def sector_index(sector: str, industry: str) -> tuple[str, str] | None:
    for text in (industry.lower(), sector.lower()):
        for words, name, sym in SECTOR_INDEX:
            if text and any(w in text for w in words):
                return name, sym
    return None


def analyze(nifty: pd.DataFrame | None, series: dict, sector_df: tuple[str, pd.DataFrame | None] | None = None,
            sector: str = "", industry: str = "") -> dict | None:
    """`series`: {"vix"|"inr"|"crude"|"us10y": daily DataFrame}; `sector_df`: (index name, its daily DataFrame)."""
    c = _close(nifty)
    if c is None or len(c) < 210:
        return None
    rows, pts = [], 0
    last, d200, d50 = float(c.iloc[-1]), float(c.rolling(200).mean().iloc[-1]), float(c.rolling(50).mean().iloc[-1])
    n3 = _chg(nifty)
    if last > d200 and d50 > d200:
        trend, sig, p = "Bull market", "Good", 1
    elif last < d200 and d50 < d200:
        trend, sig, p = "Bear market", "Bad", -1
    elif last > d200:
        trend, sig, p = "Recovering (above 200-DMA, 50-DMA still below)", "Neutral", 0
    else:
        trend, sig, p = "Correction (below 200-DMA)", "Neutral", 0
    pts += p
    rows.append(["Nifty 50 trend", f"{trend}: {num(last)} vs 200-DMA {num(d200)}; 3 months {pct(n3, 1)}", sig,
                 "Most shares follow the index: buy-side setups work best in a bull market."])

    vc = _close(series.get("vix"))
    if vc is not None and len(vc):
        v = float(vc.iloc[-1])
        sig, p, txt = (("Good", 1, "calm market, low fear") if v < 15 else ("Bad", -1, "fear is high: sharp swings likely")
                       if v > 22 else ("Neutral", 0, "normal volatility"))
        pts += p
        rows.append(["India VIX (fear gauge)", f"{v:.1f}", sig, txt.capitalize() + "."])

    inr, yld = _chg(series.get("inr")), NAN
    yc = _close(series.get("us10y"))
    if yc is not None and len(yc) > 63:
        yld = (float(yc.iloc[-1]) - float(yc.iloc[-64])) * 100  # basis points
    flow = 0
    if (not _nan(inr) and inr > 2) or (not _nan(yld) and yld > 35):
        flow = -1
    elif not _nan(inr) and inr < -1 and not _nan(yld) and yld < -25:
        flow = 1
    pts += flow
    if not _nan(inr):
        rows.append(["Rupee (USD/INR, 3 months)", f"{pct(inr, 1)} ({'weaker' if inr > 0 else 'stronger'} rupee)",
                     "Bad" if inr > 2 else "Good" if inr < -1 else "Neutral",
                     "A fast-falling rupee usually comes with foreign investors (FIIs) selling Indian shares."])
    if not _nan(yld):
        rows.append(["US 10-year yield (3 months)", f"{float(yc.iloc[-1]):.2f}% ({yld:+.0f} bps)",
                     "Bad" if yld > 35 else "Good" if yld < -25 else "Neutral",
                     "Higher US yields pull foreign money out of emerging markets like India."])
    crude = _chg(series.get("crude"))
    if not _nan(crude):
        cc = _close(series.get("crude"))
        p = -1 if crude > 12 else 1 if crude < -12 else 0
        pts += p
        rows.append(["Brent crude (3 months)", f"${float(cc.iloc[-1]):.1f} ({pct(crude, 1)})",
                     {1: "Good", -1: "Bad", 0: "Neutral"}[p],
                     "India imports most of its oil: costly crude raises inflation and squeezes margins."])

    regime = "Risk-on" if pts >= 2 else "Risk-off" if pts <= -1 else "Neutral"
    size = {"Risk-on": 1.0, "Neutral": 0.75, "Risk-off": 0.5}[regime]

    # sector + stock sensitivities
    sec_pts, sec_name, impacts = 0, None, []
    if sector_df and _close(sector_df[1]) is not None and len(_close(sector_df[1])) > 200:
        sec_name, sc = sector_df[0], _close(sector_df[1])
        rel = _chg(sector_df[1], 126) - _chg(nifty, 126)
        above = float(sc.iloc[-1]) > float(sc.rolling(200).mean().iloc[-1])
        sec_pts = 1 if rel > 3 and above else -1 if rel < -3 and not above else 0
        state = "leading" if rel > 3 else "lagging" if rel < -3 else "in line with"
        rows.append([f"Sector: {sec_name}", f"6 months vs Nifty {pct(rel, 1)}; {'above' if above else 'below'} its 200-DMA",
                     {1: "Good", -1: "Bad", 0: "Neutral"}[sec_pts],
                     f"The sector is {state} the market: money is {'flowing in' if sec_pts > 0 else 'flowing out' if sec_pts < 0 else 'neutral'}."])
        if sec_pts:
            impacts.append(f"{'Tailwind' if sec_pts > 0 else 'Headwind'}: the {sec_name} index is {state} Nifty.")
    text = f"{industry} {sector}".lower()
    moves = {"inr": 0 if _nan(inr) else 1 if inr > 2 else -1 if inr < -2 else 0,
             "crude": 0 if _nan(crude) else 1 if crude > 12 else -1 if crude < -12 else 0,
             "yield": 0 if _nan(yld) else 1 if yld > 35 else -1 if yld < -35 else 0}
    stock_pts = sec_pts
    for words, factor, sign, why in SENSITIVITY:
        if any(w in text for w in words) and moves[factor]:
            imp = sign * moves[factor]
            stock_pts += imp
            impacts.append(f"{'Tailwind' if imp > 0 else 'Headwind'}: {FACTOR_TXT[factor][0 if moves[factor] > 0 else 1]} "
                           f"and this company {why}.")
    if regime == "Risk-off" and any(w in text for w in DEFENSIVE):
        stock_pts += 1
        impacts.append("Tailwind: defensive business, which usually holds up better in weak markets.")
    tilt = "Tailwind" if stock_pts >= 1 else "Headwind" if stock_pts <= -1 else "Neutral"
    summary = (f"{regime} market (Nifty: {trend}" + (f", VIX {float(vc.iloc[-1]):.0f}" if vc is not None and len(vc) else "")
               + f"); for this stock the macro / sector picture is a {tilt.lower()}.")
    return {"regime": regime, "size_factor": size, "points": pts, "trend": trend, "rows": rows, "sector": sec_name,
            "impacts": impacts, "tilt": tilt, "advice": ADVICE[regime], "summary": summary}


def fetch(nifty: pd.DataFrame | None, sector: str = "", industry: str = "") -> dict | None:
    sec = sector_index(sector, industry)
    syms = list(SYMBOLS.values()) + ([sec[1]] if sec else [])
    hist = data.fetch_many(syms, period="2y")
    series = {k: hist.get(data.yf_symbol(s)) for k, s in SYMBOLS.items()}
    return analyze(nifty, series, (sec[0], hist.get(data.yf_symbol(sec[1]))) if sec else None, sector, industry)


def text(m: dict | None) -> str:
    if not m:
        return "Macro data is not available right now."
    out = f"**Macro backdrop: {m['regime']}**\n\n{m['summary']}\n\n"
    out += "\n".join(f"- {f}: {rd} (**{s}**) - {why}" for f, rd, s, why in m["rows"])
    if m["impacts"]:
        out += "\n\n**For this stock:**\n" + "\n".join(f"- {t}" for t in m["impacts"])
    return out + f"\n\n**What to do:** {m['advice']}"
