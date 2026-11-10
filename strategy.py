"""Portfolio strategy: the final call for each holding style after business fundamentals and macro, how much to
put in (as % of capital), how to enter, and how the stock fits the portfolio you already hold.

  * Swing (days-weeks) is chart-driven: fundamentals and macro only change the size.
  * Positional (3-6 months): a strong chart on a weak / loss-making business becomes WAIT.
  * Long-term (1-2 years): the business must be sound - grade D or loss-making becomes AVOID; a good business
    at a fair price may be started small even while the chart says WAIT.
Size = min(cap for the business grade, risk budget / stop distance) x macro factor (Risk-on 1, Neutral 0.75,
Risk-off 0.5), smaller again for a sector headwind or low confidence, and capped by your portfolio's room.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import data
import portfolio as pf
from util import inr, num

GRADE_CAP = {"swing": {"A": 10, "B": 10, "C": 8, "D": 5}, "positional": {"A": 8, "B": 8, "C": 6, "D": 3},
             "longterm": {"A": 10, "B": 8, "C": 5, "D": 0}}
NO_FUND_CAP = {"swing": 8, "positional": 5, "longterm": 5}
RISK_BUDGET = {"swing": 1.0, "positional": 1.5, "longterm": 2.5}  # % of capital lost if the stop-loss is hit
MAX_STOCK, MAX_SECTOR = 10, 30
BUYISH = ("BUY", "BUY ON DIP")


def decide(style: str, technical: str, risk_pct: float | None, confidence: str | None,
           fund: dict | None, mac: dict | None, room: float | None = None, held: bool = False) -> dict:
    g = fund.get("grade") if fund else None
    val = fund["valuation"] if fund else "Unknown"
    final, why, parts, starter, mult = technical, [], 1, False, 1.0
    buyish = technical in BUYISH
    weak = fund is not None and (g == "D" or val == "Loss-making")
    score = f"{fund['score']}/100" if fund and fund.get("score") is not None else "ungraded"
    analyst = fund.get("analyst") if fund else None
    price = fund.get("price") if fund else None
    analyst_target_down = (analyst is not None and analyst.get("total", 0) >= 5 and
                           analyst.get("target_mean") is not None and price and
                           analyst["target_mean"] / price - 1 <= -0.15)
    analyst_sell_consensus = (analyst is not None and analyst.get("total", 0) >= 5 and
                              (analyst.get("consensus") == "Sell consensus" or analyst_target_down))
    mos = fund.get("margin_of_safety_pct") if fund else None
    mos = float(mos) if mos is not None and np.isfinite(mos) else None
    required_mos = 25 if style == "longterm" else 20
    quality = g in ("A", "B")
    investment_call = "BUY" if style == "swing" and buyish else "WATCHLIST"
    value_entry = None
    if fund and fund.get("intrinsic_value") and np.isfinite(fund["intrinsic_value"]):
        value_entry = round(fund["intrinsic_value"] * (1 - required_mos / 100), 2)

    if style in ("longterm", "positional"):
        if not fund or g is None:
            final = "WAIT"
            why.append("Fundamental statements are unavailable or incomplete: keep this on the watchlist until the "
                       "business quality and valuation can be checked.")
        elif fund.get("data_status") != "Current":
            final = "HOLD" if held else "WATCHLIST"
            investment_call = "HOLD" if held else "WATCHLIST"
            why.append("Latest financial statement is stale or its date could not be verified; do not make a new "
                       "long-horizon entry until a current filing is checked.")
        elif weak:
            final = "AVOID"
            investment_call = "SELL / AVOID"
            why.append(f"Weak business ({score}{'; ' + fund['flags'][0] if fund['flags'] else ''}): "
                       "the long-horizon thesis is not supported by profitability and financial quality.")
        elif not quality:
            final = "WATCHLIST"
            why.append(f"Business quality is below the A/B threshold ({score}); wait for stronger, more durable "
                       "returns before a multi-month or multi-year investment.")
        elif mos is None:
            final = "WATCHLIST"
            why.append("Intrinsic value cannot be estimated from the available statements; do not treat missing "
                       "valuation data as a bargain.")
        elif mos >= required_mos:
            if technical == "BUY":
                final = "BUY"
                investment_call = "BUY"
                why.append(f"Grade {g} business is at least {required_mos}% below the estimated intrinsic value; "
                           "the weekly timing check is also positive.")
            elif technical == "BUY ON DIP":
                final = "BUY ON DIP"
                investment_call = "BUY"
                why.append(f"Grade {g} business clears the {required_mos}% margin-of-safety hurdle; use the chart's "
                           "buy zone rather than chasing the current price.")
            else:
                final = "HOLD" if held else "WATCHLIST"
                investment_call = "HOLD" if held else "WATCHLIST"
                why.append(f"Business quality and valuation pass the {required_mos}% margin-of-safety hurdle, but "
                           f"the chart timing is {technical}; "
                           + ("hold and wait for trend stabilization." if held else
                              "wait for trend stabilization before entry."))
        elif mos >= 0:
            final = "HOLD" if held else "WAIT"
            investment_call = "HOLD" if held else "WATCHLIST"
            why.append(f"Quality is good, but the estimated margin of safety is only {mos:.1f}% (need "
                       f"{required_mos}%); {('hold and review results' if held else 'wait for a lower entry')}.")
        else:
            final = "HOLD" if held else "WATCHLIST"
            investment_call = "HOLD" if held else "WATCHLIST"
            why.append(f"Price is above the conservative intrinsic-value estimate ({mos:.1f}% margin of safety); "
                       f"{('hold only while the business thesis remains intact' if held else 'do not start a position')}.")
        if fund:
            method_count = fund.get("value_model_count", 0)
            why.append(f"Valuation blends {method_count}/3 available sector-appropriate models; statements come "
                       "from one unverified provider and should be checked against exchange filings. Key risks: "
                       + ("; ".join(fund.get("flags", [])[:2]) if fund.get("flags") else "no major statement-based flags"))
            if analyst:
                target_upside = ((analyst["target_mean"] / price - 1) * 100
                                 if analyst.get("target_mean") and price else None)
                if analyst_sell_consensus and final in BUYISH:
                    final = "HOLD" if held else "WATCHLIST"
                    investment_call = "HOLD" if held else "WATCHLIST"
                    why.append(f"Analyst data conflicts with a fresh entry ({analyst['bearish_pct']:.0f}% sell-side "
                               f"ratings{f', mean target {target_upside:.0f}% vs current price' if target_upside is not None else ''}); "
                               "wait for updated results or a revised analyst view.")
                elif analyst.get("total", 0) >= 3:
                    why.append(f"Analyst cross-check: {analyst['consensus'].lower()} from {analyst['total']} ratings"
                               + (f", mean target {target_upside:+.0f}% vs current price" if target_upside is not None else "")
                               + "; analyst targets are not intrinsic value.")

    elif style == "swing":
        if weak:
            why.append(f"Weak company ({score}): keep the stop-loss strict and the size small.")
        else:
            why.append("Swing trades follow the chart; fundamentals only adjust the size.")

    active = final in BUYISH or starter
    if mac and active:
        if mac["regime"] == "Risk-off":
            parts = max(parts, 1 if style == "swing" else 3)
            why.append("Market is risk-off: half size" + ("" if style == "swing" else ", buy in 3 parts") + ".")
        elif mac["regime"] == "Neutral":
            parts = max(parts, 1 if style == "swing" else 2)
            why.append("Market backdrop is mixed: about 3/4 size.")
        else:
            why.append("Market backdrop is supportive.")
        if style != "swing" and mac["tilt"] != "Neutral" and mac["impacts"]:
            if mac["tilt"] == "Headwind":
                mult *= 0.75
            why.append(mac["impacts"][0])
    if confidence == "Low" and final in BUYISH:
        mult *= 0.5
        why.append("Low-confidence setup: half size.")

    cap = GRADE_CAP[style][g] if g else NO_FUND_CAP[style]
    size = min(cap, RISK_BUDGET[style] / risk_pct * 100) if risk_pct else cap
    size *= (mac["size_factor"] if mac else 0.75) * mult
    if starter:
        size /= 3
    if not active:
        size = 0.0
    if room is not None and size > room:
        size = room
        why.append(f"Capped at {room:.1f}% by your existing stock / sector exposure.")
    size = round(size, 1)
    if active and size <= 0:
        how = "No room left in your portfolio for this stock / sector: do not add."
    elif starter:
        how = f"Starter position: {size:g}% of capital now; add 2 more parts when the trend turns up."
    elif active:
        how = f"Up to {size:g}% of capital" + (f", in {parts} parts" if parts > 1 else "") + "."
    else:
        how = "Do not buy now."
    return {"style": style, "technical": technical, "final": final, "investment_call": investment_call,
            "value_entry": value_entry, "margin_of_safety_pct": mos, "size_pct": size, "parts": parts,
            "starter": starter, "why": why, "how": how, "analyst_consensus": analyst.get("consensus") if analyst else None,
            "analyst_total": analyst.get("total") if analyst else 0}


# ---------------------------------------------------------------- portfolio fit
def context(cfg: dict, symbol: str) -> dict | None:
    """Your holdings (portfolio.csv) priced at the latest close, with sectors. None when the portfolio is empty."""
    base = Path(cfg["general"]["data_dir"])
    base = base if base.is_absolute() else Path(__file__).resolve().parent / base
    df = pf.load(base / "portfolio.csv")
    if df.empty:
        return None
    hist = data.fetch_many(df["symbol"].tolist() + [symbol], period="1y")
    holdings = []
    for _, h in df.iterrows():
        d = hist.get(data.yf_symbol(h["symbol"]))
        last = float(d["Close"].iloc[-1]) if d is not None and len(d) else float(h["avg_price"])
        holdings.append({"symbol": h["symbol"], "qty": float(h["qty"]), "value": float(h["qty"]) * last,
                         "sector": data.profile(h["symbol"]).get("sector") or "Unknown",
                         "close": d["Close"] if d is not None else None})
    sd = hist.get(data.yf_symbol(symbol))
    return {"holdings": holdings, "close": sd["Close"] if sd is not None else None}


def fit(sym: str, sector: str, ctx: dict | None) -> dict:
    if not ctx:
        return {"rows": [["Your portfolio", "Empty. Add holdings (python agent.py add SYMBOL QTY PRICE) to get a "
                                            "check of stock / sector concentration and diversification."]],
                "warnings": [], "room": None}
    H = ctx["holdings"]
    total = sum(h["value"] for h in H) or 1.0
    own = next((h for h in H if h["symbol"] == sym), None)
    w_own = own["value"] / total * 100 if own else 0.0
    sec_w = sum(h["value"] for h in H if sector and h["sector"] == sector) / total * 100
    rows = [["Your portfolio", f"Rs {inr(total)} in {len(H)} stock(s)"],
            [f"Already holding {sym}", f"{own['qty']:g} shares = {w_own:.1f}% of portfolio" if own else "No"],
            [f"Sector exposure ({sector or 'unknown'})", f"{sec_w:.1f}% now (keep under {MAX_SECTOR}%)"]]
    warnings = []
    others = [h for h in H if h["symbol"] != sym and h["close"] is not None]
    if others and ctx["close"] is not None:
        rets = pd.concat({h["symbol"]: h["close"].pct_change() for h in others}, axis=1)
        w = pd.Series({h["symbol"]: h["value"] for h in others})
        port = (rets * w).sum(axis=1, min_count=1) / rets.notna().mul(w).sum(axis=1).replace(0, np.nan)
        j = pd.concat([port, ctx["close"].pct_change()], axis=1).dropna()
        if len(j) >= 60:
            corr = float(j.corr().iloc[0, 1])
            rows.append(["Correlation with your other holdings (1 year)", f"{corr:.2f}: " + (
                "moves closely with what you own - adds little diversification" if corr >= 0.7 else
                "moderate overlap" if corr >= 0.4 else "low overlap - a good diversifier")])
            if corr >= 0.7:
                warnings.append(f"{sym} moves closely with your current holdings (correlation {corr:.2f}).")
    if sec_w >= MAX_SECTOR:
        warnings.append(f"Your {sector} exposure is already {sec_w:.0f}% (limit {MAX_SECTOR}%): don't add more.")
    if w_own >= MAX_STOCK:
        warnings.append(f"{sym} is already {w_own:.0f}% of your portfolio (limit {MAX_STOCK}%).")
    room = max(0.0, min(MAX_SECTOR - sec_w, MAX_STOCK - w_own))
    return {"rows": rows, "warnings": warnings, "room": room}


def guide(mac: dict | None) -> list[list[str]]:
    rows = [
        ["Core: long-term (1-2 years)", f"50-70% of your share money in 8-12 strong businesses (grade A/B), up to "
                                        f"{MAX_STOCK}% each; review monthly."],
        ["Satellite: positional (3-6 months)", "20-30%, up to 8% each; exit on a weekly close below the 30-week EMA."],
        ["Trading: swing (days-weeks)", "10% or less; risk at most 1% of capital per trade."],
        ["Limits", f"One stock up to {MAX_STOCK}%, one sector up to {MAX_SECTOR}%, keep 5-10% cash (more when the "
                   "market is risk-off). Long-term money you need within 3 years belongs in debt / FDs, not shares."],
    ]
    if mac:
        rows.append([f"Market now: {mac['regime']}", mac["advice"]])
    return rows


def build(specs: list[dict], fund: dict | None, mac: dict | None, sym: str, ctx: dict | None,
          sector: str = "") -> dict:
    """`specs`: [{"style", "label", "period", "technical", "risk_pct", "confidence", "entry"}]."""
    ft = fit(sym, sector, ctx)
    held = any(h["symbol"].upper().removesuffix(".NS") == sym.upper().removesuffix(".NS")
               for h in (ctx or {}).get("holdings", []))
    styles = {}
    for s in specs:
        d = decide(s["style"], s["technical"], s.get("risk_pct"), s.get("confidence"), fund, mac,
                   ft["room"], held)
        styles[s["style"]] = {**d, "label": s["label"], "period": s["period"], "entry": s.get("entry")}
    return {"styles": styles, "fit": ft, "guide": guide(mac)}


def text(stg: dict, sym: str) -> str:
    out = f"**Portfolio strategy for {sym}** (chart + business + macro)\n\n"
    for d in stg["styles"].values():
        out += (f"- **{d['label']} ({d['period']})**: chart says {d['technical']}, final call **{d['final']}**"
                + (" (start small)" if d["starter"] else "") + f". {d['how']} " + " ".join(d["why"]) + "\n")
    out += "\n**Portfolio fit:**\n" + "\n".join(f"- {k}: {v}" for k, v in stg["fit"]["rows"])
    if stg["fit"]["warnings"]:
        out += "\n" + "\n".join(f"- Warning: {w}" for w in stg["fit"]["warnings"])
    out += "\n\n**How to split your money:**\n" + "\n".join(f"- {k}: {v}" for k, v in stg["guide"])
    return out + ("\n\nShares to buy = capital x size% / buy price. Example: Rs 5,00,000 x 4% = Rs 20,000; at "
                  "Rs 1,000 a share that is 20 shares.")
