"""Stock screener: run the full analysis (chart, business, valuation, results, macro, final call) on a list of
stocks and rank them into the top 5 / top 10 to buy for a holding period.

Score (0-100) = weighted sum of six factor scores, each 0-100. Weights depend on the holding period:

                 Business  Valuation  Trend  Results  Macro/sector  Final call
  Swing             10         5        45      10         10           20
  Positional        20        10        35      10         10           15
  Long-term         35        20        20      10          5           10

Only stocks whose final call for that period is a buy (BUY / BUY ON DIP / starter with size > 0) appear in the
"to buy" list; the best of the rest are shown as "next best (not a buy yet)".
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import data
import swing
from strategy import BUYISH
from util import DASH, num, pct

UNIVERSES = ["Nifty 50", "Nifty Next 50", "Nifty 100", "Nifty Midcap 50", "Nifty 500", "My watchlist (config.toml)"]
STYLES = {"swing": "Swing (days to weeks)", "positional": "Positional (3-6 months)",
          "longterm": "Long-term (1-2 years)"}
WEIGHTS = {
    "swing": {"Business": 10, "Valuation": 5, "Trend": 45, "Results": 10, "Macro": 10, "Final call": 20},
    "positional": {"Business": 20, "Valuation": 10, "Trend": 35, "Results": 10, "Macro": 10, "Final call": 15},
    "longterm": {"Business": 35, "Valuation": 20, "Trend": 20, "Results": 10, "Macro": 5, "Final call": 10},
}
VAL_PTS = {"Attractive": 100, "Fair": 65, "Expensive": 25, "Loss-making": 0}
TILT_PTS = {"Tailwind": 100, "Neutral": 50, "Headwind": 0}
CALL_PTS = {"BUY": 100, "BUY ON DIP": 75, "WAIT": 25}
COLUMNS = ["Rank", "Stock", "Company", "Sector", "Score", "Final call", "Size", "Buy at", "Target 1",
           "Stop-loss", "Business", "Valuation", "Trend", "Why"]


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def members(cfg: dict, universe: str) -> list[tuple[str, str]]:
    """[(SYMBOL, Company name)] for a universe; [] when the index list can't be downloaded."""
    if universe in data.INDEX_FILES:
        return [(s, n) for s, n, _ in data.index_members(universe)]
    names = dict(data.nse_companies())
    syms = [s.upper().removesuffix(".NS") for s in cfg.get("watchlist", {}).get("symbols", [])]
    return [(s, names.get(s, s)) for s in syms]


# ---------------------------------------------------------------- per-stock summary
def summarize(sym: str, name: str, r: dict) -> dict:
    """Compact, cache-friendly view of swing.evaluate() output."""
    f, m, stg = r.get("fund") or {}, r.get("macro") or {}, r["strategy"]["styles"]
    styles = {}
    for k, d in stg.items():
        if k == "swing":
            bull, total, p = r["bull"], 9, r.get("plan") if r["side"] == "BUY" else None
        else:
            h = r["horizons"].get(k)
            if not h:
                continue
            bull, total, p = h["bull"], h["total"], h["plan"]
        styles[k] = {"technical": d["technical"], "final": d["final"], "starter": d["starter"],
                     "size": d["size_pct"], "parts": d["parts"], "why": d["why"], "bull": bull, "total": total,
                     "entry": p["entry"] if p else None, "stop": p["stop"] if p else None,
                     "t1": p["t1"] if p else None, "t1_pct": p["t1_pct"] if p else None,
                     "risk_pct": p["risk_pct"] if p else None}
    adv = r.get("advice") or {}
    return {"sym": sym, "name": f.get("name") or name, "sector": f.get("sector") or data.profile(sym).get("sector") or "",
            "price": r["last"], "chg": r["chg_pct"], "date": r["date"], "fscore": f.get("score"),
            "grade": f.get("grade"), "grade_word": f.get("grade_word"), "valuation": f.get("valuation") or "Unknown",
            "pe": f.get("pe"), "roe": f.get("roe"), "rev_cagr": f.get("rev_cagr"), "q_ni_yoy": f.get("q_ni_yoy"),
            "strengths": f.get("strengths") or [], "flags": f.get("flags") or [], "tilt": m.get("tilt", "Neutral"),
            "sector_index": m.get("sector"), "trend": r["trend"]["overall"], "stance": adv.get("stance", DASH),
            "headline": adv.get("headline", ""), "styles": styles}


def run(cfg: dict, stocks: list[tuple[str, str]], progress=None, workers: int = 6, stop=None) -> dict:
    """Evaluate every stock. `progress(done, total, symbol)` is called from the calling thread; when `stop()` returns
    True the remaining stocks are skipped (returned with 'stopped': True)."""
    s = swing.settings(cfg)
    idx_sym = next(iter(cfg["indices"].values()))
    idx_df = data.history(idx_sym, s["history"])
    regime = swing.market_regime(idx_df)

    def one(sym: str, name: str):
        if stop and stop():
            return "stopped", None
        try:
            data.financials(sym)
            r = swing.evaluate(cfg, sym, data.history(sym, s["history"]), idx_df, regime, s)
            return (summarize(sym, name, r), r.get("macro")) if r else (None, None)
        except Exception:
            return None, None

    rows, failed, mac, skipped = [], [], None, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, sym, name): sym for sym, name in stocks}
        for i, fut in enumerate(as_completed(futs), 1):
            row, m = fut.result()
            if row == "stopped":
                skipped += 1
            elif row:
                rows.append(row)
                mac = mac or m
            else:
                failed.append(futs[fut])
            if progress:
                progress(i, len(futs), futs[fut])
    date = max((x["date"] for x in rows), default=None)
    market = {k: mac[k] for k in ("regime", "trend", "advice", "size_factor")} if mac else None
    return {"rows": rows, "failed": sorted(failed), "date": date, "market": market, "stopped": skipped > 0}


# ---------------------------------------------------------------- scoring and ranking
def is_buy(st: dict) -> bool:
    return (st["final"] in BUYISH or st["starter"]) and st["size"] > 0


def parts(row: dict, style: str) -> dict:
    """Factor scores 0-100."""
    st = row["styles"][style]
    q = row["q_ni_yoy"]
    call = 60 if st["starter"] else CALL_PTS.get(st["final"], 0)
    return {"Business": 40 if row["fscore"] is None else row["fscore"],
            "Valuation": VAL_PTS.get(row["valuation"], 40),
            "Trend": st["bull"] / st["total"] * 100 if st["total"] else 0,
            "Results": 50 if _nan(q) else min(max(50 + q * 2.5, 0), 100),
            "Macro": TILT_PTS.get(row["tilt"], 50),
            "Final call": call}


def score(row: dict, style: str) -> float:
    w = WEIGHTS[style]
    return round(sum(v * w[k] for k, v in parts(row, style).items()) / sum(w.values()), 1)


def rank(rows: list[dict], style: str, n: int = 10, max_per_sector: int | None = None) -> dict:
    """{'buy': top buys (<= n), 'watch': best non-buys filling up to n, 'all': every stock, best first}."""
    scored = sorted(({**x, "score": score(x, style)} for x in rows if style in x["styles"]),
                    key=lambda x: -x["score"])
    buy, watch, count = [], [], {}
    for x in scored:
        if not is_buy(x["styles"][style]):
            continue
        sec = x["sector"] or "Other"
        if max_per_sector and count.get(sec, 0) >= max_per_sector:
            continue
        count[sec] = count.get(sec, 0) + 1
        buy.append(x)
        if len(buy) == n:
            break
    taken = {x["sym"] for x in buy}
    watch = [x for x in scored if x["sym"] not in taken and x["styles"][style]["final"] not in ("AVOID", "SELL / AVOID")]
    for i, x in enumerate(buy, 1):
        x["rank"] = i
    return {"buy": buy, "watch": watch[:max(n - len(buy), 0)], "all": scored}


def reasons(x: dict, style: str) -> str:
    st, out = x["styles"][style], []
    if x["grade"]:
        out.append(f"{x['grade_word']} business (grade {x['grade']}, {x['fscore']}/100)")
    if x["valuation"] != "Unknown":
        out.append(f"{x['valuation'].lower()} valuation" + (f" (P/E {num(x['pe'], 0)})" if not _nan(x["pe"]) else ""))
    out.append(f"{st['bull']}/{st['total']} {STYLES[style].split(' (')[0].lower()} trend checks bullish")
    if not _nan(x["q_ni_yoy"]):
        out.append(f"latest quarter profit {pct(x['q_ni_yoy'], 0)} YoY")
    if x["tilt"] != "Neutral":
        out.append(f"sector {x['tilt'].lower()}")
    return "; ".join(out)


def call_str(st: dict) -> str:
    return "START SMALL" if st["starter"] else st["final"]


def table(entries: list[dict], style: str) -> list[list]:
    out = []
    for x in entries:
        st = x["styles"][style]
        sz = f"{st['size']:g}%" + (f" in {st['parts']} parts" if st["parts"] > 1 else "") if st["size"] else DASH
        entry = st["entry"] if st["entry"] else x["price"]
        buy_at = (num(entry) if st["final"] == "BUY" or st["starter"] else f"dip to {num(entry)}"
                  if st["final"] == "BUY ON DIP" else DASH)
        out.append([x.get("rank", DASH), x["sym"], x["name"], x["sector"] or DASH, x["score"], call_str(st), sz, buy_at,
                    f"{num(st['t1'])} ({pct(st['t1_pct'], 1)})" if st["t1"] else DASH,
                    f"{num(st['stop'])} (-{st['risk_pct']:.1f}%)" if st["stop"] else DASH,
                    f"{x['grade']} ({x['fscore']})" if x["grade"] else DASH,
                    x["valuation"] + (f", P/E {num(x['pe'], 0)}" if not _nan(x["pe"]) else ""),
                    f"{st['bull']}/{st['total']} bullish", reasons(x, style)])
    return out
