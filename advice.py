"""Investment advice: one plain-English recommendation for the stock the user picked, combining the business
analysis (fundamentals), valuation, price trend, macro backdrop and portfolio fit, with a concrete action for each
holding period (swing, 3-6 months, 1-2 years) and what would change the view.

Built only from the analysis dict returned by swing.detail(); no extra downloads.
"""
from __future__ import annotations

import math

from fundamentals import cr
from strategy import BUYISH
from util import num

AREAS = {
    "Growth": {"Good": "Growing well", "OK": "Growing slowly", "Weak": "Not growing"},
    "Profitability": {"Good": "Highly profitable", "OK": "Moderately profitable", "Weak": "Low returns"},
    "Financial health": {"Good": "Strong balance sheet", "OK": "Acceptable balance sheet",
                         "Weak": "Balance-sheet / cash-flow concerns"},
    "Recent results": {"Good": "Improving", "OK": "Steady", "Weak": "Weakening"},
}
VALUATION = {"Attractive": "Attractively priced", "Fair": "Fairly priced", "Expensive": "Expensive",
             "Loss-making": "No profit to value", "Unknown": "Could not be valued"}
STANCE_KIND = {"INVEST": "buy", "BUY FOR 3-6 MONTHS": "buy", "START SMALL, ADD LATER": "wait",
               "BUY ON DIPS": "wait", "SHORT-TERM TRADE ONLY": "wait", "WAIT (WATCHLIST)": "wait", "AVOID": "avoid"}


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _rating(statuses: list[str]) -> str:
    if all(s == "Good" for s in statuses):
        return "Good"
    if "Weak" in statuses and "Good" not in statuses:
        return "Weak"
    return "OK"


def business(f: dict | None, name: str) -> tuple[str, list[list[str]]]:
    """(profile sentence, rows [Area, Assessment, Evidence])."""
    if not f:
        return (f"Financial statements for {name} are not available, so the business could not be analysed. "
                "Check sales and profit growth, debt and valuation in the annual report before investing."), []
    kind = " ".join(x for x in (f["size"].lower() if f["size"] != "–" else "",
                                 "lender" if f["lender"] else "company") if x)
    where = ", ".join(x for x in (f["industry"], f["sector"]) if x)
    profile = (f"{name} is a {kind}" + (f" in {where}" if where else "")
               + (f", market value {cr(f['mcap'])}" if not _nan(f["mcap"]) else "") + ". "
               + (f"Overall: {f['grade_word'].lower()} business (grade {f['grade']}, {f['score']}/100)."
                  if f["grade"] else "Not enough history to grade the business."))
    rows = []
    for area, words in AREAS.items():
        cs = [c for c in f["checks"] if c[0] == area and c[3] != "n/a"]
        if cs:
            rows.append([area, words[_rating([c[3] for c in cs])],
                         "; ".join(f"{c[1].split(' (')[0]} {c[2]}" for c in cs)])
    vc = [c for c in f["checks"] if c[0] == "Valuation"]
    ev = "; ".join(f"{c[1].split(' (')[0]} {c[2]}" for c in vc)
    if not _nan(f["div_yield"]) and f["div_yield"] > 0:
        ev += f"; dividend yield {f['div_yield']:.1f}%"
    rows.append(["Valuation", VALUATION.get(f["valuation"], f["valuation"]), ev or "–"])
    return profile, rows


def factors(r: dict) -> list[list[str]]:
    """[Factor, Reading, Impact] where Impact is Positive / Neutral / Negative."""
    f, m, stg, tv = r.get("fund"), r.get("macro"), r.get("strategy"), r["trend"]
    rows = []
    if f and f["grade"]:
        rows.append(["Business quality", f"{f['grade_word']} (grade {f['grade']}, {f['score']}/100)",
                     "Positive" if f["grade"] in ("A", "B") else "Negative" if f["grade"] == "D" else "Neutral"])
    else:
        rows.append(["Business quality", "not available", "Neutral"])
    if f:
        pe = f" (P/E {f['pe']:.0f})" if not _nan(f["pe"]) else ""
        rows.append(["Valuation", f["valuation"] + pe, {"Attractive": "Positive", "Fair": "Neutral"}.get(
            f["valuation"], "Neutral" if f["valuation"] == "Unknown" else "Negative")])
        q = f["q_ni_yoy"]
        if not _nan(q):
            rows.append(["Latest profit trend", f"{q:+.0f}% vs a year ago",
                         "Positive" if q >= 10 else "Negative" if q <= -10 else "Neutral"])
    low = tv["overall"].lower()
    rows.append(["Price trend", f"{tv['overall']}, {tv['strength']}",
                 "Positive" if "uptrend" in low else "Negative" if "downtrend" in low else "Neutral"])
    if m:
        rows.append(["Market (macro)", f"{m['regime']}; Nifty {m['trend'].split(' (')[0].lower()}",
                     {"Risk-on": "Positive", "Risk-off": "Negative"}.get(m["regime"], "Neutral")])
        rows.append(["Sector & macro sensitivity", m["impacts"][0] if m["impacts"] else m["tilt"],
                     {"Tailwind": "Positive", "Headwind": "Negative"}.get(m["tilt"], "Neutral")])
    if stg:
        ft = stg["fit"]
        if ft["warnings"]:
            rows.append(["Portfolio fit", ft["warnings"][0], "Negative"])
        elif ft["room"] is None:
            rows.append(["Portfolio fit", "no holdings entered (add them to check overlap)", "Neutral"])
        else:
            rows.append(["Portfolio fit", "within stock and sector limits", "Positive"])
    return rows


def _plan_for(r: dict, style: str) -> tuple[dict | None, list[str], str, str]:
    """(price plan, steps, when, technical reason) for one style."""
    if style == "swing":
        return (r["plan"] if r["side"] == "BUY" else None, r["outlook"]["steps"], r["outlook"]["when"] or "",
                r.get("why_none") or "")
    h = r["horizons"][style]
    return h["plan"], h["steps"], h["when"] or "", h["why"]


def action(r: dict, style: str, d: dict) -> str:
    p, steps, when, why = _plan_for(r, style)
    size, final = d["size_pct"], d["final"]
    if d["starter"]:
        entry = p["entry"] if p else r["last"]
        return (f"Start with {size:g}% of capital around Rs {num(entry)}; add 2 more parts when the trend turns up"
                + (f" ({steps[0]})" if steps else "") + ".")
    if final in BUYISH and size:
        s = f"Buy up to {size:g}% of capital" + (f" in {d['parts']} parts" if d["parts"] > 1 else "")
        if p:
            s += (f" around Rs {num(p['entry'])}" if final == "BUY" else f" on a dip to about Rs {num(p['entry'])}")
            s += (f"; targets Rs {num(p['t1'])} ({p['t1_pct']:+.0f}%) and Rs {num(p['t2'])} ({p['t2_pct']:+.0f}%)"
                  f"; stop-loss Rs {num(p['stop'])}" + (" (weekly close)" if style != "swing" else ""))
        return s + "."
    if final in BUYISH:
        return "Good setup, but your portfolio has no room for more of this stock / sector: do not add."
    if final == "WAIT":
        reason = d["why"][0] if d["technical"] in BUYISH else why
        return ("Do not buy yet. " + reason.rstrip(".") + "." + (f" Next: {steps[0]}" if steps else "")
                + (f" (estimate: {when})" if when else ""))
    reason = d["why"][0] if d["final"] != d["technical"] else why
    return "Avoid for now. " + reason.rstrip(".") + "."


def _stance(styles: dict, f: dict | None) -> tuple[str, str | None]:
    lt, pos, sw = styles.get("longterm"), styles.get("positional"), styles.get("swing")
    if lt and lt["size_pct"] and lt["final"] == "BUY":
        return "INVEST", "longterm"
    if lt and lt["size_pct"] and lt["starter"]:
        return "START SMALL, ADD LATER", "longterm"
    if lt and lt["size_pct"] and lt["final"] == "BUY ON DIP":
        return "BUY ON DIPS", "longterm"
    if pos and pos["size_pct"] and pos["final"] in BUYISH:
        return "BUY FOR 3-6 MONTHS", "positional"
    if sw and sw["size_pct"] and sw["final"] == "BUY":
        return "SHORT-TERM TRADE ONLY", "swing"
    weak = f is not None and (f["grade"] == "D" or f["valuation"] == "Loss-making")
    if (lt and lt["final"] == "AVOID") or weak:
        return "AVOID", None
    return "WAIT (WATCHLIST)", None


def _headline(r: dict, stance: str, best: str | None, styles: dict) -> str:
    f, m, tv = r.get("fund"), r.get("macro"), r["trend"]
    if f and f["grade"]:
        biz = f"{f['grade_word']} business (grade {f['grade']})"
        pe = f" (P/E {f['pe']:.0f})" if not _nan(f["pe"]) else ""
        biz += {"Attractive": f" at an attractive price{pe}", "Fair": f" at a fair price{pe}",
                "Expensive": f", but expensive{pe}", "Loss-making": ", currently loss-making"}.get(f["valuation"], "")
    else:
        biz = "Business could not be graded"
    ctx = f"price {tv['overall'].lower()}" + (f", {m['regime'].lower()} market" if m else "")
    what = {
        "INVEST": "Suitable for a 1-2 year investment.",
        "START SMALL, ADD LATER": "Good business to own, but the price trend has not confirmed yet: start small.",
        "BUY ON DIPS": "Good to own for 1-2 years, but the price is stretched: buy on dips.",
        "BUY FOR 3-6 MONTHS": "Suits a 3-6 month position; not yet a 1-2 year investment.",
        "SHORT-TERM TRADE ONLY": "Only a short swing trade with a strict stop-loss; not an investment now.",
        "WAIT (WATCHLIST)": "Not a buy for any holding period today: keep it on your watchlist.",
        "AVOID": "Not suitable to buy now.",
    }[stance]
    out = f"{biz}; {ctx}. {what}"
    if best:
        out += f" {styles[best]['how']}"
    return out


def watch(r: dict, stance: str, best: str | None) -> list[str]:
    """What would change the view."""
    f, m = r.get("fund"), r.get("macro")
    out = []
    if best:
        p, *_ = _plan_for(r, best)
        if p:
            out.append(f"Exit / re-think if the {'weekly ' if best != 'swing' else ''}close falls below "
                       f"Rs {num(p['stop'])} (stop-loss).")
    else:
        lt = r["horizons"].get("longterm")
        if lt and lt["steps"]:
            out += lt["steps"][:2]
        elif r["outlook"]["steps"]:
            out.append(r["outlook"]["steps"][0])
    if f:
        out.append("Re-check after each quarterly result: a profit fall of more than 25% from a year ago, rising debt "
                   "or falling margins would weaken the case.")
        if f["valuation"] == "Expensive":
            out.append("The price is high for its profits: a 10-15% fall would give a safer entry.")
    if m:
        out.append("If the market turns Risk-off (Nifty below its 200-DMA, India VIX above 22), slow down new buying."
                   if m["regime"] != "Risk-off" else
                   "The market is Risk-off now: when Nifty regains its 200-DMA and VIX cools, normal size is fine again.")
    return out


def build(r: dict, name: str = "") -> dict | None:
    stg = r.get("strategy")
    if not stg:
        return None
    styles, f = stg["styles"], r.get("fund")
    stance, best = _stance(styles, f)
    profile, biz = business(f, name or "This company")
    fac = factors(r)
    pros = [f"{k}: {v}" for k, v, i in fac if i == "Positive"] + (f["strengths"][:3] if f else [])
    cons = [f"{k}: {v}" for k, v, i in fac if i == "Negative"] + (f["flags"][:3] if f else [])
    plan = [[f"{d['label']} ({d['period']})", d["final"] + (" (start small)" if d["starter"] else ""),
             f"{d['size_pct']:g}%" if d["size_pct"] else "–", action(r, k, d)] for k, d in styles.items()]
    return {"stance": stance, "kind": STANCE_KIND[stance], "best": best,
            "headline": _headline(r, stance, best, styles), "profile": profile, "business": biz, "factors": fac,
            "pros": pros, "cons": cons, "plan": plan, "watch": watch(r, stance, best)}


def text(a: dict | None, sym: str) -> str:
    """Markdown for the chat."""
    if not a:
        return f"Not enough data to give investment advice for {sym}."
    out = f"**Investment advice for {sym}: {a['stance']}**\n\n{a['headline']}\n\n**Business analysis.** {a['profile']}\n"
    out += "\n".join(f"- {k}: **{v}** ({e})" for k, v, e in a["business"])
    out += "\n\n**Other factors:**\n" + "\n".join(f"- {k}: {v} (**{i.lower()}**)" for k, v, i in a["factors"])
    out += "\n\n**What to do:**\n" + "\n".join(f"- {s}: **{c}**. {t}" for s, c, _, t in a["plan"])
    if a["watch"]:
        out += "\n\n**What would change this view:**\n" + "\n".join(f"- {w}" for w in a["watch"])
    return out + ("\n\n_Rule-based analysis of public data (Yahoo Finance), not personalised financial advice. "
                  "Management, competition and news are not assessed._")


def rows(a: dict) -> list[list[str]]:
    """[Item, Detail] rows for the report."""
    out = [["Advice", a["headline"]], ["Business", a["profile"]]]
    out += [[k, f"{v}: {e}"] for k, v, e in a["business"]]
    out += [[f"{s}: {c}", t] for s, c, _, t in a["plan"]]
    out += [["Positive", p] for p in a["pros"]] + [["Risk", c] for c in a["cons"]]
    out += [["Would change the view", w] for w in a["watch"]]
    return out
