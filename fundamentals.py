"""Business & fundamental analysis, computed from the company's reported financial statements (Yahoo Finance).

Yahoo's ratio endpoint is often blocked, so every ratio is computed here from the annual / quarterly statements:

  * Growth: sales and profit, 3-year CAGR and last year.
  * Profitability: ROE, ROCE (or ROA for lenders), margin trend.
  * Financial health: debt / equity, interest cover, cash-flow quality (or book-value growth for lenders).
  * Recent results: latest quarter profit vs the same quarter a year earlier; share dilution.
  * Valuation: P/E on the last 12 months' profit with PEG, P/B for lenders, dividend yield.

Banks, NBFCs and insurers are judged on ROE / ROA / book value / P/B instead of debt and cash-flow ratios.
Each check is Good / OK / Weak; quality checks give a 0-100 score and a grade (A strong ... D weak).
"""
from __future__ import annotations

import math

import pandas as pd

from util import DASH, inr, num, pct

CRORE = 1e7
NAN = float("nan")
LENDER_WORDS = ("bank", "credit services", "mortgage", "insurance")
GRADES = ((75, "A", "Strong"), (60, "B", "Good"), (40, "C", "Average"), (0, "D", "Weak"))


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _row(df, *names) -> pd.Series:
    """First matching statement line as a date-sorted numeric series (empty when missing)."""
    if df is None or getattr(df, "empty", True):
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            v = df.loc[n]
            if isinstance(v, pd.DataFrame):
                v = v.iloc[0]
            s = pd.to_numeric(v, errors="coerce").dropna()
            if len(s):
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
    return pd.Series(dtype=float)


def _last(s: pd.Series) -> float:
    return float(s.iloc[-1]) if len(s) else NAN


def _cagr(s: pd.Series, years: int = 3) -> float:
    if len(s) < 2:
        return NAN
    n = min(years, len(s) - 1)
    a, b = float(s.iloc[-1 - n]), float(s.iloc[-1])
    return ((b / a) ** (1 / n) - 1) * 100 if a > 0 and b > 0 else NAN


def _yoy(s: pd.Series) -> float:
    if len(s) < 2 or s.iloc[-2] <= 0:
        return NAN
    return (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100


def _q_yoy(s: pd.Series) -> float:
    """Latest quarter vs the same quarter a year earlier (Yahoo sometimes skips quarters)."""
    for i in range(len(s) - 2, -1, -1):
        if 330 <= (s.index[-1] - s.index[i]).days <= 400:
            a = float(s.iloc[i])
            return (float(s.iloc[-1]) / a - 1) * 100 if a > 0 else NAN
    return NAN


def _ttm(q: pd.Series, annual: pd.Series) -> tuple[float, str]:
    if len(q) >= 4 and (q.index[-1] - q.index[-4]).days <= 300:
        return float(q.iloc[-4:].sum()), "last 4 quarters"
    return _last(annual), "last financial year"


def _rate(v: float, good: float, weak: float, higher: bool = True) -> str | None:
    if _nan(v):
        return None
    if higher:
        return "Good" if v >= good else "Weak" if v < weak else "OK"
    return "Good" if v <= good else "Weak" if v > weak else "OK"


def _fy(d) -> str:
    return f"FY{(d.year + (1 if d.month > 3 else 0)) % 100:02d}"


def cr(x: float) -> str:
    return DASH if _nan(x) else f"Rs {inr(x / CRORE)} cr"


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b * 100 on matching dates."""
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    j = j[j.iloc[:, 1] != 0]
    return j.iloc[:, 0] / j.iloc[:, 1] * 100


# ---------------------------------------------------------------- analysis
def analyze(raw: dict | None, price: float) -> dict | None:
    if not raw or raw.get("income") is None:
        return None
    inc, q, bs, cf = raw.get("income"), raw.get("qincome"), raw.get("balance"), raw.get("cash")
    sector, industry = raw.get("sector") or "", raw.get("industry") or ""
    rev = _row(inc, "Total Revenue", "Operating Revenue")
    ni = _row(inc, "Net Income Common Stockholders", "Net Income")
    if len(rev) < 2 or len(ni) < 2:
        return None
    ebit = _row(inc, "EBIT", "Operating Income")
    interest = _row(inc, "Interest Expense")
    eps = _row(inc, "Diluted EPS", "Basic EPS")
    eq = _row(bs, "Stockholders Equity", "Common Stock Equity")
    debt = _row(bs, "Total Debt")
    cash = _row(bs, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
    assets = _row(bs, "Total Assets")
    cur_liab = _row(bs, "Current Liabilities")
    shares_s = _row(bs, "Ordinary Shares Number", "Share Issued")
    ocf, fcf = _row(cf, "Operating Cash Flow"), _row(cf, "Free Cash Flow")
    q_rev, q_ni = _row(q, "Total Revenue", "Operating Revenue"), _row(q, "Net Income Common Stockholders", "Net Income")

    low = f"{sector} {industry}".lower()
    lender = any(w in low for w in LENDER_WORDS) or (
        not sector and bs is not None and "Current Assets" not in bs.index and "Net Interest Income" in inc.index)

    shares = raw.get("shares") or _last(shares_s)
    mcap = price * shares if not _nan(shares) and shares > 0 else NAN
    ttm_ni, ttm_src = _ttm(q_ni, ni)
    ttm_rev, _ = _ttm(q_rev, rev)
    if _nan(ttm_rev) or ttm_rev < 0.75 * _last(rev):  # quarterly 'revenue' defined differently (e.g. banks)
        ttm_rev = _last(rev)
    loss = not _nan(ttm_ni) and ttm_ni <= 0
    pe = mcap / ttm_ni if not _nan(mcap) and not _nan(ttm_ni) and ttm_ni > 0 else NAN
    e_last = _last(eq)
    pb = mcap / e_last if not _nan(mcap) and e_last > 0 else NAN
    roe_s = _ratio(ni, eq)
    roe = float(ni.iloc[-1] / eq.iloc[-2:].mean() * 100) if len(eq) >= 1 and eq.iloc[-2:].mean() > 0 else NAN
    roce = (_last(ebit) / (_last(assets) - _last(cur_liab)) * 100
            if len(ebit) and len(assets) and len(cur_liab) and _last(assets) > _last(cur_liab) else NAN)
    roa = _last(ni) / _last(assets) * 100 if len(assets) and _last(assets) > 0 else NAN
    d_last = _last(debt)
    de = d_last / e_last if not _nan(d_last) and e_last > 0 else NAN
    net_cash = _last(cash) - d_last if len(cash) and not _nan(d_last) else NAN
    int_last = _last(interest)
    icov = _last(ebit) / int_last if not _nan(int_last) and int_last > 0 and len(ebit) else NAN
    opm, npm = _ratio(ebit, rev), _ratio(ni, rev)
    rev_c, rev_y, ni_c, ni_y = _cagr(rev), _yoy(rev), _cagr(ni), _yoy(ni)
    eps_c = _cagr(eps)
    growth = eps_c if not _nan(eps_c) else ni_c
    peg = pe / growth if not _nan(pe) and not _nan(growth) and growth > 0 else NAN
    q_ni_y, q_rev_y = _q_yoy(q_ni), _q_yoy(q_rev)
    dil = NAN
    if len(shares_s) >= 2:
        n = min(3, len(shares_s) - 1)
        dil = (float(shares_s.iloc[-1]) / float(shares_s.iloc[-1 - n]) - 1) * 100
    divs = raw.get("dividends")
    div_y = NAN
    if divs is not None and len(divs) and price > 0:
        d = divs.copy()
        d.index = pd.to_datetime(d.index).tz_localize(None) if getattr(d.index, "tz", None) else pd.to_datetime(d.index)
        div_y = float(d[d.index > pd.Timestamp.today() - pd.Timedelta(days=365)].sum()) / price * 100

    checks: list[list[str]] = []

    def add(area: str, label: str, status: str | None, reading: str) -> None:
        checks.append([area, label, reading, status or "n/a"])

    # growth
    add("Growth", "Sales growth (3-year CAGR)", _rate(rev_c if not _nan(rev_c) else rev_y, 10, 3),
        f"{pct(rev_c, 1)} a year; last year {pct(rev_y, 1)}")
    if _last(ni) <= 0:
        add("Growth", "Profit growth (3-year CAGR)", "Weak", f"loss of {cr(-_last(ni))} last year")
    else:
        add("Growth", "Profit growth (3-year CAGR)", _rate(ni_c if not _nan(ni_c) else ni_y, 10, 0),
            f"{pct(ni_c, 1)} a year; last year {pct(ni_y, 1)}")
    # profitability
    add("Profitability", "Return on equity (ROE)", _rate(roe, 14 if lender else 15, 10),
        f"{num(roe, 1)}% (profit / shareholders' money)")
    if lender:
        add("Profitability", "Return on assets (ROA)", _rate(roa, 1.5, 0.8), f"{num(roa, 2)}% (profit / total assets)")
        m = npm
    else:
        add("Profitability", "Return on capital employed (ROCE)", _rate(roce, 15, 10),
            f"{num(roce, 1)}% (operating profit / capital used)")
        m = opm
    if len(m) >= 2:
        now, then = float(m.iloc[-1]), float(m.iloc[max(-4, -len(m))])
        floor_good, floor_weak = (15, 8) if lender else (10, 5)
        st = ("Good" if now >= then - 1 and now >= floor_good else
              "Weak" if now < then - 3 or now < floor_weak else "OK")
        add("Profitability", f"{'Net' if lender else 'Operating'} margin trend", st,
            f"{now:.1f}% now vs {then:.1f}% in {_fy(m.index[max(-4, -len(m))])}")
    # financial health
    if lender:
        add("Financial health", "Book value growth (3-year CAGR)", _rate(_cagr(eq), 12, 5),
            f"{pct(_cagr(eq), 1)} a year (shareholders' equity)")
    else:
        st = (None if _nan(de) else "Good" if de <= 0.5 or (not _nan(net_cash) and net_cash > 0)
              else "Weak" if de > 1.5 else "OK")
        cash_txt = ("" if _nan(net_cash) else f", net cash {cr(net_cash)}" if net_cash > 0 else f", net debt {cr(-net_cash)}")
        add("Financial health", "Debt to equity", st, f"{num(de)}{cash_txt}")
        if _nan(icov) and not _nan(de) and de <= 0.1:
            add("Financial health", "Interest cover (operating profit / interest)", "Good", "negligible interest cost")
        else:
            add("Financial health", "Interest cover (operating profit / interest)", _rate(icov, 5, 2),
                f"{num(icov, 1)}x" if not _nan(icov) else DASH)
        j = pd.concat([ocf, ni], axis=1, join="inner").dropna().iloc[-3:]
        f3 = fcf.iloc[-3:]
        if len(j) and j.iloc[:, 1].sum() > 0:
            conv = float(j.iloc[:, 0].sum() / j.iloc[:, 1].sum())
            pos = int((f3 > 0).sum())
            st = ("Good" if conv >= 0.8 and pos >= min(2, len(f3)) else
                  "Weak" if conv < 0.5 or (len(f3) >= 2 and pos == 0) else "OK")
            add("Financial health", "Cash-flow quality", st,
                f"operating cash flow = {conv:.0%} of profit (3 yrs); free cash flow positive {pos} of {len(f3)} yrs")
        else:
            add("Financial health", "Cash-flow quality", None, DASH)
    # recent results
    if not _nan(q_ni_y):
        add("Recent results", "Latest quarter profit vs a year ago", _rate(q_ni_y, 10, -10),
            f"profit {pct(q_ni_y, 1)}, sales {pct(q_rev_y, 1)} ({q_ni.index[-1]:%b %Y} quarter)")
    elif ttm_src == "last 4 quarters" and q_ni.index[-1] > ni.index[-1] and _last(ni) > 0:
        q_ni_y = (ttm_ni / _last(ni) - 1) * 100
        add("Recent results", "Last 12 months profit vs last financial year", _rate(q_ni_y, 8, -10),
            f"{pct(q_ni_y, 1)} (12 months to {q_ni.index[-1]:%b %Y} vs {_fy(ni.index[-1])})")
    else:
        add("Recent results", "Latest quarter profit vs a year ago", None, "quarterly data unavailable")
    add("Recent results", "Share dilution (3 years)", _rate(dil, 3, 15, higher=False),
        f"share count {pct(dil, 1)}" if not _nan(dil) else DASH)
    # valuation
    if loss:
        add("Valuation", "P/E (price / last 12 months' profit)", "Weak", "loss-making: no earnings to value")
    else:
        st = (None if _nan(pe) else
              "Good" if pe <= 20 or (not _nan(peg) and peg <= 1.5 and pe <= 50) else
              "Weak" if pe > 60 or (not _nan(peg) and peg > 3) or ((_nan(growth) or growth <= 0) and pe > 30) else "OK")
        add("Valuation", "P/E (price / last 12 months' profit)", st,
            f"{num(pe, 1)} on {ttm_src} profit" + (f"; PEG {num(peg, 2)} (P/E / EPS growth)" if not _nan(peg) else ""))
    if lender:
        st = (None if _nan(pb) else "Good" if pb <= 2 or (pb <= 3.5 and roe >= 15) else
              "Weak" if pb > 5 or (pb > 3 and roe < 12) else "OK")
        add("Valuation", "P/B (price / book value)", st, f"{num(pb, 2)}")

    quality = [c for c in checks if c[0] != "Valuation" and c[3] != "n/a"]
    score = grade = word = None
    if len(quality) >= 4:
        score = round(100 * sum({"Good": 2, "OK": 1}.get(c[3], 0) for c in quality) / (2 * len(quality)))
        _, grade, word = next(g for g in GRADES if score >= g[0])
    vals = [c[3] for c in checks if c[0] == "Valuation" and c[3] != "n/a"]
    valuation = ("Loss-making" if loss else "Unknown" if not vals else "Expensive" if "Weak" in vals
                 else "Attractive" if all(v == "Good" for v in vals) else "Fair")

    flags = []
    if loss or _last(ni) <= 0:
        flags.append("Made a loss in the latest period")
    if not _nan(rev_y) and rev_y < -5:
        flags.append(f"Sales fell {abs(rev_y):.0f}% last year")
    if not lender and not _nan(de) and de > 2:
        flags.append(f"High debt (debt / equity {de:.1f})")
    if not lender and len(fcf.iloc[-3:]) >= 2 and (fcf.iloc[-3:] <= 0).all():
        flags.append("Negative free cash flow in each recent year")
    if not _nan(q_ni_y) and q_ni_y < -25:
        flags.append(f"Recent profit down {abs(q_ni_y):.0f}% from a year ago")
    if not _nan(dil) and dil > 15:
        flags.append(f"Share count up {dil:.0f}% in 3 years (dilution)")
    if not _nan(icov) and icov < 1.5:
        flags.append(f"Operating profit barely covers interest ({icov:.1f}x)")
    if not _nan(pe) and pe > 80:
        flags.append(f"Very high valuation (P/E {pe:.0f})")
    strengths = [f"{c[1].split(' (')[0]}: {c[2].split(';')[0]}" for c in checks if c[3] == "Good"]

    size = (DASH if _nan(mcap) else "Large cap" if mcap >= 1e5 * CRORE else "Mid cap" if mcap >= 3e4 * CRORE
            else "Small cap")
    metrics = [
        ["Market cap", f"{cr(mcap)} ({size})"],
        ["P/E", "loss-making" if loss else num(pe, 1)], ["PEG", num(peg, 2)], ["P/B", num(pb, 2)],
        ["Dividend yield", f"{num(div_y, 2)}%" if not _nan(div_y) else DASH],
        ["ROE", f"{num(roe, 1)}%"],
        ["ROA" if lender else "ROCE", f"{num(roa if lender else roce, 1 + lender)}%"],
        ["Sales (last 12 months)", cr(ttm_rev)], ["Net profit (last 12 months)", cr(ttm_ni)],
    ]
    if not lender:
        metrics.insert(7, ["Debt / equity", num(de)])
    if not _nan(shares) and not _nan(ttm_ni) and shares > 0:
        metrics.append(["EPS (last 12 months)", f"Rs {num(ttm_ni / shares)}"])
    history = []
    for d in rev.index[-5:]:
        r_, n_ = float(rev[d]), float(ni.get(d, NAN))
        history.append([_fy(d), inr(r_ / CRORE), inr(n_ / CRORE), f"{n_ / r_ * 100:.1f}%" if r_ else DASH,
                        num(float(eps.get(d, NAN))), f"{num(float(roe_s.get(d, NAN)), 1)}%"])

    f = {
        "name": raw.get("name") or "", "sector": sector, "industry": industry, "lender": lender, "mcap": mcap,
        "size": size, "score": score, "grade": grade, "grade_word": word, "valuation": valuation, "pe": pe,
        "pb": pb, "peg": peg, "roe": roe, "roce": roce, "roa": roa, "de": de, "rev_cagr": rev_c, "ni_cagr": ni_c,
        "q_ni_yoy": q_ni_y, "div_yield": div_y, "loss": loss, "checks": checks, "metrics": metrics,
        "history": history, "flags": flags, "strengths": strengths,
    }
    f["summary"] = summary(f)
    return f


def summary(f: dict) -> str:
    head = (f"{f['grade_word']} business (grade {f['grade']}, {f['score']}/100)" if f["grade"]
            else "Not enough data to grade the business")
    val = f["valuation"].lower() + (f" (P/E {f['pe']:.0f})" if not _nan(f["pe"]) else "")
    out = f"{head}; valuation {val}."
    if f["strengths"]:
        out += " Strengths: " + "; ".join(f["strengths"][:3]) + "."
    if f["flags"]:
        out += " Watch: " + "; ".join(f["flags"][:3]) + "."
    return out


def text(f: dict | None, sym: str) -> str:
    """Markdown explanation for the chat."""
    if not f:
        return (f"Fundamental data for {sym} is not available right now (Yahoo did not return the financial "
                "statements). Check the company's results, debt and valuation on screener.in or the annual report.")
    where = ", ".join(x for x in (f["sector"], f["industry"], f["size"]) if x and x != DASH)
    out = f"**Business & fundamentals for {sym}**" + (f" ({where})" if where else "") + f"\n\n{f['summary']}\n\n"
    out += "\n".join(f"- {c[1]}: {c[2]} (**{c[3]}**)" for c in f["checks"])
    if f["lender"]:
        out += "\n\nThis is a lender, so it is judged on ROE, ROA, book-value growth and P/B instead of debt ratios."
    return out
