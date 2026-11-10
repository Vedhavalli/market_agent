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
LENDER_WORDS = ("bank", "credit services", "mortgage", "insurance", "financial services", "capital markets",
                "asset management")
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


def _sector_assumptions(sector: str, industry: str, lender: bool) -> dict:
    text = f"{sector} {industry}".lower()
    if lender:
        return {"group": "Financial", "fair_pe": 14.0, "fair_pb_cap": 3.0, "wacc": None,
                "growth_cap": 8.0, "roe_good": 14.0, "roce_good": None, "debt_good": None}
    if any(word in text for word in ("utility", "utilities", "electric", "power")):
        return {"group": "Utilities", "fair_pe": 13.0, "fair_pb_cap": None, "wacc": 0.13,
                "growth_cap": 6.0, "roe_good": 12.0, "roce_good": 12.0, "debt_good": 1.5}
    if any(word in text for word in ("technology", "software", "semiconductor", "information technology")):
        return {"group": "Technology", "fair_pe": 20.0, "fair_pb_cap": None, "wacc": 0.11,
                "growth_cap": 12.0, "roe_good": 15.0, "roce_good": 15.0, "debt_good": 0.75}
    if any(word in text for word in ("health", "pharma", "biotech", "medical")):
        return {"group": "Healthcare", "fair_pe": 18.0, "fair_pb_cap": None, "wacc": 0.12,
                "growth_cap": 10.0, "roe_good": 14.0, "roce_good": 14.0, "debt_good": 0.75}
    if any(word in text for word in ("consumer defensive", "staple", "household", "food")):
        return {"group": "Consumer defensive", "fair_pe": 18.0, "fair_pb_cap": None, "wacc": 0.11,
                "growth_cap": 8.0, "roe_good": 15.0, "roce_good": 15.0, "debt_good": 0.75}
    if any(word in text for word in ("consumer cyclical", "consumer discretionary", "retail", "automotive")):
        return {"group": "Consumer cyclical", "fair_pe": 16.0, "fair_pb_cap": None, "wacc": 0.12,
                "growth_cap": 8.0, "roe_good": 14.0, "roce_good": 14.0, "debt_good": 1.0}
    if any(word in text for word in ("real estate", "reit")):
        return {"group": "Real estate", "fair_pe": 14.0, "fair_pb_cap": None, "wacc": 0.13,
                "growth_cap": 6.0, "roe_good": 12.0, "roce_good": 12.0, "debt_good": 1.5}
    if "communication services" in text:
        return {"group": "Communication services", "fair_pe": 15.0, "fair_pb_cap": None, "wacc": 0.12,
                "growth_cap": 7.0, "roe_good": 12.0, "roce_good": 12.0, "debt_good": 1.25}
    if any(word in text for word in ("basic materials", "metal", "oil", "gas", "energy", "industrial", "mining")):
        return {"group": "Cyclical / capital intensive", "fair_pe": 14.0, "fair_pb_cap": None, "wacc": 0.14,
                "growth_cap": 5.0, "roe_good": 12.0, "roce_good": 14.0, "debt_good": 1.0}
    return {"group": "Diversified", "fair_pe": 16.0, "fair_pb_cap": None, "wacc": 0.12,
            "growth_cap": 8.0, "roe_good": 15.0, "roce_good": 15.0, "debt_good": 0.75}


def _intrinsic_values(price: float, eps: float, bvps: float, shares: float, fcf: float, net_cash: float,
                      growth: float, fcf_growth: float, roe: float, assumptions: dict, lender: bool) -> dict:
    """Conservative per-share estimates; unavailable methods stay None rather than implying precision."""
    fair_pe = assumptions["fair_pe"] + min(max(growth, 0.0), assumptions["growth_cap"]) * 0.5 if not _nan(growth) else assumptions["fair_pe"]
    methods = {"Graham number": math.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else None,
               "Earnings power": eps * fair_pe if eps > 0 else None,
               "Book value": None,
               "Discounted cash flow": None}
    fair_pb = None
    if lender and bvps > 0:
        fair_pb = min(assumptions["fair_pb_cap"], max(0.5, roe / 12)) if not _nan(roe) else 1.0
        methods["Book value"] = bvps * fair_pb
    elif not lender and fcf > 0 and shares > 0 and assumptions["wacc"]:
        cash_growth = fcf_growth if not _nan(fcf_growth) else growth
        g = min(max(cash_growth, 0.0), assumptions["growth_cap"]) / 100 if not _nan(cash_growth) else 0.03
        wacc, terminal = assumptions["wacc"], 0.03
        if wacc > terminal:
            projected = [fcf * (1 + g) ** year for year in range(1, 6)]
            present = sum(value / (1 + wacc) ** year for year, value in enumerate(projected, 1))
            terminal_value = projected[-1] * (1 + terminal) / (wacc - terminal)
            equity_value = present + terminal_value / (1 + wacc) ** 5 + (net_cash if not _nan(net_cash) else 0.0)
            methods["Discounted cash flow"] = max(0.0, equity_value / shares)
    estimates = [v for v in methods.values() if v is not None and math.isfinite(v) and v > 0]
    intrinsic = float(pd.Series(estimates).median()) if len(estimates) >= 2 else NAN
    upside = (intrinsic / price - 1) * 100 if price > 0 and not _nan(intrinsic) else NAN
    mos = (1 - price / intrinsic) * 100 if price > 0 and not _nan(intrinsic) else NAN
    rating = ("Unknown" if _nan(mos) else "Undervalued" if mos >= 25 else
              "Fairly valued" if mos >= -20 else "Overvalued")
    return {"methods": methods, "intrinsic_value": intrinsic, "upside_pct": upside, "margin_of_safety_pct": mos,
            "value_rating": rating, "fair_pe": fair_pe, "fair_pb": fair_pb, "model_count": len(estimates),
            "assumptions": {"sector_group": assumptions["group"], "DCF WACC": assumptions["wacc"],
                            "terminal growth": 3.0 if assumptions["wacc"] else None,
                            "growth cap": assumptions["growth_cap"]}}


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
    today = pd.Timestamp.today().normalize()

    def statement_date(statement):
        if statement is None or getattr(statement, "empty", True):
            return None
        dates = pd.to_datetime(statement.columns, errors="coerce").dropna()
        return pd.Timestamp(dates.max()).normalize() if len(dates) else None

    quarter_date, annual_date = statement_date(raw.get("qincome")), statement_date(inc)
    as_of = max((d for d in (quarter_date, annual_date) if d is not None), default=None)
    age_days = max(0, (today - as_of).days) if as_of is not None else None
    balance_date, cash_date = statement_date(raw.get("balance")), statement_date(raw.get("cash"))
    balance_age = max(0, (today - balance_date).days) if balance_date is not None else None
    cash_age = max(0, (today - cash_date).days) if cash_date is not None else None
    income_limit = 210 if quarter_date is not None else 550
    if as_of is None:
        data_status = "Unknown"
    elif age_days > income_limit or (balance_age is not None and balance_age > 550) or (cash_age is not None and cash_age > 550):
        data_status = "Stale"
    elif balance_date is None:
        data_status = "Incomplete"
    else:
        data_status = "Current"
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
    assumptions = _sector_assumptions(sector, industry, lender)

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
    eps_ttm = ttm_ni / shares if not _nan(shares) and shares > 0 and not _nan(ttm_ni) else NAN
    bvps = e_last / shares if not _nan(shares) and shares > 0 and e_last > 0 else NAN
    fcf_latest = _last(fcf)
    fcf_net = NAN
    fcf_join = pd.concat([fcf, ni], axis=1, join="inner").dropna().iloc[-3:]
    if len(fcf_join) and fcf_join.iloc[:, 1].sum() > 0:
        fcf_net = float(fcf_join.iloc[:, 0].sum() / fcf_join.iloc[:, 1].sum())
    fcf_growth = _cagr(fcf)
    value = _intrinsic_values(price, eps_ttm, bvps, shares, fcf_latest, net_cash, growth, fcf_growth, roe,
                              assumptions, lender)
    analyst = raw.get("analyst") if isinstance(raw.get("analyst"), dict) else None
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
    add("Profitability", "Return on equity (ROE)", _rate(roe, assumptions["roe_good"], 8 if lender else 8),
        f"{num(roe, 1)}% (profit / shareholders' money)")
    if lender:
        add("Profitability", "Return on assets (ROA)", _rate(roa, 1.5, 0.8), f"{num(roa, 2)}% (profit / total assets)")
        m = npm
    else:
        add("Profitability", "Return on capital employed (ROCE)", _rate(roce, assumptions["roce_good"], 8),
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
        st = (None if _nan(de) else "Good" if de <= assumptions["debt_good"] or
              (not _nan(net_cash) and net_cash > 0) else "Weak" if de > 2.0 else "OK")
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
        add("Financial health", "Free cash flow / net income (3 years)",
            _rate(fcf_net, 0.8, 0.5), f"{num(fcf_net * 100, 0)}%" if not _nan(fcf_net) else DASH)
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
    mos = value["margin_of_safety_pct"]
    add("Valuation", "Estimated intrinsic value (median of available methods)",
        None if _nan(mos) else "Good" if mos >= 25 else "Weak" if mos < -20 else "OK",
        f"Rs {num(value['intrinsic_value'])}; margin of safety {num(mos, 1)}% ({value['model_count']}/3 models)"
        if not _nan(mos) else DASH)

    quality = [c for c in checks if c[0] != "Valuation" and c[3] != "n/a"]
    score = grade = word = None
    if len(quality) >= 4:
        score = round(100 * sum({"Good": 2, "OK": 1}.get(c[3], 0) for c in quality) / (2 * len(quality)))
        _, grade, word = next(g for g in GRADES if score >= g[0])
    vals = [c[3] for c in checks if c[0] == "Valuation" and c[3] != "n/a"]
    valuation = ("Loss-making" if loss else
                 {"Undervalued": "Attractive", "Fairly valued": "Fair", "Overvalued": "Expensive"}.get(
                     value["value_rating"], "Unknown" if not vals else "Expensive" if "Weak" in vals
                     else "Attractive" if all(v == "Good" for v in vals) else "Fair"))

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
    if not lender and not _nan(fcf_latest) and fcf_latest < 0:
        flags.append("Latest reported free cash flow is negative")
    if not _nan(fcf_net) and fcf_net < 0.5:
        flags.append(f"Weak free-cash-flow conversion ({fcf_net:.0%} of net income over 3 years)")
    if not _nan(mos) and mos < -20:
        flags.append(f"Price is {abs(mos):.0f}% above the conservative intrinsic-value estimate")
    if data_status == "Stale":
        flags.append(f"Latest Yahoo financial statement is {age_days} days old")
    elif data_status == "Unknown":
        flags.append("Financial statement reporting date could not be verified")
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
        metrics.insert(8, ["FCF / net income (3 years)", f"{num(fcf_net * 100, 0)}%" if not _nan(fcf_net) else DASH])
        metrics.insert(9, ["Latest free cash flow", cr(fcf_latest)])
    metrics.extend([["Graham number", f"Rs {num(value['methods']['Graham number'])}" if value["methods"]["Graham number"] else DASH],
                    ["Earnings-power value", f"Rs {num(value['methods']['Earnings power'])}" if value["methods"]["Earnings power"] else DASH],
                    ["Book-value estimate", f"Rs {num(value['methods']['Book value'])}" if value["methods"]["Book value"] else DASH],
                    ["DCF estimate", f"Rs {num(value['methods']['Discounted cash flow'])}" if value["methods"]["Discounted cash flow"] else DASH],
                    ["Median intrinsic value", f"Rs {num(value['intrinsic_value'])}" if not _nan(value["intrinsic_value"]) else DASH],
                    ["Margin of safety", f"{num(mos, 1)}%" if not _nan(mos) else DASH],
                    ["Value model coverage", f"{value['model_count']}/3 (need at least 2)"],
                    ["Statement data", f"Yahoo Finance only; {data_status.lower()}"
                     + (f" as of {as_of:%d %b %Y}" if as_of is not None else "; date unavailable")]])
    if analyst:
        target_upside = ((analyst["target_mean"] / price - 1) * 100
                         if analyst.get("target_mean") and price > 0 else None)
        metrics.append(["Analyst consensus (Yahoo)", f"{analyst['consensus']} ({analyst['total']} ratings)"])
        metrics.append(["Analyst ratings", f"{analyst['strongBuy']} strong buy / {analyst['buy']} buy / "
                        f"{analyst['hold']} hold / {analyst['sell']} sell / {analyst['strongSell']} strong sell"])
        metrics.append(["Analyst mean target", f"Rs {num(analyst['target_mean'])}"
                        + (f" ({target_upside:+.1f}% vs price)" if target_upside is not None else "")
                        if analyst.get("target_mean") else DASH])
    if not _nan(shares) and not _nan(ttm_ni) and shares > 0:
        metrics.append(["EPS (last 12 months)", f"Rs {num(ttm_ni / shares)}"])
    history = []
    for d in rev.index[-5:]:
        r_, n_ = float(rev[d]), float(ni.get(d, NAN))
        history.append([_fy(d), inr(r_ / CRORE), inr(n_ / CRORE), f"{n_ / r_ * 100:.1f}%" if r_ else DASH,
                        num(float(eps.get(d, NAN))), f"{num(float(roe_s.get(d, NAN)), 1)}%"])

    f = {
        "name": raw.get("name") or "", "sector": sector, "industry": industry, "lender": lender,
        "price": price, "mcap": mcap,
        "size": size, "score": score, "grade": grade, "grade_word": word, "valuation": valuation, "pe": pe,
        "pb": pb, "peg": peg, "roe": roe, "roce": roce, "roa": roa, "de": de, "rev_cagr": rev_c, "ni_cagr": ni_c,
        "q_ni_yoy": q_ni_y, "div_yield": div_y, "loss": loss, "fcf_net": fcf_net, "fcf_latest": fcf_latest,
        "eps_ttm": eps_ttm, "bvps": bvps, "value_methods": value["methods"],
        "intrinsic_value": value["intrinsic_value"], "upside_pct": value["upside_pct"],
        "margin_of_safety_pct": mos, "value_rating": value["value_rating"], "fair_pe": value["fair_pe"],
        "fair_pb": value["fair_pb"], "value_growth_pct": fcf_growth if not _nan(fcf_growth) else growth,
        "value_assumptions": value["assumptions"], "balance_as_of": balance_date, "cash_as_of": cash_date,
        "value_model_count": value["model_count"], "analyst": analyst, "statement_as_of": as_of,
        "statement_age_days": age_days, "data_status": data_status,
        "checks": checks, "metrics": metrics,
        "history": history, "flags": flags, "strengths": strengths,
    }
    f["summary"] = summary(f)
    return f


def summary(f: dict) -> str:
    head = (f"{f['grade_word']} business (grade {f['grade']}, {f['score']}/100)" if f["grade"]
            else "Not enough data to grade the business")
    val = f["valuation"].lower() + (f" (P/E {f['pe']:.0f})" if not _nan(f["pe"]) else "")
    out = f"{head}; valuation {val}."
    if not _nan(f.get("intrinsic_value")):
        out += (f" Median estimated intrinsic value Rs {num(f['intrinsic_value'])}, "
                f"margin of safety {num(f['margin_of_safety_pct'], 1)}% using sector-adjusted methods.")
    out += (f" Statement data: {f.get('data_status', 'Unknown').lower()}, from Yahoo Finance only; "
            "not independently reconciled against exchange filings.")
    analyst = f.get("analyst")
    if analyst:
        out += f" Analyst snapshot: {analyst['consensus'].lower()} ({analyst['total']} ratings; Yahoo)."
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
    if f.get("analyst"):
        a = f["analyst"]
        out += (f"\n\n**Analyst cross-check (Yahoo, {a['period']}):** {a['consensus']} from {a['total']} ratings "
                f"({a['strongBuy']} strong buy, {a['buy']} buy, {a['hold']} hold, {a['sell']} sell, "
                f"{a['strongSell']} strong sell). Analyst targets are consensus estimates, not intrinsic value.")
    out += ("\n\nFundamental statements are from one aggregator (Yahoo Finance via yfinance), "
            "not independently verified. Confirm latest figures against NSE/BSE company filings.")
    if f["lender"]:
        out += "\n\nThis is a lender, so it is judged on ROE, ROA, book-value growth and P/B instead of debt ratios."
    return out
