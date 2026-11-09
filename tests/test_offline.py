"""Offline tests: synthetic prices stand in for Yahoo Finance. Run: python tests/test_offline.py"""
import hashlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import data  # noqa: E402
import portfolio as pf  # noqa: E402
import report as rpt  # noqa: E402
from indicators import analyze, rsi  # noqa: E402
from util import inr  # noqa: E402


def fake_history(symbol, period="2y"):
    seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    n = 500
    drift = rng.uniform(-0.0006, 0.0012)
    close = 1000 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": 1e6}, index=idx)


def stmt(rows: dict, dates) -> pd.DataFrame:
    """Yahoo-style statement: line items as rows, newest period first."""
    return pd.DataFrame(rows, index=pd.to_datetime(dates)).T.iloc[:, ::-1]


YEARS = ["2023-03-31", "2024-03-31", "2025-03-31", "2026-03-31"]
QTRS = ["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]
CR = 1e7


def company(rev, ni, eq, debt, sector="Technology", industry="Information Technology Services", q_ni=None):
    rev, ni, eq, debt = (np.array(x, dtype=float) * CR for x in (rev, ni, eq, debt))
    inc = stmt({"Total Revenue": rev, "Net Income": ni, "EBIT": ni * 1.4 + debt * 0.1,
                "Interest Expense": debt * 0.09, "Diluted EPS": ni / 1e8}, YEARS)
    bs = stmt({"Stockholders Equity": eq, "Total Debt": debt, "Cash And Cash Equivalents": eq * 0.2,
               "Total Assets": (eq + debt) * 1.4, "Current Liabilities": (eq + debt) * 0.3,
               "Current Assets": eq * 0.5, "Ordinary Shares Number": [1e8] * 4}, YEARS)
    cf = stmt({"Operating Cash Flow": ni * 1.1, "Free Cash Flow": ni * 0.8}, YEARS)
    qn = np.array(q_ni if q_ni is not None else [ni[-1] / CR / 4 * k for k in (0.9, 0.95, 1.0, 1.05, 1.1)]) * CR
    q = stmt({"Total Revenue": [rev[-1] / 4] * 5, "Net Income": qn}, QTRS)
    return {"income": inc, "qincome": q, "balance": bs, "cash": cf, "dividends": None, "shares": 1e8,
            "name": "Test Co", "sector": sector, "industry": industry}


def main():
    data.history = fake_history  # patch network layer

    # util
    assert inr(1234567) == "12,34,567" and inr(-98765.4, 1) == "-98,765.4" and inr(999) == "999"

    # indicators: pure uptrend -> RSI 100, bullish; pure downtrend -> bearish
    up = pd.DataFrame({"Close": np.linspace(100, 200, 300)})
    up["High"], up["Low"] = up["Close"] * 1.005, up["Close"] * 0.995
    up.index = pd.bdate_range(end="2026-11-09", periods=300)
    a = analyze(up)
    assert a["signal"] == "BUY-WATCH" and a["rsi"] == 100.0, a
    dn = up.copy()
    dn["Close"] = np.linspace(200, 100, 300)
    dn["High"], dn["Low"] = dn["Close"] * 1.005, dn["Close"] * 0.995
    assert analyze(dn)["signal"] == "SELL-WATCH"
    assert analyze(up.iloc[:10]) is None
    assert 0 <= rsi(fake_history("X")["Close"]).iloc[-1] <= 100

    # portfolio operations
    tmp = Path(tempfile.mkdtemp())
    p, realized = tmp / "portfolio.csv", tmp / "realized.csv"
    df = pf.load(p)
    df = pf.add(df, "sbin.ns", 10, 800, stop=700, target=1500)
    df = pf.add(df, "SBIN", 10, 900)  # average -> 850
    assert float(df.loc[0, "avg_price"]) == 850 and float(df.loc[0, "qty"]) == 20
    assert float(df.loc[0, "stop_loss"]) == 700  # kept
    df = pf.add(df, "TCS", 5, 3000, stop=2999999)  # absurd stop -> must trigger STOP-LOSS alert
    df, pnl = pf.sell(df, "SBIN", 5, 950, realized)
    assert pnl == 500.0 and float(df.loc[0, "qty"]) == 15
    try:
        pf.sell(df, "SBIN", 99, 1, realized)
        raise SystemExit("oversell should fail")
    except ValueError:
        pass
    pf.save(df, p)
    assert len(pf.load(p)) == 2

    # full report on top of the CSV
    cfg = agent.load_config(None)
    rep = rpt.build(cfg, pf.load(p))
    titles = [s.title for s in rep.sections]
    assert any("Market snapshot" in t for t in titles) and any("Portfolio alerts" in t for t in titles)
    alerts = next(s for s in rep.sections if "alerts" in s.title)
    assert any("STOP-LOSS" in r[1] for r in alerts.rows), alerts.rows
    from rich.console import Console

    buf = io.StringIO()
    rpt.render_console(rep, Console(file=buf, width=140))
    assert "Market snapshot" in buf.getvalue()
    page = rpt.render_html(rep, "commentary <b>x</b>")
    assert "&lt;b&gt;" in page and "<table>" in page  # escaped
    assert "Nifty 50" in rpt.to_text(rep)

    # swing engine: trade-plan geometry
    import swing

    s = swing.settings(cfg)
    bp = swing.plan("BUY", 100.0, 2.0, 97.0, s)
    assert bp["stop"] < 100 < bp["t1"] < bp["t2"]
    assert abs((bp["t1"] - 100) - s["t1_rr"] * (100 - bp["stop"])) < 1e-9
    sp = swing.plan("SELL", 100.0, 2.0, 103.0, s)
    assert sp["t2"] < sp["t1"] < 100 < sp["stop"]
    assert swing.plan("BUY", 100.0, 2.0, 50.0, s)["stop"] == 100 - s["atr_stop_max"] * 2  # far swing low -> capped
    assert swing.plan("BUY", 100.0, 2.0, 99.9, s)["stop"] == 100 - s["atr_stop_min"] * 2  # near swing low -> min gap

    # uptrend -> pullback -> breakout must produce a BUY with a full plan; mirror image must produce a SELL
    def shaped(path):
        close = np.array(path, dtype=float)
        idx = pd.bdate_range(end="2026-11-06", periods=len(close))
        op = np.r_[close[0], close[:-1]]
        vol = np.where(np.diff(np.r_[close[0], close]) >= 0, 2e6, 1e6)
        return pd.DataFrame({"Open": op, "High": np.maximum(op, close) * 1.004,
                             "Low": np.minimum(op, close) * 0.996, "Close": close, "Volume": vol}, index=idx)

    t = np.arange(400)
    trend = 100 * 1.003 ** t * (1 + 0.01 * np.sin(t / 2))
    path = list(trend) + list(trend[-1] * 0.993 ** np.arange(1, 9))
    path += list(path[-1] * 1.012 ** np.arange(1, 7))
    rb = swing.analyze(shaped(path), s, "Bullish")
    assert rb["side"] == "BUY", (rb["bull"], rb["conditions"], rb["trig_buy"])
    assert rb["plan"]["stop"] < rb["last"] < rb["plan"]["t1"] and rb["plan"]["t1_eta"]["fast"] >= 1
    rs = swing.analyze(shaped([20000 / v for v in path]), s, "Bearish")
    assert rs["side"] == "SELL", (rs["bear"], rs["conditions"], rs["trig_sell"])
    assert swing.analyze(shaped(path[:100]), s) is None

    # no look-ahead: signals for past bars don't change when future bars are appended
    full = swing.compute(shaped(path))
    part = swing.compute(shaped(path).iloc[:-5])
    b1 = swing.signals(full, s)[2].iloc[:-5]
    b2 = swing.signals(part, s)[2]
    assert (b1.values == b2.values).all()

    srep = swing.scan(cfg, cfg["watchlist"]["symbols"] + ["NODATA"])
    stitles = [x.title for x in srep.sections]
    assert any("BUY signals" in x for x in stitles) and any("SELL signals" in x for x in stitles)
    rpt.render_html(srep)

    # business fundamentals: strong / weak / lender companies from synthetic statements
    import advice
    import fundamentals
    import macro
    import strategy

    strong = company([1000, 1150, 1320, 1520], [150, 175, 205, 240], [800, 900, 1020, 1150], [50, 50, 50, 50])
    weak = company([1000, 900, 820, 700], [40, 10, -30, -60], [500, 470, 440, 380], [900, 1000, 1100, 1200],
                   q_ni=[-20, -15, -15, -10, -25])
    bank = company([1000, 1150, 1320, 1520], [150, 175, 205, 240], [800, 920, 1060, 1220], [5000] * 4,
                   sector="Financial Services", industry="Banks - Regional")
    fs = fundamentals.analyze(strong, 400.0)
    assert fs["grade"] in ("A", "B") and fs["valuation"] in ("Attractive", "Fair"), (fs["score"], fs["checks"])
    assert 14 < fs["pe"] < 18 and not fs["lender"] and fs["history"], fs["pe"]
    fw = fundamentals.analyze(weak, 100.0)
    assert fw["grade"] == "D" and fw["valuation"] == "Loss-making" and fw["flags"], (fw["score"], fw["checks"])
    fb = fundamentals.analyze(bank, 400.0)
    assert fb["lender"] and any("Return on assets" in c[1] for c in fb["checks"])
    assert not any("Debt to equity" in c[1] for c in fb["checks"])
    assert fundamentals.analyze(None, 100.0) is None and "not available" in fundamentals.text(None, "X")
    assert "Business & fundamentals" in fundamentals.text(fs, "X")

    # macro: bull market + calm VIX -> risk-on; bear market + high VIX -> risk-off
    t2 = np.arange(900)
    up_long = 100 * 1.0012 ** t2 * (1 + 0.03 * np.sin(t2 / 15))
    calm, fear = shaped(np.full(300, 12.0)), shaped(np.full(300, 25.0))
    m_on = macro.analyze(shaped(up_long), {"vix": calm})
    m_off = macro.analyze(shaped(20000 / up_long), {"vix": fear}, ("Nifty IT", shaped(20000 / up_long)),
                          "Technology", "Software")
    assert m_on["regime"] == "Risk-on" and m_on["size_factor"] == 1.0, m_on["rows"]
    assert m_off["regime"] == "Risk-off" and m_off["size_factor"] == 0.5, m_off["rows"]
    assert macro.analyze(None, {}) is None and macro.sector_index("Financial Services", "Banks - Regional")[0] == "Bank Nifty"
    assert "Risk-off" in macro.text(m_off)

    # strategy: weak business vetoes long-term buys; risk-off halves size; good business at fair price -> starter
    d = strategy.decide("longterm", "BUY", 10.0, "High", fw, m_on)
    assert d["final"] == "AVOID" and d["size_pct"] == 0, d
    on = strategy.decide("longterm", "BUY", 10.0, "High", fs, m_on)
    off = strategy.decide("longterm", "BUY", 10.0, "High", fs, m_off)
    assert on["final"] == "BUY" and on["size_pct"] > 0 and abs(off["size_pct"] - on["size_pct"] / 2) < 0.11, (on, off)
    assert off["parts"] == 3
    st_ = strategy.decide("longterm", "WAIT", 10.0, "Medium", fs, m_on)
    assert st_["starter"] and 0 < st_["size_pct"] < on["size_pct"], st_
    assert strategy.decide("positional", "BUY", 8.0, "High", fw, m_on)["final"] == "WAIT"
    assert strategy.decide("swing", "WAIT", None, None, fs, m_on)["size_pct"] == 0
    assert strategy.decide("longterm", "BUY", 10.0, "High", fs, m_on, room=2.0)["size_pct"] == 2.0

    # full detail report includes business, macro and portfolio-strategy sections
    data.financials = lambda s: strong
    data.profile = lambda s: {"sector": "Technology", "industry": "Information Technology Services"}
    cfg2 = {**cfg, "general": {**cfg["general"], "data_dir": str(tmp)}}  # tmp holds SBIN + TCS
    drep, dr, _ = swing.detail(cfg2, "TCS")
    dtitles = [x.title for x in drep.sections]
    for want in ("Final verdict", "Business & fundamentals: grade", "Key numbers", "Financial history", "Macro:",
                 "Portfolio strategy"):
        assert any(x.startswith(want) for x in dtitles), (want, dtitles)
    assert set(dr["strategy"]["styles"]) == {"swing", "positional", "longterm"}
    assert any("Already holding TCS" in row[0] and "5 shares" in row[1] for row in dr["strategy"]["fit"]["rows"])
    assert any(row[0].startswith("Correlation") for row in dr["strategy"]["fit"]["rows"])
    assert drep and any("checklist" in x.title for x in drep.sections)
    # investment advice: first section, business rows, factors, an action per style
    adv = dr["advice"]
    assert drep.sections[0].title.startswith("Investment advice: "), dtitles[0]
    assert adv["stance"] in advice.STANCE_KIND and adv["kind"] in ("buy", "wait", "avoid")
    assert {row[0] for row in adv["business"]} >= {"Growth", "Profitability", "Valuation"}, adv["business"]
    assert any(row[0] == "Business quality" and row[2] == "Positive" for row in adv["factors"]), adv["factors"]
    assert len(adv["plan"]) == 3 and all(row[3] for row in adv["plan"]) and adv["watch"] and adv["headline"]
    if adv["best"]:
        assert dr["strategy"]["styles"][adv["best"]]["size_pct"] > 0
    rpt.render_console(drep, Console(file=io.StringIO(), width=140))
    rpt.render_html(drep)

    import explain

    assert "RSI now" in explain.answer("what is RSI?", rb, "X")
    assert "BUY" in explain.answer("should I buy?", rb, "X")
    assert "trading days" in explain.answer("how long to target?", rb, "X")
    assert "50 shares" in explain.answer("what position size?")
    assert explain.answer("weather today") is None
    assert "Business & fundamentals" in explain.answer("how are the fundamentals?", dr, "TCS")
    assert "valuation" in explain.answer("is it expensive?", dr, "TCS")
    assert "Macro backdrop" in explain.answer("how is the macro?", dr, "TCS")
    assert "Portfolio strategy" in explain.answer("how much should I invest?", dr, "TCS")
    assert "Investment advice for TCS" in explain.answer("should I invest in it? suggest", dr, "TCS")
    assert "Business analysis" in explain.answer("I need investment advice", dr, "TCS")
    roe = explain.answer("what is ROE?", dr, "TCS")
    assert "ROE (return on equity)" in roe and "TCS ROE:" in roe, roe
    assert "Final call" in explain.answer("is it good to hold for 1-2 years?", dr, "TCS")
    assert explain.answer("what is the process?") is None  # 'roce' must not match inside words

    # missing financials: report still builds, final call falls back to chart + macro
    data.financials = lambda s: None
    data._cache.clear()
    nrep, nr, _ = swing.detail(cfg2, "TCS")
    assert nr["fund"] is None and any(x.title == "Business & fundamentals: not available" for x in nrep.sections)
    assert all(d["size_pct"] <= strategy.NO_FUND_CAP[k] for k, d in nr["strategy"]["styles"].items())
    assert nr["advice"] and not nr["advice"]["business"] and "not available" in nr["advice"]["profile"]

    # weak business: advice is AVOID for any holding period
    data.financials = lambda s: weak
    data._cache.clear()
    _, wr, _ = swing.detail(cfg2, "TCS")
    assert wr["advice"]["stance"] in ("AVOID", "SHORT-TERM TRADE ONLY"), wr["advice"]["stance"]
    assert wr["advice"]["cons"]

    # stock screener: ranking rules on hand-made rows
    import screener

    def srow(sym, fscore, val, bull, final, sector="IT", size=5.0):
        st_ = {"technical": final, "final": final, "starter": False, "size": size, "parts": 1, "why": [],
               "bull": bull, "total": 10, "entry": 100.0, "stop": 90.0, "t1": 120.0, "t1_pct": 20.0, "risk_pct": 10.0}
        return {"sym": sym, "name": sym, "sector": sector, "price": 100.0, "chg": 0.0, "date": None, "fscore": fscore,
                "grade": "A", "grade_word": "Strong", "valuation": val, "pe": 20.0, "roe": 20.0, "rev_cagr": 10.0,
                "q_ni_yoy": 10.0, "strengths": [], "flags": [], "tilt": "Neutral", "sector_index": None,
                "trend": "Uptrend", "stance": "INVEST", "headline": "", "styles": {"longterm": st_}}

    rows = [srow("A", 90, "Fair", 9, "BUY"), srow("B", 50, "Expensive", 9, "BUY"), srow("C", 80, "Fair", 8, "BUY"),
            srow("D", 95, "Attractive", 9, "WAIT"), srow("E", 85, "Fair", 7, "BUY", "Bank"),
            srow("F", 30, "Loss-making", 2, "AVOID", "Bank", 0.0)]
    assert screener.score(rows[0], "longterm") == 82.5
    rk = screener.rank(rows, "longterm", n=10, max_per_sector=2)
    assert [x["sym"] for x in rk["buy"]] == ["A", "C", "E"] and [x["rank"] for x in rk["buy"]] == [1, 2, 3]
    assert [x["sym"] for x in rk["watch"]] == ["D", "B"]  # WAIT and sector-capped; AVOID never listed
    assert [x["sym"] for x in screener.rank(rows, "longterm", n=10)["buy"]] == ["A", "C", "E", "B"]
    assert [x["sym"] for x in screener.rank(rows, "longterm", n=2)["buy"]] == ["A", "C"]
    assert rk["all"][0]["sym"] == "D" and screener.rank(rows, "swing")["all"] == []  # no swing data in these rows
    assert all(len(t) == len(screener.COLUMNS) for t in screener.table(rk["buy"] + rk["watch"], "longterm"))

    # stock screener: full run on synthetic prices, one stock without data
    data.financials = lambda s: weak if "SBIN" in s else strong
    data.history = lambda s, period="2y": None if "NODATA" in s else fake_history(s, period)
    data._cache.clear()
    done = []
    syms = ["TCS", "SBIN", "INFY", "ITC", "LT", "NODATA"]
    sc = screener.run(cfg2, [(x, x.title()) for x in syms], progress=lambda d, t, sym: done.append((d, t)))
    assert done[-1] == (6, 6) and sc["failed"] == ["NODATA"] and len(sc["rows"]) == 5, sc["failed"]
    assert sc["market"] and sc["date"] and not sc["stopped"]
    stopped = screener.run(cfg2, [(x, x) for x in syms], stop=lambda: True)
    assert stopped["stopped"] and stopped["rows"] == [] and stopped["failed"] == []
    for style in screener.STYLES:
        rk = screener.rank(sc["rows"], style, n=10, max_per_sector=2)
        scores = [x["score"] for x in rk["all"]]
        assert scores == sorted(scores, reverse=True) and all(0 <= v <= 100 for v in scores)
        assert all(screener.is_buy(x["styles"][style]) for x in rk["buy"])
        assert not {x["sym"] for x in rk["buy"]} & {x["sym"] for x in rk["watch"]}
        if style == "longterm":
            assert "SBIN" not in {x["sym"] for x in rk["buy"]}  # weak business is never a long-term buy
    data.history = fake_history
    # trend + outlook: uptrend reads as up; a no-trade bar gets steps and a time window
    assert "uptrend" in rb["trend"]["overall"].lower() and rb["history"]["BUY"]["ago"] <= 5
    rn = swing.analyze(shaped(path[:-8]), s, "Bullish")
    assert not rn["side"] and rn["outlook"]["steps"] and rn["outlook"]["when"], rn["outlook"]
    assert "When (estimate)" in explain.answer("when can I buy?", rn, "X")
    assert "trend" in explain.answer("what is the trend?", rn, "X")
    assert any(x.title.startswith("Trend:") for x in drep.sections)

    # positional (3-6 months) + long-term (1-2 years): steady uptrend -> buy-side, steady downtrend -> AVOID
    import horizons

    for key in horizons.STYLES:
        hu = horizons.analyze(shaped(up_long), key)
        assert hu["verdict"] in ("BUY", "BUY ON DIP"), (key, hu["verdict"], hu["conditions"])
        hp = hu["plan"]
        assert hp["stop"] < hp["entry"] < hp["t1"] <= hp["t2"] and hp["by1"] < hp["by2"], hp
        hd = horizons.analyze(shaped(20000 / up_long), key)
        assert hd["verdict"] == "AVOID" and hd["plan"] is None and hd["steps"], (key, hd["verdict"])
    assert horizons.analyze(shaped(up_long[:200]), "positional") is None
    # weekly values must not look ahead
    xf = horizons.frame(shaped(up_long))
    xp = horizons.frame(shaped(up_long).iloc[:-7])
    assert np.allclose(xf["e10"].iloc[:-7].dropna(), xp["e10"].dropna())
    rb["horizons"] = horizons.analyze_all(shaped(up_long), None)
    assert "1-2 years" in explain.answer("is it good to hold for 1-2 years?", rb, "X")
    assert "3-6 months" in explain.answer("can I hold for 6 months?", rb, "X")
    assert len(swing.holding_rows(rb)) == 3
    assert any(x.title.startswith("Holding period") for x in drep.sections)

    # autocomplete ranking
    cos = [("RELIANCE", "Reliance Industries Limited"), ("RPOWER", "Reliance Power Limited"),
           ("TATAMOTORS", "Tata Motors Limited"), ("RELAXO", "Relaxo Footwears Limited")]
    assert data.suggest("rel", cos)[0][0] in ("RELIANCE", "RELAXO")
    assert [x[0] for x in data.suggest("motors", cos)] == ["TATAMOTORS"]
    assert data.suggest("", cos) == []

    # empty portfolio + missing-data symbol do not crash
    rep2 = rpt.build(cfg, pf.load(tmp / "nope.csv"), with_watchlist=False)
    assert "empty" in rep2.sections[-1].rows[0][0]
    data.history = lambda s, period="2y": None
    data._cache.clear()
    rep3 = rpt.build(cfg, pf.load(p))
    assert rep3.session_date is None
    assert swing.scan(cfg, ["SBIN"]).sections[-1].rows == [["SBIN"]]
    assert swing.detail(cfg, "SBIN") is None

    print("ALL OFFLINE TESTS PASSED")
    buf = io.StringIO()
    data.history = fake_history
    rpt.render_console(rpt.build(cfg, pf.load(p)), Console(file=buf, width=130))
    print(buf.getvalue()[:3500])


if __name__ == "__main__":
    main()
