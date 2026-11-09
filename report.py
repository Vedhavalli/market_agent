"""Builds the daily report as a list of Sections, then renders to console / HTML / text."""
from __future__ import annotations

import html
from dataclasses import dataclass, field

import pandas as pd

import data
from indicators import analyze
from util import DASH, inr, num, pct, qty_fmt, signed

DISCLAIMER = (
    "Rule-based screens built from delayed public price data. This is information, not investment "
    "advice, and not a recommendation by a SEBI-registered adviser. Verify prices with your broker."
)


@dataclass
class Section:
    title: str
    columns: list[str]
    rows: list[list[str]]
    note: str = ""
    align: str = ""  # one char per column: l / r. Default: first column left, rest right.

    def aligns(self) -> str:
        return self.align or "l" + "r" * (len(self.columns) - 1)


@dataclass
class Report:
    title: str
    session_date: object
    sections: list[Section] = field(default_factory=list)


# ---------------------------------------------------------------- helpers
def _key(s: str) -> str:
    return data.yf_symbol(s)


def _a(ana: dict, name: str):
    return ana.get(_key(name))


def tone(cell: str) -> str:
    c = cell.strip()
    if c.startswith("+") or c in ("BUY-WATCH", "BUY", "Bullish", "Mild bullish") or c.startswith("TARGET"):
        return "pos"
    if c.startswith("-") or c in ("SELL-WATCH", "SELL", "Bearish", "Mild bearish") or c.startswith("STOP-LOSS"):
        return "neg"
    return ""


# ---------------------------------------------------------------- sections
def _market(cfg, ana) -> Section:
    rows = []
    for name, sym in cfg["indices"].items():
        r = _a(ana, sym)
        if not r:
            rows.append([name, DASH, DASH, DASH, DASH, DASH, DASH])
            continue
        vix = "VIX" in name.upper()
        d200 = (r["last"] / r["sma200"] - 1) * 100 if r["sma200"] else None
        rows.append([
            name, num(r["last"]), signed(r["chg"]), pct(r["chg_pct"]),
            DASH if vix else r["trend"], num(r["rsi"], 1), pct(d200, 1),
        ])
    return Section("1. Market snapshot", ["Index", "Last", "Chg", "Chg %", "Trend", "RSI", "vs 200-DMA"], rows,
                   align="lrrrlrr")


def _sectors(cfg, ana) -> Section:
    items = []
    for name, sym in cfg["sectors"].items():
        r = _a(ana, sym)
        if r:
            items.append((r["chg_pct"], [name, pct(r["chg_pct"]), r["trend"], num(r["rsi"], 1)]))
    items.sort(key=lambda t: t[0], reverse=True)
    return Section("2. Sector performance (best to worst)", ["Sector", "Chg %", "Trend", "RSI"],
                   [row for _, row in items], align="lrlr")


def _levels(cfg, ana) -> Section:
    names = cfg["technicals"]["indices"]
    cols = ["Level"] + names
    lookup = {n: _a(ana, cfg["indices"][n]) for n in names if n in cfg["indices"]}
    spec = [
        ("Close", "last"), ("20-DMA", "sma20"), ("50-DMA", "sma50"), ("200-DMA", "sma200"),
        ("RSI(14)", "rsi"), ("MACD hist", "macd_hist"),
        ("Resistance 2 (R2)", "r2"), ("Resistance 1 (R1)", "r1"), ("Pivot", "pivot"),
        ("Support 1 (S1)", "s1"), ("Support 2 (S2)", "s2"),
        ("20-day high", "hi20"), ("20-day low", "lo20"), ("52-week high", "hi52"), ("52-week low", "lo52"),
    ]
    rows = [[label] + [num(lookup[n][k]) if lookup.get(n) else DASH for n in names] for label, k in spec]
    rows.append(["Trend / score"] + [f"{lookup[n]['trend']} ({lookup[n]['score']:+d})" if lookup.get(n) else DASH
                                    for n in names])
    return Section("3. Technical levels", cols, rows,
                   note="Pivot levels are computed from the latest session's high/low/close.")


def _movers(cfg, ana) -> list[Section]:
    rows = []
    for s in cfg["watchlist"]["symbols"]:
        r = _a(ana, s)
        if r:
            rows.append((s, r))
    if not rows:
        return []
    rows.sort(key=lambda t: t[1]["chg_pct"], reverse=True)
    fmt = lambda t: [t[0], num(t[1]["last"]), pct(t[1]["chg_pct"]), num(t[1]["rsi"], 1), t[1]["trend"]]
    cols, al = ["Symbol", "LTP", "Chg %", "RSI", "Trend"], "lrrrl"
    return [
        Section("4a. Watchlist top gainers", cols, [fmt(t) for t in rows[:5]], align=al),
        Section("4b. Watchlist top losers", cols, [fmt(t) for t in rows[-5:][::-1]], align=al),
    ]


def _screens(cfg, ana) -> list[Section]:
    buys, sells = [], []
    for s in cfg["watchlist"]["symbols"]:
        r = _a(ana, s)
        if not r:
            continue
        row = [s, num(r["last"]), f"{r['score']:+d}", num(r["s1"]), num(r["r1"]), ", ".join(r["why"])]
        (buys if r["signal"] == "BUY-WATCH" else sells if r["signal"] == "SELL-WATCH" else []).append((r["score"], row))
    cols, al = ["Symbol", "LTP", "Score", "S1", "R1", "Why"], "lrrrrl"
    buys.sort(key=lambda t: -t[0])
    sells.sort(key=lambda t: t[0])
    note = ("Score = +/-1 for each of: price vs 50-DMA, price vs 200-DMA, 50 vs 200-DMA, MACD sign, RSI zone. "
            "BUY-WATCH needs score >= +3, SELL-WATCH <= -3. These are candidates to research, not instructions.")
    return [
        Section("5a. Buy-watch screen (technically strong)", cols, [r for _, r in buys[:8]] or [["none today"] + [DASH] * 5],
                note=note, align=al),
        Section("5b. Sell / reduce-watch screen (technically weak)", cols, [r for _, r in sells[:8]] or [["none today"] + [DASH] * 5],
                align=al),
    ]


def _portfolio(cfg, ana, pf: pd.DataFrame) -> list[Section]:
    if pf.empty:
        return [Section("6. Portfolio", ["Info"], [["Portfolio is empty. Add holdings with:  python agent.py add SBIN 10 850"]],
                        align="l")]
    limit = cfg["portfolio"]
    rows, alerts = [], []
    tot_inv = tot_val = tot_day = 0.0
    prelim = []
    for _, h in pf.iterrows():
        r = _a(ana, h["symbol"])
        q, avg = float(h["qty"]), float(h["avg_price"])
        val = q * r["last"] if r else q * avg
        prelim.append((h, r, q, avg, val))
    total_val = sum(p[4] for p in prelim) or 1.0

    for h, r, q, avg, val in prelim:
        sym = h["symbol"]
        inv = q * avg
        tot_inv += inv
        tot_val += val
        if not r:
            rows.append([sym, qty_fmt(q), num(avg), DASH, DASH, inr(val), DASH, DASH, DASH, "no data"])
            alerts.append([sym, "no price data (check the symbol / .NS .BO suffix)"])
            continue
        day = q * r["chg"]
        tot_day += day
        pnl = val - inv
        alloc = val / total_val * 100
        rows.append([sym, qty_fmt(q), num(avg), num(r["last"]), pct(r["chg_pct"]), inr(val),
                     _signed_inr(pnl), pct(pnl / inv * 100 if inv else 0),
                     f"{alloc:.1f}%", r["signal"]])
        stop, tgt = h["stop_loss"], h["target"]
        if pd.notna(stop):
            if r["last"] <= stop:
                alerts.append([sym, f"STOP-LOSS hit: {num(r['last'])} <= {num(stop)}"])
            elif r["last"] <= stop * (1 + limit["near_stop_pct"] / 100):
                alerts.append([sym, f"within {limit['near_stop_pct']:g}% of stop-loss {num(stop)}"])
        if pd.notna(tgt) and r["last"] >= tgt:
            alerts.append([sym, f"TARGET reached: {num(r['last'])} >= {num(tgt)}"])
        if r["signal"] == "SELL-WATCH":
            alerts.append([sym, f"technicals weak (score {r['score']:+d}): {', '.join(r['why'])}"])
        elif r["rsi"] and r["rsi"] > 75:
            alerts.append([sym, f"RSI {r['rsi']:.0f} overbought - review profit-booking / trailing stop"])
        if abs(r["chg_pct"]) >= limit["big_move_pct"]:
            alerts.append([sym, f"big move today ({pct(r['chg_pct'])})"])
        if alloc > limit["max_position_pct"]:
            alerts.append([sym, f"concentration: {alloc:.1f}% of portfolio (limit {limit['max_position_pct']:g}%)"])

    tot_pnl = tot_val - tot_inv
    rows.append(["TOTAL", "", "", "", pct(tot_day / (tot_val - tot_day) * 100 if tot_val - tot_day else 0),
                 inr(tot_val), _signed_inr(tot_pnl), pct(tot_pnl / tot_inv * 100 if tot_inv else 0), "100%", ""])
    cols = ["Symbol", "Qty", "Avg", "LTP", "Day %", "Value (Rs)", "P&L (Rs)", "P&L %", "Alloc", "Signal"]
    out = [Section("6. Portfolio", cols, rows,
                   note=f"Invested Rs {inr(tot_inv)} | Value Rs {inr(tot_val)} | Today's change Rs {_signed_inr(tot_day)} "
                        "(unrealised, before brokerage, STT and tax).", align="lrrrrrrrrl")]
    out.append(Section("7. Portfolio alerts", ["Symbol", "Alert"], alerts or [["all clear", "no alerts triggered"]],
                       align="ll"))
    return out


def _signed_inr(x: float) -> str:
    return ("+" if x >= 0 else "") + inr(x)


# ---------------------------------------------------------------- build
def build(cfg: dict, pf: pd.DataFrame, with_watchlist: bool = True) -> Report:
    wl = cfg["watchlist"]["symbols"] if with_watchlist else []
    cfg = {**cfg, "watchlist": {**cfg["watchlist"], "symbols": wl}}
    syms = list(cfg["indices"].values()) + list(cfg["sectors"].values()) + wl + list(pf["symbol"])
    hist = data.fetch_many(syms)
    ana = {k: analyze(v) for k, v in hist.items()}

    nifty = _a(ana, next(iter(cfg["indices"].values())))
    session = nifty["date"] if nifty else None
    rep = Report(title=f"Indian market report - session of {session or 'unknown date'}", session_date=session)
    rep.sections += [_market(cfg, ana), _sectors(cfg, ana), _levels(cfg, ana)]
    if with_watchlist:
        rep.sections += _movers(cfg, ana) + _screens(cfg, ana)
    rep.sections += _portfolio(cfg, ana, pf)
    return rep


# ---------------------------------------------------------------- renderers
def render_console(rep: Report, console=None) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = console or Console()
    console.print(f"\n[bold cyan]{escape(rep.title)}[/]")
    for s in rep.sections:
        t = Table(title=s.title, title_justify="left", header_style="bold")
        for col, a in zip(s.columns, s.aligns()):
            t.add_column(col, justify="left" if a == "l" else "right", overflow="fold")
        for row in s.rows:
            cells = []
            for c in row:
                tn = tone(c)
                cells.append(f"[green]{escape(c)}[/]" if tn == "pos" else f"[red]{escape(c)}[/]" if tn == "neg" else escape(c))
            t.add_row(*cells)
        console.print(t)
        if s.note:
            console.print(f"[dim]{escape(s.note)}[/]")
    console.print(f"[dim]{escape(DISCLAIMER)}[/]\n")


def render_html(rep: Report, commentary: str = "") -> str:
    e = html.escape
    css = (
        "body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1b1f24}"
        "h1{font-size:22px}h2{font-size:16px;margin:26px 0 6px}table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{padding:5px 8px;border-bottom:1px solid #e3e6ea}th{background:#f4f6f8;text-align:left}"
        ".r{text-align:right;font-variant-numeric:tabular-nums}.pos{color:#0a7d33}.neg{color:#c62828}"
        ".note{color:#667;font-size:12px;margin:4px 0}.box{background:#f4f6f8;padding:10px 14px;border-radius:6px;white-space:pre-wrap}"
    )
    out = [f"<!doctype html><meta charset='utf-8'><title>{e(rep.title)}</title><style>{css}</style>",
           f"<h1>{e(rep.title)}</h1>"]
    if commentary:
        out.append(f"<h2>AI commentary</h2><div class='box'>{e(commentary)}</div>")
    for s in rep.sections:
        out.append(f"<h2>{e(s.title)}</h2><table><tr>")
        for col, a in zip(s.columns, s.aligns()):
            out.append(f"<th class='{'r' if a == 'r' else ''}'>{e(col)}</th>")
        out.append("</tr>")
        for row in s.rows:
            out.append("<tr>")
            for c, a in zip(row, s.aligns()):
                out.append(f"<td class='{'r ' if a == 'r' else ''}{tone(c)}'>{e(c)}</td>")
            out.append("</tr>")
        out.append("</table>")
        if s.note:
            out.append(f"<div class='note'>{e(s.note)}</div>")
    out.append(f"<p class='note'>{e(DISCLAIMER)}</p>")
    return "".join(out)


def to_text(rep: Report) -> str:
    lines = [rep.title, ""]
    for s in rep.sections:
        lines.append(s.title)
        lines.append(" | ".join(s.columns))
        lines += [" | ".join(r) for r in s.rows]
        if s.note:
            lines.append(f"({s.note})")
        lines.append("")
    return "\n".join(lines)
