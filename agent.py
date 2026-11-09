#!/usr/bin/env python3
"""Market Agent: daily Indian market report + portfolio tracker, running locally.

  python agent.py report [--ai] [--no-watchlist] [--no-save]
  python agent.py holdings
  python agent.py add SBIN 10 850 [--stop 780] [--target 1000]
  python agent.py sell SBIN 5 940
  python agent.py set SBIN --stop 800 --target 1050
  python agent.py levels TCS
  python agent.py swing [SYMBOL ...] [--no-save]   # swing BUY/SELL signals with stop, targets, profit %, timeline
  python agent.py analyze TCS                     # full technical analysis + trade plan for one stock
  python agent.py daemon          # writes the report every weekday after the close
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

import data
import portfolio as pf_mod
import report as rpt
import swing
from indicators import analyze

HERE = Path(__file__).resolve().parent


def load_config(path: str | None) -> dict:
    with open(path or HERE / "config.toml", "rb") as f:
        return tomllib.load(f)


def paths(cfg: dict):
    base = Path(cfg["general"]["data_dir"])
    base = base if base.is_absolute() else HERE / base
    rep = Path(cfg["general"]["report_dir"])
    rep = rep if rep.is_absolute() else HERE / rep
    return base / "portfolio.csv", base / "realized.csv", rep


def run_report(cfg, ai=False, watchlist=True, save=True, require_today=False) -> Path | None:
    pf_path, _, rep_dir = paths(cfg)
    rep = rpt.build(cfg, pf_mod.load(pf_path), with_watchlist=watchlist)
    if require_today:
        today = datetime.now(ZoneInfo(cfg["general"]["timezone"])).date()
        if rep.session_date != today:
            print(f"No new trading session today (last data: {rep.session_date}); skipping.")
            return None
    commentary = ""
    if ai:
        import ai as ai_mod

        commentary = ai_mod.commentary(rpt.to_text(rep), cfg["ai"]["model"])
    rpt.render_console(rep)
    if commentary:
        print("AI commentary\n" + commentary + "\n")
    if not save:
        return None
    rep_dir.mkdir(parents=True, exist_ok=True)
    out = rep_dir / f"report-{rep.session_date or datetime.now().date()}.html"
    out.write_text(rpt.render_html(rep, commentary), encoding="utf-8")
    print(f"Saved: {out}")
    return out


def cmd_levels(cfg, symbol: str):
    a = analyze(data.history(symbol))
    if not a:
        sys.exit(f"No data for {symbol}")
    sec = rpt.Section(
        f"{symbol.upper()} technicals ({a['date']})", ["Metric", "Value"],
        [["Close", f"{a['last']:.2f}"], ["Chg %", rpt.pct(a["chg_pct"])], ["20/50/200-DMA",
         " / ".join(rpt.num(a[k]) for k in ("sma20", "sma50", "sma200"))], ["RSI(14)", rpt.num(a["rsi"], 1)],
         ["MACD hist", rpt.num(a["macd_hist"])], ["Support S2 / S1", f"{rpt.num(a['s2'])} / {rpt.num(a['s1'])}"],
         ["Pivot", rpt.num(a["pivot"])], ["Resistance R1 / R2", f"{rpt.num(a['r1'])} / {rpt.num(a['r2'])}"],
         ["20D low / high", f"{rpt.num(a['lo20'])} / {rpt.num(a['hi20'])}"],
         ["52W low / high", f"{rpt.num(a['lo52'])} / {rpt.num(a['hi52'])}"],
         ["Score", f"{a['score']:+d}  ({a['signal']})"], ["Why", ", ".join(a["why"])]],
        align="ll")
    rpt.render_console(rpt.Report(title=f"Levels for {symbol.upper()}", session_date=a["date"], sections=[sec]))


def run_swing(cfg, symbols=None, save=True) -> Path | None:
    rep = swing.scan(cfg, symbols or cfg["watchlist"]["symbols"])
    rpt.render_console(rep)
    if not save:
        return None
    _, _, rep_dir = paths(cfg)
    rep_dir.mkdir(parents=True, exist_ok=True)
    out = rep_dir / f"swing-{rep.session_date or datetime.now().date()}.html"
    out.write_text(rpt.render_html(rep), encoding="utf-8")
    print(f"Saved: {out}")
    return out


def cmd_analyze(cfg, symbol: str):
    res = swing.detail(cfg, symbol)
    if not res:
        sys.exit(f"No / insufficient data for {symbol} (needs ~{swing.MIN_BARS} sessions)")
    rpt.render_console(res[0])


def daemon(cfg):
    tz = ZoneInfo(cfg["general"]["timezone"])
    hh, mm = map(int, cfg["general"]["report_time"].split(":"))
    last_run = None
    print(f"Daemon running. Report at {hh:02d}:{mm:02d} {cfg['general']['timezone']} on weekdays. Ctrl+C to stop.")
    while True:
        now = datetime.now(tz)
        if now.weekday() < 5 and (now.hour, now.minute) >= (hh, mm) and last_run != now.date():
            last_run = now.date()
            data._cache.clear()
            try:
                if run_report(cfg, ai=False, require_today=True):
                    run_swing(cfg)
            except Exception as exc:  # keep the loop alive
                print(f"Report failed: {exc}")
        time.sleep(30)


def main(argv=None):
    p = argparse.ArgumentParser(description="Local Indian market report + portfolio tracker")
    p.add_argument("--config", help="path to config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="print and save the daily report")
    r.add_argument("--ai", action="store_true", help="add Claude commentary (needs ANTHROPIC_API_KEY)")
    r.add_argument("--no-watchlist", action="store_true")
    r.add_argument("--no-save", action="store_true")
    sub.add_parser("holdings", help="show portfolio with live prices and alerts")
    for name in ("add", "sell"):
        s = sub.add_parser(name, help=f"{name} shares")
        s.add_argument("symbol")
        s.add_argument("qty", type=float)
        s.add_argument("price", type=float)
        if name == "add":
            s.add_argument("--stop", type=float)
            s.add_argument("--target", type=float)
            s.add_argument("--date", help="buy date YYYY-MM-DD")
            s.add_argument("--notes", default="")
    st = sub.add_parser("set", help="set stop-loss / target for a holding")
    st.add_argument("symbol")
    st.add_argument("--stop", type=float)
    st.add_argument("--target", type=float)
    lv = sub.add_parser("levels", help="technicals for one symbol")
    lv.add_argument("symbol")
    sw = sub.add_parser("swing", help="swing BUY/SELL signals with stop-loss, targets, profit %% and timeline")
    sw.add_argument("symbols", nargs="*", help="default: watchlist from config.toml")
    sw.add_argument("--no-save", action="store_true")
    an = sub.add_parser("analyze", help="full technical analysis + trade plan + backtest for one stock")
    an.add_argument("symbol")
    sub.add_parser("daemon", help="run forever, report every weekday after market close")

    a = p.parse_args(argv)
    cfg = load_config(a.config)
    pf_path, realized_path, _ = paths(cfg)

    try:
        if a.cmd == "report":
            run_report(cfg, ai=a.ai, watchlist=not a.no_watchlist, save=not a.no_save)
        elif a.cmd == "holdings":
            run_report_pf_only(cfg, pf_path)
        elif a.cmd == "add":
            df = pf_mod.add(pf_mod.load(pf_path), a.symbol, a.qty, a.price, a.date, a.stop, a.target, a.notes)
            pf_mod.save(df, pf_path)
            print(f"Added {a.qty:g} {pf_mod.norm(a.symbol)} @ {a.price:g}")
        elif a.cmd == "sell":
            df, pnl = pf_mod.sell(pf_mod.load(pf_path), a.symbol, a.qty, a.price, realized_path)
            pf_mod.save(df, pf_path)
            print(f"Sold {a.qty:g} {pf_mod.norm(a.symbol)} @ {a.price:g}; realised P&L Rs {pnl:,.2f} "
                  "(before brokerage, STT and tax)")
        elif a.cmd == "set":
            pf_mod.save(pf_mod.set_levels(pf_mod.load(pf_path), a.symbol, a.stop, a.target), pf_path)
            print("Updated.")
        elif a.cmd == "levels":
            cmd_levels(cfg, a.symbol)
        elif a.cmd == "swing":
            run_swing(cfg, a.symbols, save=not a.no_save)
        elif a.cmd == "analyze":
            cmd_analyze(cfg, a.symbol)
        elif a.cmd == "daemon":
            daemon(cfg)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    except KeyboardInterrupt:
        print("\nStopped.")


def run_report_pf_only(cfg, pf_path):
    pf = pf_mod.load(pf_path)
    syms = list(pf["symbol"])
    ana = {k: analyze(v) for k, v in data.fetch_many(syms).items()}
    rpt.render_console(rpt.Report(title="Portfolio", session_date=None, sections=rpt._portfolio(cfg, ana, pf)))


if __name__ == "__main__":
    main()
