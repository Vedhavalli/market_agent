"""Market data via yfinance (free, unofficial Yahoo Finance data, ~15 min delayed)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

_cache: dict = {}


def yf_symbol(s: str) -> str:
    """'SBIN' -> 'SBIN.NS'. Indices (^NSEI) and suffixed symbols (X.BO) pass through."""
    s = s.strip().upper()
    if s.startswith("^") or "." in s or "=" in s:
        return s
    return s + ".NS"


def history(symbol: str, period: str = "2y") -> pd.DataFrame | None:
    key = (yf_symbol(symbol), period)
    if key in _cache:
        return _cache[key]
    df = None
    try:
        import yfinance as yf

        raw = yf.Ticker(key[0]).history(period=period, interval="1d", auto_adjust=False)
        if raw is not None and not raw.empty:
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
    except Exception:
        df = None
    _cache[key] = df
    return df


def fetch_many(symbols: list[str], workers: int = 6, period: str = "2y") -> dict[str, pd.DataFrame | None]:
    """Fetch history for many symbols concurrently. Keys are yf_symbol()-normalised."""
    syms = list(dict.fromkeys(yf_symbol(s) for s in symbols))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda s: history(s, period), syms))
    return dict(zip(syms, results))


def profile(symbol: str) -> dict:
    """Company name, sector and industry from Yahoo search ({} when unavailable)."""
    ys = yf_symbol(symbol)
    key = ("profile", ys)
    if key in _cache:
        return _cache[key]
    out = {}
    try:
        import yfinance as yf

        for q in yf.Search(ys, max_results=5, news_count=0).quotes:
            if str(q.get("symbol", "")).upper() == ys:
                out = {"name": q.get("longname") or q.get("shortname") or ys,
                       "sector": q.get("sectorDisp") or q.get("sector") or "",
                       "industry": q.get("industryDisp") or q.get("industry") or ""}
                break
    except Exception:
        out = {}
    _cache[key] = out
    return out


def analyst_snapshot(recommendations, targets) -> dict | None:
    """Normalize Yahoo's latest recommendation counts and target range; no values are inferred when absent."""
    if recommendations is None or getattr(recommendations, "empty", True):
        return None
    table = recommendations.copy()
    if "period" in table.columns:
        current = table[table["period"].astype(str) == "0m"]
        row = current.iloc[0] if not current.empty else table.iloc[0]
        period = str(row.get("period", "unknown"))
    else:
        row, period = table.iloc[0], "unknown"
    keys = ("strongBuy", "buy", "hold", "sell", "strongSell")
    counts = {}
    for key in keys:
        try:
            counts[key] = max(0, int(row.get(key, 0)))
        except (TypeError, ValueError, OverflowError):
            counts[key] = 0
    total = sum(counts.values())
    if total < 1:
        return None
    bullish, bearish = counts["strongBuy"] + counts["buy"], counts["sell"] + counts["strongSell"]
    consensus = ("Buy consensus" if bullish / total >= 0.6 and bearish / total < 0.2 else
                 "Sell consensus" if bearish / total >= 0.3 else "Mixed / hold consensus")
    target_data = targets if isinstance(targets, dict) else {}

    def target(key):
        try:
            value = float(target_data[key])
            return value if pd.notna(value) and value > 0 else None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    return {"source": "Yahoo Finance analyst data via yfinance", "period": period, **counts, "total": total,
            "bullish_pct": bullish / total * 100, "bearish_pct": bearish / total * 100,
            "consensus": consensus, "target_mean": target("mean"), "target_median": target("median"),
            "target_low": target("low"), "target_high": target("high"), "target_current": target("current")}


def financials(symbol: str) -> dict | None:
    """Reported statements (annual + quarterly), dividends and share count. Yahoo's ratio endpoint (`info`) is
    often blocked, so fundamentals.py computes the ratios itself from these statements."""
    ys = yf_symbol(symbol)
    key = ("fin", ys)
    if key in _cache:
        return _cache[key]

    def get(fn):
        try:
            v = fn()
            return None if v is None or getattr(v, "empty", False) else v
        except Exception:
            return None

    out = None
    try:
        import yfinance as yf

        t = yf.Ticker(ys)
        inc = get(lambda: t.income_stmt)
        if inc is not None:
            recommendations = get(lambda: t.recommendations_summary)
            if recommendations is None:
                recommendations = get(lambda: t.recommendations)
            try:
                targets = t.get_analyst_price_targets()
            except Exception:
                targets = None
            out = {"income": inc, "qincome": get(lambda: t.quarterly_income_stmt),
                   "balance": get(lambda: t.balance_sheet), "cash": get(lambda: t.cashflow),
                   "dividends": get(lambda: t.dividends), "shares": get(lambda: float(t.fast_info["shares"])),
                   "analyst": analyst_snapshot(recommendations, targets),
                   **profile(symbol)}
    except Exception:
        out = None
    _cache[key] = out
    return out


def search(query: str, limit: int = 8) -> list[tuple[str, str]]:
    """Company name -> NSE/BSE listings via Yahoo search, NSE first: [('RELIANCE.NS', 'Reliance Industries'), ...]."""
    try:
        import yfinance as yf

        quotes = yf.Search(query, max_results=limit * 2, news_count=0).quotes
    except Exception:
        return []
    out = []
    for q in quotes:
        sym = str(q.get("symbol", ""))
        if sym.endswith((".NS", ".BO")) and q.get("quoteType") == "EQUITY":
            out.append((sym, q.get("longname") or q.get("shortname") or sym))
    out.sort(key=lambda t: not t[0].endswith(".NS"))
    return out[:limit]


NSE_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_LIST_FILE = Path(__file__).parent / "data" / "nse_equity.csv"


def _cached_csv(url: str, f: Path, header: str, max_age_days: int = 7) -> pd.DataFrame | None:
    """Download a CSV (if the disk copy is older than `max_age_days`) and read it; None when unavailable."""
    fresh = f.exists() and time.time() - f.stat().st_mtime < max_age_days * 86400
    if not fresh:
        try:
            import requests

            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            resp.raise_for_status()
            if resp.text.lstrip("\ufeff").startswith(header):
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(resp.text, encoding="utf-8")
        except Exception:
            pass  # fall back to the cached copy (if any)
    if not f.exists():
        return None
    try:
        return pd.read_csv(f, encoding="utf-8-sig")
    except Exception:
        return None


INDEX_URL = "https://www.niftyindices.com/IndexConstituent/{}"
INDEX_FILES = {"Nifty 50": "ind_nifty50list.csv", "Nifty Next 50": "ind_niftynext50list.csv",
               "Nifty 100": "ind_nifty100list.csv", "Nifty Midcap 50": "ind_niftymidcap50list.csv",
               "Nifty 500": "ind_nifty500list.csv"}


def index_members(name: str, max_age_days: int = 7) -> list[tuple[str, str, str]]:
    """Constituents of a Nifty index [(SYMBOL, Company name, Industry)], cached on disk for a week."""
    file = INDEX_FILES[name]
    df = _cached_csv(INDEX_URL.format(file), Path(__file__).parent / "data" / file, "Company Name", max_age_days)
    if df is None:
        return []
    try:
        df.columns = [c.strip() for c in df.columns]
        return [(str(s).strip(), str(n).strip(), str(i).strip())
                for s, n, i in zip(df["Symbol"], df["Company Name"], df["Industry"])]
    except Exception:
        return []


def nse_companies(max_age_days: int = 7) -> list[tuple[str, str]]:
    """All NSE-listed equities [(SYMBOL, Company name)], downloaded from NSE and cached on disk for a week."""
    df = _cached_csv(NSE_LIST_URL, NSE_LIST_FILE, "SYMBOL", max_age_days)
    if df is None:
        return []
    try:
        df.columns = [c.strip().upper() for c in df.columns]
        return list(zip(df["SYMBOL"].astype(str).str.strip(), df["NAME OF COMPANY"].astype(str).str.strip()))
    except Exception:
        return []


def suggest(term: str, companies: list[tuple[str, str]], limit: int = 10,
            prefer: set[str] | frozenset = frozenset()) -> list[tuple[str, str]]:
    """Rank companies for an autocomplete box: symbol prefix, name prefix, word prefix, then substring.
    Symbols in `prefer` (e.g. the large-cap watchlist) come first within each rank."""
    t = term.strip().upper()
    if not t:
        return []
    ranked = []
    for sym, name in companies:
        n = name.upper()
        if sym == t:
            rank = 0
        elif sym.startswith(t):
            rank = 1
        elif n.startswith(t):
            rank = 2
        elif any(w.startswith(t) for w in n.split()):
            rank = 3
        elif t in n or t in sym:
            rank = 4
        else:
            continue
        ranked.append((rank, sym not in prefer, len(name), sym, name))
    ranked.sort()
    return [(sym, name) for *_, sym, name in ranked[:limit]]
