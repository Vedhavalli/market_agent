"""Portfolio storage. Holdings live in a plain CSV you can also edit by hand."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

COLS = ["symbol", "qty", "avg_price", "buy_date", "stop_loss", "target", "notes"]
REALIZED_COLS = ["date", "symbol", "qty", "sell_price", "avg_price", "pnl"]


def norm(symbol: str) -> str:
    s = symbol.strip().upper()
    return s[:-3] if s.endswith(".NS") else s


def load(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(path)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df["symbol"] = df["symbol"].astype(str).map(norm)
    for c in ("qty", "avg_price", "stop_loss", "target"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[COLS]


def save(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(path, index=False)


def add(df, symbol, qty, price, buy_date=None, stop=None, target=None, notes=""):
    """Buy more (or open) a position; average price is quantity-weighted."""
    symbol = norm(symbol)
    if qty <= 0 or price <= 0:
        raise ValueError("qty and price must be positive")
    df = df.copy()
    hit = df.index[df["symbol"] == symbol]
    if len(hit):
        i = hit[0]
        q0, p0 = float(df.at[i, "qty"]), float(df.at[i, "avg_price"])
        df.at[i, "qty"] = q0 + qty
        df.at[i, "avg_price"] = round((q0 * p0 + qty * price) / (q0 + qty), 4)
        if stop is not None:
            df.at[i, "stop_loss"] = stop
        if target is not None:
            df.at[i, "target"] = target
        if notes:
            df.at[i, "notes"] = notes
    else:
        row = {
            "symbol": symbol, "qty": qty, "avg_price": price,
            "buy_date": buy_date or date.today().isoformat(),
            "stop_loss": stop, "target": target, "notes": notes,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def sell(df, symbol, qty, price, realized_path: Path):
    """Reduce or close a position and log realised P&L (before charges and tax)."""
    symbol = norm(symbol)
    hit = df.index[df["symbol"] == symbol]
    if not len(hit):
        raise ValueError(f"{symbol} is not in the portfolio")
    i = hit[0]
    held, avg = float(df.at[i, "qty"]), float(df.at[i, "avg_price"])
    if qty <= 0 or qty > held:
        raise ValueError(f"cannot sell {qty:g}; you hold {held:g}")
    pnl = round((price - avg) * qty, 2)
    rp = Path(realized_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(rp) if rp.exists() else pd.DataFrame(columns=REALIZED_COLS)
    row = {"date": date.today().isoformat(), "symbol": symbol, "qty": qty,
           "sell_price": price, "avg_price": avg, "pnl": pnl}
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(rp, index=False)
    df = df.copy()
    if qty == held:
        df = df.drop(index=i).reset_index(drop=True)
    else:
        df.at[i, "qty"] = held - qty
    return df, pnl


def set_levels(df, symbol, stop=None, target=None):
    symbol = norm(symbol)
    hit = df.index[df["symbol"] == symbol]
    if not len(hit):
        raise ValueError(f"{symbol} is not in the portfolio")
    df = df.copy()
    if stop is not None:
        df.at[hit[0], "stop_loss"] = stop
    if target is not None:
        df.at[hit[0], "target"] = target
    return df
