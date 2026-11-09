"""Small formatting helpers."""
from __future__ import annotations

import math

DASH = "–"


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def inr(x, decimals: int = 0) -> str:
    """Format a number with Indian digit grouping (12,34,567)."""
    if _bad(x):
        return DASH
    neg = x < 0
    ip, _, fp = f"{abs(x):.{decimals}f}".partition(".")
    if len(ip) > 3:
        head, tail = ip[:-3], ip[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        ip = ",".join(parts + [tail])
    return ("-" if neg else "") + ip + ("." + fp if fp else "")


def num(x, d: int = 2) -> str:
    return DASH if _bad(x) else f"{x:,.{d}f}"


def signed(x, d: int = 2, suffix: str = "") -> str:
    return DASH if _bad(x) else f"{x:+,.{d}f}{suffix}"


def pct(x, d: int = 2) -> str:
    return signed(x, d, "%")


def qty_fmt(q: float) -> str:
    return str(int(q)) if float(q).is_integer() else f"{q:g}"
