"""Optional: ask Claude to write a short plain-English commentary on the computed report.

Needs `pip install anthropic` and the ANTHROPIC_API_KEY environment variable.
The model only sees the numbers already in the report, and is told not to invent data.
"""
from __future__ import annotations

SYSTEM = (
    "You summarise a daily Indian stock-market report for one retail investor. Use ONLY the numbers in the "
    "report; never invent prices, news or events. Write at most 180 words in plain English: (1) what the market "
    "did and why that is visible in the data, (2) what stands out in the portfolio and any alerts, (3) two or "
    "three things to watch next session. You are not a financial adviser: do not tell the user to buy or sell; "
    "frame everything as observations and things to research."
)


def commentary(report_text: str, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        return "(AI commentary skipped: run `pip install anthropic`.)"
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model, max_tokens=700, system=SYSTEM,
            messages=[{"role": "user", "content": report_text}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    except Exception as exc:  # network, auth, bad model name...
        return f"(AI commentary skipped: {exc})"


ASK_SYSTEM = (
    "You are a patient tutor helping an Indian retail investor understand a stock's technical analysis. The "
    "analysis below was computed by a rule-based tool from delayed daily prices. For anything about this stock use "
    "ONLY its numbers; never invent prices, news, fundamentals or events - say the tool does not have that data. "
    "Explain trading terms in simple English, with a small rupee example where it helps. Keep answers under 200 "
    "words unless asked for detail. You are not a SEBI-registered adviser: explain what the tool's rules say, but "
    "do not tell the user to buy or sell, and remind them to use a stop-loss and size positions sensibly."
)


def ask(question: str, context: str, history: list[dict], model: str) -> str | None:
    """Answer a user's doubt using the analysis text as context. Returns None if the AI is unavailable."""
    try:
        import anthropic
    except ImportError:
        return None
    past = [m for m in history[-8:] if m["role"] in ("user", "assistant")]
    while past and past[0]["role"] != "user":
        past = past[1:]
    try:
        msg = anthropic.Anthropic().messages.create(
            model=model, max_tokens=800, system=ASK_SYSTEM + "\n\nAnalysis:\n" + context,
            messages=past + [{"role": "user", "content": question}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip() or None
    except Exception:
        return None
