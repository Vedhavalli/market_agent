"""Offline answers to common doubts, so the app explains things even without an AI key.

`answer(question, r)` returns a plain-English reply built from the stock's analysis dict `r` (from swing.analyze),
or None when no topic matches.
"""
from __future__ import annotations

import re

import advice
import fundamentals
import horizons
import macro
import strategy
from util import num

LONG_WORDS = ("long term", "long-term", "longterm", "1 year", "2 year", "1-2 year", "years", "invest")
POS_WORDS = ("positional", "3 month", "6 month", "3-6 month", "months")
FUND_WORDS = ("fundamental", "business", "financial", "debt", "roe", "roce", "valuation", "p/e", "pe ratio",
              "earnings", "revenue", "sales growth", "profit growth", "expensive", "cheap", "overvalued",
              "undervalued", "balance sheet", "results", "dividend", "company quality", "management")
MACRO_WORDS = ("macro", "economy", "rupee", "crude", "oil price", "vix", "interest rate", "fii", "inflation",
               "global", "sector", "dollar", "bond yield", "market condition")
PF_WORDS = ("portfolio", "allocat", "how much", "how many shares", "diversif", "my capital", "position size",
            "strategy")
ADVICE_WORDS = ("advice", "advise", "recommend", "suggest", "should i invest", "worth investing",
                "good investment", "invest in", "good to invest")

GLOSSARY = {
    "rsi": ("RSI (Relative Strength Index)",
            "RSI measures how strong recent up-moves are versus down-moves, on a 0-100 scale. Above 70 = overbought "
            "(the rise may be stretched), below 30 = oversold, 50-70 = healthy uptrend momentum, 30-50 = weak."),
    "macd": ("MACD",
             "MACD is the 12-day EMA minus the 26-day EMA. Its 9-day average is the signal line; the gap between them is "
             "the histogram. Histogram above 0 = momentum is up. A 'bullish crossover' is when it turns from negative to positive."),
    "ema": ("EMA (Exponential Moving Average)",
            "An average of closing prices that gives more weight to recent days. Price above the 20-EMA = short-term "
            "uptrend, above the 50-EMA = medium-term uptrend, above the 200-EMA = long-term uptrend."),
    "dma": ("DMA (Daily Moving Average)",
            "Same idea as EMA: the average closing price of the last N days. The 50 and 200-day averages are watched most."),
    "adx": ("ADX",
            "ADX measures trend STRENGTH (not direction) from 0-100. Below 20 = no clear trend (sideways), 20-25 = trend "
            "forming, above 25 = strong trend. +DI above -DI means buyers are in control."),
    "supertrend": ("Supertrend",
                   "A trend-following line drawn using ATR. When price is above the line it is green (uptrend) and the line "
                   "acts as a trailing stop-loss; a close below it flips the trend to down."),
    "atr": ("ATR (Average True Range)",
            "The average daily price range over 14 days. It tells you how much the stock normally moves in a day. "
            "The app uses ATR to place stop-losses (not too tight, not too loose) and to estimate the timeline."),
    "obv": ("OBV (On-Balance Volume)",
            "Running total that adds volume on up days and subtracts it on down days. OBV rising faster than its average "
            "means buying volume is stronger: 'smart money' accumulation."),
    "stochastic": ("Stochastic oscillator",
                   "Shows where today's close sits within the last 14 days' range (0-100). Above 80 = near the top of the "
                   "range (overbought), below 20 = near the bottom (oversold)."),
    "bollinger": ("Bollinger Bands",
                  "A 20-day average with bands 2 standard deviations above and below. Price at the upper band = strong / "
                  "stretched, at the lower band = weak / stretched. Narrow bands often come before a big move."),
    "pivot": ("Pivot points",
              "Levels calculated from yesterday's high, low and close. Traders use R1/R2 as possible resistance and S1/S2 "
              "as possible support for the next session."),
    "support": ("Support and resistance",
                "Support is a price where buyers stepped in before (price bounced). Resistance is where sellers appeared "
                "(price turned down). The app finds them from recent swing lows and highs."),
    "resistance": ("Support and resistance",
                   "Resistance is a price where sellers appeared before and the stock turned down. A close above it is a "
                   "breakout. Support is the opposite: where buyers stepped in."),
    "breakout": ("Breakout",
                 "When the price closes above a recent high (here: the 20-day high). It shows buyers are strong enough to "
                 "push past where sellers stopped it before."),
    "stop": ("Stop-loss",
             "A pre-decided exit price that limits your loss if the trade goes wrong. Place the order with your broker "
             "right after buying. Never move a stop-loss further away to 'give it room'."),
    "target": ("Targets",
               "Target 1 = entry + 1.5 x risk, Target 2 = entry + 3 x risk (risk = entry - stop-loss). A common plan: sell "
               "half at T1, move the stop-loss to your entry price, and let the rest run to T2."),
    "risk": ("Risk : reward and position size",
             "Risk per share = entry - stop-loss. A common rule is to risk at most 1-2% of your capital per trade. "
             "Example: Rs 1,00,000 capital, 1% risk = Rs 1,000. If risk per share is Rs 20, buy 1000 / 20 = 50 shares."),
    "swing": ("Swing trading",
              "Holding a stock for a few days to a few weeks to capture one 'swing' (a move up or down), using a stop-loss "
              "and targets. It is between intraday trading and long-term investing."),
    "positional": ("Positional trading (3-6 months)",
                   "Holding for a few months to ride a medium-term trend. The app reads the WEEKLY chart: 10/30-week "
                   "averages, weekly MACD and RSI, 6-month strength vs Nifty and distance from the 52-week high. The "
                   "stop-loss is checked on the Friday close, so normal daily noise does not shake you out."),
    "long term": ("Long-term investing (1-2 years)",
                  "Holding for a year or more to ride the primary trend. The app checks the 40-week (~200-day) average "
                  "and its slope, the 50/200-DMA golden cross, 12-month return vs Nifty, the 52-week range and higher "
                  "yearly lows. For such long holds also check the company's profits, debt and valuation."),
    "scalp": ("Scalping / intraday",
              "Scalpers hold for minutes and intraday traders exit the same day. That needs live tick or 1-minute data "
              "and a broker terminal; this app uses delayed daily data, so it covers swing (days-weeks), positional "
              "(3-6 months) and long-term (1-2 years) holding instead."),
    "score": ("Score (0-9)",
              "The app checks 9 technical conditions (trend, moving averages, MACD, RSI, ADX, Supertrend, volume, weekly "
              "trend). Each bullish one adds 1 to the bullish score. A BUY needs 7+ AND a fresh trigger in the last 3 days."),
    "backtest": ("Backtest",
                 "The app replays its own rules on the stock's last ~10 years: buy at the next day's open, exit at the "
                 "stop-loss, Target 1, or after 20 days. Win rate shows how often that worked. Past results do not "
                 "guarantee the future."),
    "confidence": ("Confidence",
                   "High = score 8+, the Nifty trend agrees and past signals on this stock won 50%+ of the time. Low = the "
                   "market trend is against the trade or past win rate was below 40%. Otherwise Medium."),
    "short": ("Selling / shorting",
              "In India you cannot hold a short position in shares overnight in the cash market. A SELL signal means: "
              "exit if you hold it, avoid buying. To profit from a fall you need F&O (futures or put options), which "
              "is riskier."),
    "candle": ("Candlestick patterns",
               "Each candle shows a day's open, high, low and close. Hammer = buyers pushed price back up from the low "
               "(bullish). Shooting star = sellers pushed it down from the high (bearish). Engulfing = today's body covers "
               "yesterday's (a reversal sign)."),
    "volume": ("Volume",
               "Number of shares traded. Price moves on high volume (1.5x+ the 20-day average) are more reliable than "
               "moves on low volume."),
    "fundamental": ("Fundamental analysis",
                    "Judging the business behind the share: is it growing sales and profit, earning good returns on "
                    "its capital, carrying little debt, producing real cash, and is the share price reasonable for "
                    "that? The tool grades it A (strong) to D (weak) from 3-5 years of reported statements."),
    "p/e": ("P/E (price to earnings)",
            "Share price divided by profit per share for the last 12 months. P/E 20 = you pay Rs 20 for every Rs 1 of "
            "yearly profit. Fast-growing companies deserve a higher P/E; compare with growth (PEG) and peers."),
    "peg": ("PEG ratio",
            "P/E divided by yearly profit growth (%). Around 1 or below = fair for the growth; above 2-3 = you are "
            "paying a lot for the growth."),
    "p/b": ("P/B (price to book)",
            "Share price divided by book value (net worth) per share. Mainly used for banks and lenders, where 1-3 is "
            "typical and quality lenders trade higher."),
    "roe": ("ROE (return on equity)",
            "Yearly profit as a % of shareholders' money. Above 15% is good; it shows how well the company uses "
            "your money. Check it is not inflated by heavy debt."),
    "roce": ("ROCE (return on capital employed)",
             "Operating profit as a % of all capital used (equity + debt). Above 15% is good: the business earns well "
             "on every rupee invested, whoever funded it."),
    "debt to equity": ("Debt to equity",
                       "Borrowings divided by shareholders' money. Under 0.5 is comfortable, above 1.5 is risky for "
                       "non-financial companies (banks naturally run with high leverage and are judged differently)."),
    "vix": ("India VIX",
            "The 'fear gauge': expected Nifty volatility over the next month. Below 15 = calm; above 22 = fear and "
            "big swings, so trade smaller."),
    "fii": ("FII / FPI flows",
            "Foreign investors' buying or selling of Indian shares. Heavy selling usually comes with a weaker rupee "
            "and rising US bond yields, and pressures large caps and banks."),
    "macro": ("Macro analysis",
              "The big-picture backdrop: the Nifty trend, volatility (VIX), the rupee, US interest rates (foreign "
              "money flows) and crude oil, plus whether the stock's sector is leading or lagging the market."),
}
ALIASES = {
    "relative strength": "rsi", "moving average": "ema", "stoploss": "stop", "stop loss": "stop", "sl": "stop",
    "t1": "target", "t2": "target", "profit": "target", "position size": "risk", "how many shares": "risk",
    "quantity": "risk", "win rate": "backtest", "bollinger band": "bollinger", "short sell": "short",
    "pattern": "candle", "hammer": "candle", "doji": "candle", "engulfing": "candle",
    "long-term": "long term", "intraday": "scalp",
    "price to earnings": "p/e", "pe ratio": "p/e", "price to book": "p/b", "return on equity": "roe",
    "return on capital": "roce", "d/e": "debt to equity", "leverage": "debt to equity", "fpi": "fii",
    "foreign investor": "fii", "fear": "vix",
}


def _has(q: str, k: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(k), q) is not None


def _topic(q: str) -> str | None:
    for k, v in ALIASES.items():
        if _has(q, k):
            return v
    for k in GLOSSARY:
        if _has(q, k):
            return k
    return None


def _final_line(r: dict, style: str) -> str:
    d = (r.get("strategy") or {}).get("styles", {}).get(style)
    if not d:
        return ""
    return (f"\n\n**Final call after business & macro checks: {d['final']}" + (" (start small)" if d["starter"] else "")
            + f".** {d['how']} " + " ".join(d["why"]))


def _stock_reply(q: str, r: dict, sym: str) -> str | None:
    side, L = r["side"], r["L"]
    hz = r.get("horizons") or {}
    asks_def = q.startswith(("what is", "what's", "explain", "define", "meaning"))
    if not asks_def:
        if r.get("strategy") and any(_has(q, w) for w in PF_WORDS):
            return strategy.text(r["strategy"], sym)
        if r.get("advice") and any(_has(q, w) for w in ADVICE_WORDS):
            return advice.text(r["advice"], sym)
        if any(_has(q, w) for w in FUND_WORDS):
            return fundamentals.text(r.get("fund"), sym)
        if any(_has(q, w) for w in MACRO_WORDS):
            return macro.text(r.get("macro"))
    style = ("longterm" if any(w in q for w in LONG_WORDS) else
             "positional" if any(w in q for w in POS_WORDS) else None)
    if style in hz and not asks_def:
        return horizons.text(hz[style], sym) + _final_line(r, style)
    if any(w in q for w in ("should i buy", "buy now", "good time", "should i sell", "verdict", "summary",
                            "what do you think", "is it good")):
        if side:
            p = r["plan"]
            return (f"The tool's rules currently show a **{side}** signal for {sym} with **{r['confidence']}** confidence "
                    f"(score {r['score']}/9). Plan: entry ~{num(p['entry'])}, stop-loss {num(p['stop'])} "
                    f"(risk {p['risk_pct']:.1f}%), T1 {num(p['t1'])} (+{p['t1_pct']:.1f}%), T2 {num(p['t2'])} "
                    f"(+{p['t2_pct']:.1f}%). This is a rule-based screen, not advice: decide your own position size "
                    "and always place the stop-loss.") + _final_line(r, "swing")
        return (f"There is **no trade signal** for {sym} right now. {r.get('why_none', '')}\n\n"
                f"A bullish trigger would be a close above {num(L['hi20'])}; a bearish one a close below {num(L['lo20'])}."
                + "".join(f"\n\n- {d['label']} ({d['period']}): chart {d['technical']}, final **{d['final']}**"
                          + (f", up to {d['size_pct']:g}% of capital" if d["size_pct"] else "")
                          for d in (r.get("strategy") or {}).get("styles", {}).values()))
    if any(w in q for w in ("why", "reason")):
        bulls = [c for c, s in r["conditions"] if s == "Bullish"]
        bears = [c for c, s in r["conditions"] if s == "Bearish"]
        trig = r["trig_buy"] + r["trig_sell"]
        return ((f"**{r['why_none']}**\n\n" if not side else "") +
                f"Bullish conditions ({len(bulls)}): {', '.join(bulls) or 'none'}.\n\n"
                f"Bearish conditions ({len(bears)}): {', '.join(bears) or 'none'}.\n\n"
                f"Fresh triggers: {', '.join(trig) or 'none'}. A signal needs 7+ conditions on one side plus a trigger.")
    if any(w in q for w in ("how long", "when", "timeline", "days")) and side:
        p = r["plan"]
        e = p["t1_eta"]
        return (f"Target 1 is estimated in **{e['fast']}-{e['slow']} trading days** (by {e['by']:%d %b}). This comes from "
                f"the stock's ATR ({num(L['atr'])}/day): fastest assumes a steady 0.8 ATR/day of progress, slowest 0.3. "
                "If T1 is not reached in about 20 sessions, the setup has likely failed - consider exiting.")
    if any(w in q for w in ("when", "how long", "wait", "next", "future", "timeline", "days", "outlook")) and not side:
        ok = r["outlook"]
        steps = "\n".join(f"- {t}" for t in ok["steps"])
        return (f"**Why not today:** {r['why_none']}\n\n**What to wait for:**\n{steps}\n\n**When (estimate):** {ok['when']}")
    if "trend" in q:
        tv = r["trend"]
        rows = "\n".join(f"- {h}: **{t}** ({rt:+.1f}% in {per}; {rd})" for h, t, rt, per, rd in tv["rows"])
        hist = "\n".join(f"- Last {sd} signal: {h['date']:%d %b %Y} at {num(h['price'])} ({h['ago']} sessions ago, "
                         f"{h['since_pct']:+.1f}% since)" for sd, h in r["history"].items())
        return f"{sym} trend: **{tv['overall']}**, {tv['strength']}.\n\n{rows}" + (f"\n\n{hist}" if hist else "")
    return None


def answer(question: str, r: dict | None = None, sym: str = "") -> str | None:
    q = question.lower()
    if r:
        reply = _stock_reply(q, r, sym)
        if reply:
            return reply
    t = _topic(q)
    if not t:
        return None
    title, text = GLOSSARY[t]
    out = f"**{title}**\n\n{text}"
    if r:
        L = r["L"]
        now = {
            "rsi": f"{sym} RSI now: {num(L['rsi'], 1)}.", "macd": f"{sym} MACD histogram now: {num(L['macd_hist'])}.",
            "ema": f"{sym}: close {num(r['last'])}, 20/50/200-EMA {num(L['ema20'])} / {num(L['ema50'])} / {num(L['ema200'])}.",
            "adx": f"{sym} ADX now: {num(L['adx'], 1)} (+DI {num(L['pdi'], 1)}, -DI {num(L['mdi'], 1)}).",
            "supertrend": f"{sym} Supertrend is {'UP' if L['st_dir'] == 1 else 'DOWN'}, line at {num(L['st_line'])}.",
            "atr": f"{sym} ATR now: {num(L['atr'])} ({L['atr'] / r['last'] * 100:.1f}% of price).",
            "support": f"{sym} supports: {', '.join(num(v) for v in r['sup']) or 'none nearby'}; "
                       f"resistances: {', '.join(num(v) for v in r['res']) or 'none nearby'}.",
        }.get(t)
        f = r.get("fund")
        if not now and f:
            now = {"roe": f"{sym} ROE: {num(f['roe'], 1)}%.", "roce": f"{sym} ROCE: {num(f['roce'], 1)}%.",
                   "p/e": f"{sym} P/E: {num(f['pe'], 1)} (valuation: {f['valuation'].lower()}).",
                   "peg": f"{sym} PEG: {num(f['peg'], 2)}.", "p/b": f"{sym} P/B: {num(f['pb'], 2)}.",
                   "debt to equity": f"{sym} debt / equity: {num(f['de'])}.",
                   "fundamental": f"{sym}: {f['summary']}"}.get(t)
        m = r.get("macro")
        if not now and m and t in ("vix", "fii", "macro"):
            now = m["summary"]
        if now:
            out += f"\n\n{now}"
    return out
