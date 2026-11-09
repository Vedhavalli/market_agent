"""Stock analysis tab: type a share name, get the full analysis, ask doubts in a chat. Run via app.py."""
from __future__ import annotations

import html
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_searchbox import st_searchbox

import agent
import ai
import data
import explain
import horizons
import report as rpt
import swing
from util import num

cfg = agent.load_config(None)
PREFER = frozenset(s.upper() for s in cfg["watchlist"]["symbols"])


@st.cache_data(ttl=86400, show_spinner=False)
def companies():
    return data.nse_companies()


@st.cache_data(ttl=900, show_spinner=False)
def find(query: str):
    return data.search(query)


@st.cache_data(ttl=900, show_spinner=False)
def run_analysis(symbol: str, code_version: float):
    """`code_version` makes cached results expire when any module changes (st.cache_data only hashes this function)."""
    data._cache.clear()  # fresh data on each (cache-expired) run
    return swing.detail(cfg, symbol)


def section_df(sec: rpt.Section) -> pd.DataFrame:
    return pd.DataFrame(sec.rows, columns=sec.columns)


def suggestions(term: str) -> list[tuple[str, str]]:
    """Autocomplete: (label, 'SYMBOL.NS|Company') once 3+ characters are typed."""
    term = term.strip()
    if len(term) < 3:
        return []
    hits = [(f"{s}.NS", n) for s, n in data.suggest(term, companies(), prefer=PREFER)] or find(term)
    opts = [(f"{n}  ({s.split('.')[0]})", f"{s}|{n}") for s, n in hits]
    if not opts and term.replace(" ", "").isalnum():
        sym = term.replace(" ", "").upper()
        opts = [(f"Analyse '{sym}' as an NSE symbol", f"{data.yf_symbol(sym)}|{sym}")]
    return opts


# ---------------------------------------------------------------- page
st.title("📈 Indian Share Tips")
st.caption("Start typing a company name or NSE symbol (3+ letters, e.g. *rel*, *tat*, *inf*) and pick a suggestion. "
           "Technical analysis is recomputed from fresh daily data each time.")

picked = st_searchbox(suggestions, placeholder="Type 3+ letters of the company name...", label="Share name",
                      key="share_search", debounce=200)
if picked:
    st.session_state.pop("screener_pick", None)  # a new search replaces a stock opened from the screener
choice = picked or st.session_state.get("screener_pick")

if not choice:
    st.info("Type at least 3 letters and select a company from the suggestions to start the analysis.")
    st.stop()

symbol, name = choice.split("|", 1)
with st.spinner(f"Analysing {name}..."):
    res = run_analysis(symbol, max(p.stat().st_mtime for p in Path(__file__).parent.glob("*.py")))
if not res:
    st.error(f"No data for **{symbol}**. Check the name, or try the exact NSE symbol.")
    st.stop()
rep, r, df = res
sym = symbol.split(".")[0]
L, side = r["L"], r["side"]

# ---------------------------------------------------------------- verdict
st.subheader(f"{name} ({symbol}) - {r['date']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close", f"Rs {num(r['last'])}", f"{r['chg_pct']:+.2f}%")
c2.metric("Swing signal", side or "NO TRADE")
c3.metric("Score", f"{r['bull']}/9 bull · {r['bear']}/9 bear")
c4.metric("Market (Nifty)", r["regime"])

def banner(color: str, bg: str, title: str, lines: list[str]) -> None:
    body = "".join(f"<div style='font-size:1.15rem;margin-top:6px'>{x}</div>" for x in lines)
    st.markdown(f"<div style='border:3px solid {color};background:{bg};border-radius:12px;padding:16px 20px;"
                f"margin:8px 0 16px'><div style='font-size:1.9rem;font-weight:800;color:{color}'>{title}</div>"
                f"{body}</div>", unsafe_allow_html=True)


def b(text: str, color: str) -> str:
    return f"<b style='color:{color}'>{text}</b>"


GREEN, RED, AMBER = "#0a8f3c", "#c62828", "#b26a00"
hold_max = swing.settings(cfg)["max_hold_days"]
safe_name = html.escape(name)
ok = r["outlook"]
wait_lines = [f"➡️ {html.escape(t)}" for t in ok["steps"]]
if ok["when"]:
    wait_lines.append(f"⏱️ {b('When: ' + html.escape(ok['when']), AMBER)}")

# ---------------------------------------------------------------- investment advice
adv = r.get("advice")
if adv:
    st.markdown("## 🧭 Investment advice")
    kc, kbg = {"buy": (GREEN, "#e8f7ee"), "wait": (AMBER, "#fff6e0"), "avoid": (RED, "#fdecea")}[adv["kind"]]
    banner(kc, kbg, f"{html.escape(adv['stance'])}: {safe_name}", [html.escape(adv["headline"])])
    st.markdown("**What to do, by holding period**\n\n" + "\n".join(
        f"- **{s}: {c}**" + (f" · size {z}" if z != "–" else "") + f" — {t}" for s, c, z, t in adv["plan"]))
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**🏢 Business analysis**")
        st.write(adv["profile"])
        if adv["business"]:
            st.dataframe(pd.DataFrame(adv["business"], columns=["Area", "Assessment", "Evidence"]), hide_index=True,
                         width="stretch")
    with a2:
        st.markdown("**🌏 Other factors**")
        dot = {"Positive": "🟢 Positive", "Neutral": "🟡 Neutral", "Negative": "🔴 Negative"}
        st.dataframe(pd.DataFrame([[k, v, dot[i]] for k, v, i in adv["factors"]],
                                  columns=["Factor", "Reading", "Impact"]), hide_index=True, width="stretch")
    p1, p2 = st.columns(2)
    p1.markdown("**👍 In favour**\n\n" + ("\n".join(f"- {x}" for x in adv["pros"]) or "- nothing notable"))
    p2.markdown("**⚠️ Risks**\n\n" + ("\n".join(f"- {x}" for x in adv["cons"]) or "- no major red flags found"))
    if adv["watch"]:
        st.markdown("**🔄 What would change this view**\n\n" + "\n".join(f"- {x}" for x in adv["watch"]))
    st.caption("Rule-based analysis of public Yahoo Finance data (statements, prices, macro), not personalised "
               "financial advice. Management, competition and news are not assessed; check them before investing.")
    st.divider()

# ---------------------------------------------------------------- holding period
STYLE_OPTS = {"Swing (days to weeks)": "swing", "Positional (3-6 months)": "positional",
              "Long-term (1-2 years)": "longterm"}
style = STYLE_OPTS[st.radio("**How long do you want to hold?**", list(STYLE_OPTS), horizontal=True, key="style")]
hz = r.get("horizons", {})
if style != "swing" and style not in hz:
    st.warning("Not enough price history for this holding period (needs more than a year of data). "
               "Showing the swing view.")
    style = "swing"


def verdict_color(v: str) -> str:
    return GREEN if v == "BUY" else AMBER if v in ("WAIT", "BUY ON DIP") else RED


summary = swing.holding_rows(r)
stg, fund, mac = r.get("strategy"), r.get("fund"), r.get("macro")
finals = list(stg["styles"].values()) if stg else []
head = ["Style", "Hold for", "Chart verdict", "Final call", "Size", "Buy at", "Sell targets", "Stop-loss",
        "Past record"]
if finals:
    summary = [row[:3] + [fd["final"] + (" (starter)" if fd["starter"] else ""),
                          f"{fd['size_pct']:g}%" if fd["size_pct"] else "–"] + row[3:]
               for row, fd in zip(summary, finals)]
else:
    head = head[:3] + head[5:]
trs = "".join(
    "<tr>" + "".join(f"<td style='padding:4px 10px'>{b(html.escape(v), verdict_color(v.split(' (')[0])) if i in (2, 3) and head[i] != 'Buy at' else html.escape(v)}</td>"
                     for i, v in enumerate(row)) + "</tr>" for row in summary)
st.markdown("<table style='width:100%;font-size:0.95rem'><tr>"
            + "".join(f"<th style='text-align:left;padding:4px 10px'>{x}</th>" for x in head)
            + f"</tr>{trs}</table>", unsafe_allow_html=True)


def horizon_banner(h: dict) -> None:
    per, p, v = h["period"], h["plan"], h["verdict"]
    tail = [f"📊 {html.escape(h['record'])}", f"🧾 {html.escape(h['holder'])}"]
    plan_lines = []
    if p:
        tgt1 = "Rs %s  (%+.1f%% profit)" % (num(p["t1"]), p["t1_pct"])
        tgt2 = "Rs %s  (%+.1f%% profit)" % (num(p["t2"]), p["t2_pct"])
        stop = "Rs %s  (-%.1f%%)" % (num(p["stop"]), p["risk_pct"])
        c = GREEN if v == "BUY" else AMBER
        plan_lines = [
            f"⏱️ Hold for {b(per, c)}: Target 1 around {b(p['by1'].strftime('%b %Y'), c)}, "
            f"Target 2 around {b(p['by2'].strftime('%b %Y'), c)}. Review {h['review']}.",
            f"💰 Sell at {b(tgt1, c)} (~{h['names'][0]}), or hold for {b(tgt2, c)} (~{h['names'][1]}).",
            f"🛑 Stop-loss: exit on a weekly close below {b(stop, RED)}",
        ]
    steps = [f"➡️ {html.escape(t)}" for t in h["steps"]]
    if h["when"]:
        steps.append(f"⏱️ {b('When: ' + html.escape(h['when']), AMBER)}")
    if v == "BUY":
        banner(GREEN, "#e8f7ee", f"✅ GOOD TO BUY for {per}: {safe_name}", [
            f"Buy around {b('Rs ' + num(p['entry']), GREEN)} · Confidence {b(h['confidence'], GREEN)}",
        ] + plan_lines + tail)
    elif v == "BUY ON DIP":
        banner(AMBER, "#fff6e0", f"🟡 GOOD STOCK, BUY ON DIP for {per}: {safe_name}", [
            b("Why not right now: " + html.escape(h["why"]), AMBER),
        ] + steps + [f"If bought at {b('Rs ' + num(p['entry']), AMBER)} · Confidence {b(h['confidence'], AMBER)}"]
            + plan_lines + tail)
    elif v == "WAIT":
        banner(AMBER, "#fff6e0", f"⏳ NOT A {h['adj'].upper()} BUY YET: WAIT ({safe_name})", [
            b("Why not now: " + html.escape(h["why"]), AMBER)] + steps
            + ["No hold period or profit target until the setup turns bullish."] + tail)
    else:
        banner(RED, "#fdecea", f"❌ NOT GOOD TO BUY FOR {per.upper()}: AVOID {safe_name}", [
            b("Why not: " + html.escape(h["why"]), RED)] + steps + tail)


if stg and style in stg["styles"]:
    fd = stg["styles"][style]
    fc = verdict_color(fd["final"]) if not fd["starter"] else AMBER
    title = fd["final"] + (" (start small)" if fd["starter"] else "")
    if fd["size_pct"]:
        title += f" · up to {fd['size_pct']:g}% of capital"
    changed = fd["final"] != fd["technical"] or fd["starter"]
    ctx_line = " · ".join(x for x in (
        f"Business: grade {fund['grade']} ({fund['score']}/100), {fund['valuation'].lower()}" if fund and fund["grade"]
        else "Business: not graded", f"Market: {mac['regime']}, {mac['tilt'].lower()} for this stock" if mac else "") if x)
    banner(fc, "#f4f6fb", f"🧭 FINAL CALL ({fd['label'].upper()}): {title}", [
        f"Chart says {b(fd['technical'], verdict_color(fd['technical']))}"
        + (f" → after business & macro checks: {b(fd['final'], fc)}" if changed else " · business & macro agree"),
        html.escape(ctx_line)] + [f"• {html.escape(t)}" for t in fd["why"]] + [b(html.escape(fd["how"]), fc)])
    st.caption("Chart view for this holding period:")

if style != "swing":
    horizon_banner(hz[style])
elif side == "BUY":
    p = r["plan"]
    e1, e2 = p["t1_eta"], p["t2_eta"]
    hold1 = "%d-%d trading days" % (e1["fast"], e1["slow"])
    tgt1 = "Rs %s  (+%.1f%% profit)" % (num(p["t1"]), p["t1_pct"])
    tgt2 = "Rs %s  (+%.1f%% profit)" % (num(p["t2"]), p["t2_pct"])
    stop = "Rs %s  (-%.1f%%)" % (num(p["stop"]), p["risk_pct"])
    banner(GREEN, "#e8f7ee", f"✅ GOOD TO BUY: {safe_name}", [
        f"Buy around {b('Rs ' + num(p['entry']), GREEN)} · Confidence {b(r['confidence'], GREEN)}",
        f"⏱️ Hold for {b(hold1, GREEN)} (Target 1 expected by {e1['by']:%d %b %Y}). "
        f"Exit anyway after {b('%d sessions' % hold_max, GREEN)} if Target 1 is not hit.",
        f"💰 Sell at {b(tgt1, GREEN)} = Target 1, or hold for {b(tgt2, GREEN)} = Target 2 "
        f"(usually {e2['fast']}-{e2['slow']} days: hold for it only if Target 1 comes early, "
        "and move the stop-loss up to your buy price).",
        f"🛑 Stop-loss: sell if it closes below {b(stop, RED)}",
    ] + ([f"⚠️ Caution: {b(html.escape(', '.join(r['notes'])), AMBER)}"] if r["notes"] else []))
elif side == "SELL":
    p = r["plan"]
    fall1 = "Rs %s (-%.1f%%)" % (num(p["t1"]), p["t1_pct"])
    fall2 = "Rs %s (-%.1f%%)" % (num(p["t2"]), p["t2_pct"])
    banner(RED, "#fdecea", f"❌ NOT GOOD TO BUY: SELL / AVOID {safe_name}", [
        f"Downtrend signal · Confidence {b(r['confidence'], RED)}. If you hold it, consider exiting near "
        f"{b('Rs ' + num(p['entry']), RED)}.",
        f"Expected fall: {b(fall1, RED)} in {b(swing.eta_str(p['t1_eta']), RED)}, then {b(fall2, RED)}",
        f"The signal is cancelled if it closes above {b('Rs ' + num(p['stop']), GREEN)}.",
    ])
elif r["bull"] >= r["bear"]:
    banner(AMBER, "#fff6e0", f"⏳ NOT A BUY YET: WAIT ({safe_name})", [
        b("Why not today: " + html.escape(r["why_none"]), AMBER),
    ] + wait_lines + ["No hold period or profit target until a BUY signal appears."])
else:
    banner(RED, "#fdecea", f"❌ NOT GOOD TO BUY NOW: AVOID {safe_name}", [
        b("Why not today: " + html.escape(r["why_none"]), RED),
    ] + wait_lines)

# ---------------------------------------------------------------- trend beyond today
tv = r["trend"]
trend_color = (GREEN if "uptrend" in tv["overall"].lower() else RED if "downtrend" in tv["overall"].lower()
               else AMBER)
st.markdown(f"### Trend: {b(tv['overall'], trend_color)} <span style='font-size:1rem'>· {tv['strength']}</span>",
            unsafe_allow_html=True)
cols = st.columns(4)
for col, (horizon, state, ret, per, reading) in zip(cols, tv["rows"]):
    col.metric(horizon, state, f"{ret:+.1f}% in {per}" if ret == ret else None, help=reading)
for sd, h in r["history"].items():
    st.caption(f"Last {sd} signal: {swing.history_str(sd, h)}")

if style == "swing" and side:
    p = r["plan"]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Entry", f"Rs {num(p['entry'])}")
    t2.metric("Stop-loss", f"Rs {num(p['stop'])}", f"-{p['risk_pct']:.1f}% risk", delta_color="off")
    t3.metric("Target 1", f"Rs {num(p['t1'])}", f"+{p['t1_pct']:.1f}%")
    t4.metric("Target 2", f"Rs {num(p['t2'])}", f"+{p['t2_pct']:.1f}%")
    if side == "SELL":
        st.caption(swing.SELL_NOTE)
elif style != "swing" and hz[style]["plan"]:
    h, p = hz[style], hz[style]["plan"]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Buy at" if h["verdict"] == "BUY" else "Buy on dip at", f"Rs {num(p['entry'])}")
    t2.metric("Stop-loss (weekly close)", f"Rs {num(p['stop'])}", f"-{p['risk_pct']:.1f}% risk", delta_color="off")
    t3.metric(f"Target 1 (~{h['names'][0]})", f"Rs {num(p['t1'])}", f"{p['t1_pct']:+.1f}%")
    t4.metric(f"Target 2 (~{h['names'][1]})", f"Rs {num(p['t2'])}", f"{p['t2_pct']:+.1f}%")
if style != "swing":
    st.caption(horizons.NOTE)

# ---------------------------------------------------------------- chart
bars, spans = {"swing": (180, (20, 50)), "positional": (500, (50, 150)), "longterm": (1250, (50, 200))}[style]
d = df.iloc[-bars:]
chart = pd.DataFrame({"Close": d["Close"],
                      **{f"EMA {n}": swing.ema(df["Close"], n).iloc[-bars:] for n in spans}})
st.line_chart(chart, height=320)
if style != "swing":
    st.caption("EMA 50 / 150 days ~ 10 / 30 weeks (positional); EMA 200 days ~ 40 weeks (long-term).")

# ---------------------------------------------------------------- business, macro, portfolio
st.markdown("### 🏢 Business & fundamentals")
if fund:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Business grade", f"{fund['grade']} · {fund['grade_word']}" if fund["grade"] else "n/a",
              f"{fund['score']}/100" if fund["grade"] else None, delta_color="off")
    m2.metric("Valuation", fund["valuation"], f"P/E {num(fund['pe'], 1)}" if fund["pe"] == fund["pe"] else None,
              delta_color="off")
    m3.metric("ROE", f"{num(fund['roe'], 1)}%")
    m4.metric("Sales growth (3y)", f"{fund['rev_cagr']:+.1f}%/yr" if fund["rev_cagr"] == fund["rev_cagr"] else "–")
    m5.metric("Profit growth (3y)", f"{fund['ni_cagr']:+.1f}%/yr" if fund["ni_cagr"] == fund["ni_cagr"] else "–")
    st.write(fund["summary"])
    with st.expander("Business checklist, key numbers and 5-year history"):
        st.dataframe(pd.DataFrame(fund["checks"], columns=["Area", "Check", "Reading", "Status"]), hide_index=True,
                     width="stretch")
        st.dataframe(pd.DataFrame(fund["metrics"], columns=["Metric", "Value"]), hide_index=True, width="stretch")
        if fund["history"]:
            st.dataframe(pd.DataFrame(fund["history"], columns=["Year", "Sales (Rs cr)", "Net profit (Rs cr)",
                                                                "Net margin", "EPS (Rs)", "ROE"]),
                         hide_index=True, width="stretch")
        st.caption("From Yahoo Finance statements. Management, competition and news are not assessed.")
else:
    st.info("Financial statements are not available for this stock, so the final call uses the chart and macro only.")

st.markdown("### 🌏 Macro backdrop")
if mac:
    m1, m2, m3 = st.columns(3)
    m1.metric("Market regime", mac["regime"], f"size x{mac['size_factor']:g}", delta_color="off")
    m2.metric("For this stock", mac["tilt"])
    m3.metric("Sector index", mac["sector"] or "–")
    st.write(mac["advice"])
    for t in mac["impacts"]:
        st.caption("• " + t)
    with st.expander("Macro factors (Nifty trend, VIX, rupee, US yields, crude, sector)"):
        st.dataframe(pd.DataFrame(mac["rows"], columns=["Factor", "Reading", "Signal", "Meaning"]), hide_index=True,
                     width="stretch")
else:
    st.info("Macro data is not available right now.")

if stg:
    st.markdown("### 💼 Portfolio strategy")
    capital = st.number_input("Your total capital for shares (Rs)", min_value=10000, value=500000, step=50000,
                              key="capital")
    prow = []
    for fd in stg["styles"].values():
        price = fd["entry"] or r["last"]
        amt = capital * fd["size_pct"] / 100
        prow.append([f"{fd['label']} ({fd['period']})", fd["final"] + (" (starter)" if fd["starter"] else ""),
                     f"{fd['size_pct']:g}%" if fd["size_pct"] else "–",
                     f"Rs {num(amt, 0)}" if amt else "–", str(math.floor(amt / price)) if amt and price else "–",
                     f"Rs {num(price)}", fd["how"]])
    st.dataframe(pd.DataFrame(prow, columns=["Style", "Final call", "Size", "Amount", "Shares", "At price",
                                             "How"]), hide_index=True, width="stretch")
    for w in stg["fit"]["warnings"]:
        st.warning(w)
    with st.expander("Portfolio fit and how to split your money"):
        st.dataframe(pd.DataFrame(stg["fit"]["rows"] + stg["guide"], columns=["Item", "Detail"]), hide_index=True,
                     width="stretch")

# ---------------------------------------------------------------- detail tables
HIDE = ("Investment advice", "Verdict", "Trend", "Outlook", "Holding period", "Positional (", "Long-term (", "Final verdict",
        "Business & fundamentals", "Key numbers", "Financial history", "Macro:", "Portfolio strategy")
shown = [s for s in rep.sections if not s.title.startswith(HIDE)]
tabs = st.tabs([s.title for s in shown])
for tab, sec in zip(tabs, shown):
    with tab:
        st.dataframe(section_df(sec), hide_index=True, width="stretch")
        if sec.note:
            st.caption(sec.note)

# ---------------------------------------------------------------- chat
st.divider()
st.subheader("💬 Ask a doubt")
st.caption("Examples: *Should I invest?* · *Why is this a buy?* · *When can I buy?* · *Is it good for 1-2 years?* · "
           "*Can I hold for 3-6 months?* · *How are the fundamentals?* · *Is it expensive?* · *How is the macro?* · "
           "*How much should I invest?* · *What is ROE?* · *How many shares should I buy?*")
ai_on = bool(os.environ.get("ANTHROPIC_API_KEY"))
if not ai_on:
    st.caption("Built-in explanations are used. For free-form answers set `ANTHROPIC_API_KEY` and `pip install anthropic`.")

key = f"chat_{symbol}"
st.session_state.setdefault(key, [])
for m in st.session_state[key]:
    st.chat_message(m["role"]).markdown(m["content"])

if q := st.chat_input(f"Ask anything about {sym} or trading"):
    st.chat_message("user").markdown(q)
    with st.spinner("Thinking..."):
        reply = ai.ask(q, rpt.to_text(rep), st.session_state[key], cfg["ai"]["model"]) if ai_on else None
        reply = reply or explain.answer(q, r, sym) or (
            "I can explain: the signal and *why*, the trade plan, *timeline*, stop-loss, targets, position size, "
            "RSI, MACD, EMA, ADX, Supertrend, ATR, OBV, Stochastic, Bollinger Bands, support/resistance, pivots, "
            "breakouts, candles, volume, backtest, confidence, and shorting. Try one of those words.")
    st.chat_message("assistant").markdown(reply)
    st.session_state[key] += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]

st.divider()
st.caption(rpt.DISCLAIMER)
