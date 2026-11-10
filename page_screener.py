"""Stock screener tab: runs the full analysis on every stock of an index and lists the top 5 / top 10 to buy."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import agent
import data
import screener
from util import num

cfg = agent.load_config(None)
FRESH_FOR = 3600  # seconds a screen result is shown without asking for a refresh
CLEAR_COOLDOWN = 10 * 60  # seconds between cache clears (protects the shared Yahoo quota)


@st.cache_resource
def store() -> dict:
    """Shared by all browser sessions so one screen serves everyone: finished results
    {(universe, code_version): result}, the running job and when the cache was last cleared."""
    return {"results": {}, "job": None, "cleared": 0.0}


def start(key: tuple, universe: str, stocks: list[tuple[str, str]]) -> None:
    """Run the screen in a background thread so it survives reruns, tab switches and closed browser tabs.
    Reuses data.py's cache: prices younger than 15 min and statements younger than a day are not re-downloaded."""
    job = {"key": key, "universe": universe, "done": 0, "total": len(stocks), "last": "", "t0": time.time(),
           "stop": threading.Event(), "error": None}
    s = store()

    def work() -> None:
        try:
            res = screener.run(cfg, stocks, lambda d, t, sym: job.update(done=d, last=sym), stop=job["stop"].is_set)
            if res["rows"]:
                s["results"][key] = {**res, "time": time.time(), "secs": time.time() - job["t0"]}
        except Exception as e:  # shown on the page
            job["error"] = f"{type(e).__name__}: {e}"
        finally:
            job["finished"] = True

    s["job"] = job
    threading.Thread(target=work, daemon=True, name="screener").start()


@st.fragment(run_every=2)
def progress_box(job: dict) -> None:
    if job.get("finished"):
        st.rerun()  # reload the whole page to show the results
    el = time.time() - job["t0"]
    left = el / job["done"] * (job["total"] - job["done"]) if job["done"] else None
    st.progress(job["done"] / job["total"],
                text=f"Screening {job['universe']}: {job['done']}/{job['total']} stocks analysed"
                     + (f" (last: {job['last']})" if job["last"] else "") + f" · {el / 60:.1f} min"
                     + (f" · about {left / 60:.0f} min left" if left is not None else ""))
    if job["stop"].is_set():
        st.caption("Stopping: finishing the stocks already in progress...")
    elif st.button("⏹️ Stop and show results so far"):
        job["stop"].set()
    st.caption("Runs in the background: you can switch to the Stock analysis tab or close this page and come back.")


@st.cache_data(ttl=86400, show_spinner=False)
def stocks_in(universe: str) -> list[tuple[str, str]]:
    return screener.members(cfg, universe)


def frame(entries: list[dict], style: str, rank: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(screener.table(entries, style), columns=screener.COLUMNS)
    return df if rank else df.drop(columns="Rank")


def show(df: pd.DataFrame) -> None:
    st.dataframe(df, alt="Value-investing stock rankings with valuation, timing, and risk details",
                 hide_index=True, width="stretch", column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Why": st.column_config.TextColumn("Why", width="large")})


st.title("🔎 Stock screener")
st.caption("Ranks companies by business quality, conservative intrinsic value and margin of safety first; weekly "
           "price action is used mainly to time entries. Statements and analyst snapshots come from Yahoo only; "
           "confirm filings independently. Estimates are not guarantees.")

c1, c2, c3 = st.columns([2, 4, 2])
universe = c1.selectbox("Stocks to screen", screener.UNIVERSES, index=4, key="sc_universe")
style = c2.segmented_control("Holding period", list(screener.STYLES), default="longterm",
                             format_func=screener.STYLES.get, key="sc_style") or "longterm"
diversify = c3.checkbox("Max 2 stocks per sector", value=True, key="sc_div",
                        help="Spread the top list across sectors instead of, say, five banks.")

stocks = stocks_in(universe)
if not stocks:
    st.error(f"Could not download the {universe} list. Check the internet connection or pick 'My watchlist'.")
    st.stop()

key = (universe, max(p.stat().st_mtime for p in Path(__file__).parent.glob("*.py")))
res = store()["results"].get(key)
job = store()["job"]
running = job is not None and not job.get("finished")
if job and job.get("finished") and job["error"]:
    st.error(f"The last screen of {job['universe']} failed: {job['error']}")
if running:
    progress_box(job)
    if job["universe"] != universe:
        st.info(f"Only one screen runs at a time: wait for {job['universe']} to finish (or stop it) before "
                f"screening {universe}.")
age = time.time() - res["time"] if res else None
label = (f"▶️ Run screener on {len(stocks)} stocks" if not res else
         "🔄 Refresh" + (" (results are over an hour old)" if age > FRESH_FOR else ""))
if not running:
    b1, b2, _ = st.columns([3, 2, 5])
    if b1.button(label, type="primary" if not res or age > FRESH_FOR else "secondary",
                 help="Re-uses prices fetched in the last 15 min and statements fetched in the last day."):
        start(key, universe, stocks)
        st.rerun()
    wait = CLEAR_COOLDOWN - (time.time() - store()["cleared"])
    if res and b2.button("🗑️ Clear cache", disabled=wait > 0,
                         help=(f"Available again in {wait / 60:.0f} min." if wait > 0 else
                               f"Discard the stored {universe} results and cached prices (for everyone) so the next "
                               "run fetches live prices. Company statements stay cached for a day.")):
        for k in [k for k in store()["results"] if k[0] == universe]:
            store()["results"].pop(k, None)
        data.clear(prices_only=True)
        store()["cleared"] = time.time()
        st.rerun()

if not res:
    if not running:
        st.info(f"Press **Run screener**. Each stock gets the full analysis, so it takes about "
                f"{max(1, len(stocks) * 3 // 60)}-{max(2, len(stocks) * 6 // 60)} minutes for {len(stocks)} stocks "
                "(it runs in the background; you can stop it early and see the results so far). Results are then "
                "shared with everyone using the app until refreshed.")
    st.stop()

rk = screener.rank(res["rows"], style, n=10, max_per_sector=2 if diversify else None)
m = res["market"]
period = screener.STYLES[style]
st.caption(f"Prices of {res['date']} · {len(res['rows'])} of {len(stocks)} stocks analysed in {res['secs']:.0f}s · "
           f"run at {time.strftime('%d %b %H:%M', time.localtime(res['time']))}"
           + (" · stopped early: only the stocks analysed so far are ranked" if res.get("stopped") else "")
           + (f" · no data ({len(res['failed'])}): {', '.join(res['failed'][:15])}"
              + (" ..." if len(res["failed"]) > 15 else "") if res["failed"] else ""))
if m:
    (st.success if m["regime"] == "Risk-on" else st.warning if m["regime"] == "Neutral" else st.error)(
        f"**Market: {m['regime']}** (Nifty {m['trend']}). {m['advice']}")

top = rk["buy"]
st.markdown(f"## 🏆 Top 5 to buy · {period}")
if not top:
    st.warning("No stock passes every check for this holding period today (chart, business, valuation and macro "
               "together). The best candidates to watch are listed below; try another holding period or list.")
for col, x in zip(st.columns(5), top[:5]):
    s = x["styles"][style]
    entry = s["entry"] or x["price"]
    with col.container(border=True):
        st.markdown(f"#### #{x['rank']} {x['sym']}")
        st.caption(f"{x['name']} · {x['sector'] or 'sector n/a'}")
        st.metric("Score", f"{x['score']:.0f}/100", f"Rs {num(x['price'])} ({x['chg']:+.1f}%)", delta_color="off")
        st.markdown(f"**{screener.call_str(s)}** · size {s['size']:g}%  \n"
                    + (f"Buy at Rs {num(entry)}" if s["final"] != "BUY ON DIP" else f"Buy on dip to Rs {num(entry)}")
                    + (f"  \nTarget Rs {num(s['t1'])} ({s['t1_pct']:+.1f}%)" if s["t1"] else "")
                    + (f"  \nStop-loss Rs {num(s['stop'])}" if s["stop"] else ""))
        st.caption(screener.reasons(x, style))
if top:
    st.markdown(f"## 📋 Top {len(top)} to buy · {period}")
    show(frame(top, style))
if len(top) < 10:
    st.caption(f"Only {len(top)} stock(s) are a buy for this holding period right now"
               + (" (after the 2-per-sector limit)" if diversify else "") + ".")
if rk["watch"]:
    st.markdown("### 👀 Next best")
    st.caption("Highest scores outside the top list: buys left out by the 2-per-sector limit, or stocks whose final "
               "call is not a buy yet (wait for it to turn into a buy before acting). See the *Final call* column.")
    show(frame(rk["watch"], style, rank=False))

with st.expander("How the score works"):
    w = screener.WEIGHTS[style]
    st.markdown(f"Score = weighted average of six factor scores (each 0-100). Weights for **{period}**:")
    st.dataframe(pd.DataFrame([w]), hide_index=True)
    st.markdown(
        "- **Business**: fundamentals score from growth, ROE / ROCE (or lender metrics), leverage, cash flow and results.\n"
        "- **Valuation**: margin of safety against the median of available Graham, earnings, book-value and DCF estimates.\n"
        "- **Timing**: share of weekly chart checks that are bullish; it has little weight for long-term investing.\n"
        "- **Results**: latest quarter's profit growth vs a year ago (+20% or better = 100, -20% or worse = 0).\n"
        "- **Macro**: sector / macro tailwind 100, neutral 50, headwind 0.\n"
        "- **Final call**: BUY 100, BUY ON DIP 75, START SMALL 60, WAIT 25, AVOID 0.\n\n"
        "Only stocks whose final call is a buy (with a position size above 0) are in the top list. Position size "
        "already reflects the market regime, sector headwinds and your portfolio (portfolio.csv).")

with st.expander(f"All {len(rk['all'])} analysed stocks, best score first"):
    show(frame(rk["all"], style, rank=False))

opts = {f"{x['name']} ({x['sym']})": x for x in rk["all"]}
o1, o2 = st.columns([4, 1])
pick = o1.selectbox("Open the full analysis of", list(opts), index=None, placeholder="Choose a stock...")
if o2.button("Open analysis", disabled=pick is None):
    x = opts[pick]
    st.session_state["screener_pick"] = f"{data.yf_symbol(x['sym'])}|{x['name']}"
    st.session_state.pop("share_search", None)
    st.switch_page("page_analysis.py")

st.caption("Rule-based screen of public data (Yahoo Finance, NSE). Not personalised financial advice: check the "
           "company yourself and size positions to your own risk.")
