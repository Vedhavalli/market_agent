# Market Agent

Runs on your own machine. Every day it fetches Indian market data, computes technicals, scans your watchlist,
checks your portfolio against your stop-loss / target levels, and saves an HTML report.

## Setup (Python 3.9+)
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python agent.py report                                  # first run

## Portfolio
    python agent.py add SBIN 10 850 --stop 780 --target 1000   # buy (repeat buys average the price)
    python agent.py sell SBIN 5 940                            # logs realised P&L to data/realized.csv
    python agent.py set SBIN --stop 800 --target 1050
    python agent.py holdings                                   # live value, P&L, allocation, alerts
Holdings live in `data/portfolio.csv`; you can edit it in Excel too. Use `XYZ.BO` for BSE-only symbols.

## Daily report
    python agent.py report              # console + reports/report-YYYY-MM-DD.html
    python agent.py report --ai         # adds Claude commentary (pip install anthropic; set ANTHROPIC_API_KEY)
    python agent.py levels TCS          # technicals for one stock
    python agent.py daemon              # leave running; writes the report each weekday at report_time (IST)

The daemon is timezone-independent (it uses `report_time` in `config.toml`, default 16:00 IST) and skips
market holidays by checking that today's session exists. It also writes the swing report (below).

## Web app (simple UI)

### Start the application
First time only (Windows PowerShell, from the project folder):

    cd "C:\AI Learing\market_agent\market_agent"
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Every time:

    cd "C:\AI Learing\market_agent\market_agent"
    .\.venv\Scripts\python.exe -m streamlit run app.py

The browser opens at **http://localhost:8501** (open it manually if it doesn't). Stop the app with `Ctrl+C` in the terminal.

- Use the `.venv` Python as shown. A plain `python` / `streamlit` may be a different Python without yfinance installed.
- `No module named streamlit`: you are in the wrong folder (e.g. `C:\AI Learing\market_agent`, which has its own
  `.venv`). `cd` into `C:\AI Learing\market_agent\market_agent` first.
- macOS / Linux: `source .venv/bin/activate && streamlit run app.py`.
- Different port: add `--server.port 8502`.
- Optional AI chat: `.\.venv\Scripts\python.exe -m pip install anthropic`, then `$env:ANTHROPIC_API_KEY="sk-..."`
  in the same terminal before starting.
- Data is cached for 15 minutes. Press `R` in the browser (or "Rerun") to refresh after that.
- If you see `CertificateVerifyError` cookie warnings (corporate proxy), ignore them; data still downloads.

### Using it
Start typing a company name or NSE symbol in the search box. After **3 letters** (e.g. *rel*, *tat*, *hdf*) a list of
matching companies drops down; **selecting one starts the analysis**. Suggestions come from the full NSE equity list
(downloaded from NSE and cached in `data/nse_equity.csv` for a week; Yahoo search is the fallback).

A big coloured banner gives the verdict:
- **Green: GOOD TO BUY.** Buy price, **how long to hold** (trading days and date), **what price / profit % to sell at**
  (Target 1 and Target 2), and the stop-loss.
- **Amber: NOT A BUY YET (WAIT).** Why not today, what to wait for (breakout level, pullback zone, cooling RSI,
  missing conditions with exact prices) and **when** it may happen (rough estimate in trading sessions and a date).
- **Red: NOT GOOD TO BUY (SELL / AVOID).** The downtrend levels, the exit trigger for holders, and what must change
  (with prices and an estimated time) before it can turn into a buy.

The verdict is not based on one day: a **Trend** strip shows the short (2-4 weeks), medium (1-3 months), long
(6-12 months) and weekly trend with returns, the overall trend and its strength (ADX), plus when the last BUY / SELL
signal happened and how the price moved since.

**Holding period.** Choose *How long do you want to hold?* and the banner, targets and chart switch to that style.
A table above it always shows all three side by side:

| Style | Hold | Chart used | Conditions |
|---|---|---|---|
| Swing | days to weeks | daily | 9 daily trend/momentum/volume checks + a fresh trigger |
| Positional | 3-6 months | weekly | 10/30-week EMAs and slope, weekly MACD + RSI, 6-month return vs Nifty, near 52-week high (7) |
| Long-term | 1-2 years | primary trend | 40-week EMA (~200-DMA) and slope, 50/200-DMA golden cross, 12-month return and vs Nifty, 52-week range, higher yearly lows (8) |

Positional / long-term verdicts are **BUY**, **BUY ON DIP** (good trend but stretched: shows a buy zone), **WAIT** or
**AVOID**. A BUY shows the buy price, Target 1 / Target 2 with profit % and an approximate month (~3 / 6 months or
~1 / 2 years), and a stop-loss checked on the weekly (Friday) close. Targets are the typical highest price the stock
reached after the same setup in its own past (10 years of data), with the past win rate shown. If it is not a buy,
you get what is missing (with price levels) and how long this stock usually took to turn into a BUY from a similar
state, plus a line for people who already hold it. Scalping / intraday is not covered (it needs live tick data).

**Investment advice.** At the top, an *Investment advice* box gives one recommendation for the company you picked
(INVEST, START SMALL, BUY ON DIPS, BUY FOR 3-6 MONTHS, SHORT-TERM TRADE ONLY, WAIT or AVOID), followed by:
- what to do for each holding period, with size, buy price, targets and stop-loss
- a business analysis covering growth, profitability, financial health, recent results and valuation
- other factors: price trend, market regime, sector and macro sensitivity, and portfolio fit
- points in favour, risks, and what would change the view

Ask the chat *Should I invest?* or *I need investment advice* for the same summary.

**Stock screener (second tab).** Pick a list (Nifty 50, Nifty Next 50, Nifty 100, Nifty Midcap 50, Nifty 500 or your
`[watchlist]`) and a holding period, then press *Run screener*. Every stock gets the full analysis and a score out of
100 built from business quality, valuation, price trend, latest quarterly profit, the macro / sector backdrop and the
final buy call. The weights depend on the holding period: long-term leans on the business, swing leans on the chart.
Only stocks whose final call is a buy appear in the **Top 5** cards and the **Top 10** table, each with size, buy
price, target and stop-loss. *Max 2 stocks per sector* keeps the list diversified. Below that you get the next best
stocks that are not a buy yet, and the full scored list. *Open analysis* jumps to the first tab for any stock. A screen
runs in the background, so you can switch tabs or close the page and come back. It takes about 3-4 minutes for 50
stocks and about 40 minutes for the Nifty 500, and *Stop* ranks the stocks analysed so far. Recently listed companies
with less than about a year of price history are skipped and shown as "no data". Only one screen runs at a
time. Results are then kept for an hour, and index lists are cached in `data/` for a week.

**Final call: chart + business + macro + portfolio.** The chart verdict above is then checked against the company's
business, the macro backdrop and your portfolio. A *Final call* box (and the *Final call* / *Size* columns in the
table) gives the decision for the chosen holding period and how much of your capital to put in.

| Layer | What is checked | Effect |
|---|---|---|
| Business & fundamentals | Sales / profit growth (3-yr CAGR), ROE, ROCE (ROA for banks), margin trend, debt / equity, interest cover, cash-flow quality (book-value growth for lenders), latest results vs a year ago, share dilution, P/E with PEG (P/B for lenders), dividend yield, 5-year history | Grade A (strong) to D (weak) + valuation (attractive / fair / expensive / loss-making) |
| Macro | Nifty primary trend (200-DMA), India VIX, rupee and US 10-year yield over 3 months (foreign-money flows), Brent crude, the stock's sector index vs Nifty, and known sensitivities (IT / pharma gain from a weak rupee, paints / tyres / airlines lose from costly crude, banks / realty are rate-sensitive) | Regime Risk-on / Neutral / Risk-off and a tailwind / headwind for the stock |
| Portfolio strategy | Your holdings in `data/portfolio.csv`: weight already held, sector exposure, 1-year correlation with what you own | Position size cap and warnings |

Rules: swing trades follow the chart (fundamentals / macro only change the size); a positional BUY on a weak or
loss-making business becomes WAIT; a long-term BUY on a grade-D or loss-making business becomes AVOID, while a
grade A/B business at a fair price can be started with 1/3 size even when the chart says WAIT. Size (% of capital) =
the smaller of a grade cap and a fixed risk budget to the stop-loss (1% swing, 1.5% positional, 2.5% long-term),
x 1 / 0.75 / 0.5 for Risk-on / Neutral / Risk-off, smaller again for a sector headwind or low confidence, and capped
so one stock stays under 10% and one sector under 30%. Enter your capital in the *Portfolio strategy* section to see
the rupee amount and number of shares. Ask the chat *How are the fundamentals?*, *Is it expensive?*, *How is the
macro?* or *How much should I invest?*.

Fundamentals come from Yahoo's reported statements only: management quality, competition, order books, news and
analyst estimates are not assessed - read the latest results / annual report before a long-term buy.

Below that: a price chart with EMAs, every indicator, support/resistance and the backtest. The **Ask a doubt** chat
explains the signal, the plan and any indicator in plain English (built-in answers offline; free-form answers when
`ANTHROPIC_API_KEY` is set and `anthropic` is installed).

## Swing trading: BUY / SELL signals with targets, profit % and timeline
    python agent.py swing                    # scan the watchlist -> console + reports/swing-YYYY-MM-DD.html
    python agent.py swing TATAMOTORS DLF     # scan specific stocks
    python agent.py analyze RELIANCE         # full analysis: chart, business, macro, final call + size, backtest

Each run downloads fresh daily data (5 years) and recomputes everything:
- **Indicators:** EMA 20/50/200, RSI(14), MACD(12,26,9), ADX(14) with +DI/-DI, Supertrend(10,3), Stochastic(14,3,3),
  Bollinger(20,2), ATR(14), OBV, volume vs 20-day average, weekly 20-EMA trend, candle pattern, fractal support/resistance, pivots.
- **Score (0-9):** trend vs 50-EMA, 20/50 EMA alignment, price vs 200-EMA, MACD histogram, RSI zone, ADX + DI,
  Supertrend, OBV flow, weekly trend. Each is Bullish / Bearish / Neutral.
- **Signal:** BUY if bullish score >= 7 **and** a fresh trigger in the last 3 sessions (MACD cross, Supertrend flip,
  20-day breakout, 20-EMA reclaim) **and** RSI < 72 **and** price not > 2.5 ATR above the 20-EMA. SELL is the mirror image.
- **Trade plan:** entry (close), stop-loss below the 10-day swing low (1-2.5 ATR), Target 1 = 1.5R, Target 2 = 3R,
  profit % for each, and an estimated timeline in trading days (from ATR and the stock's own past signals).
- **Confidence:** High / Medium / Low from the score, the Nifty trend and the stock's backtest win rate.
- **Backtest:** the same rules replayed on the stock's history (entry next open, exit at stop / T1 / 20 sessions).
- **Setups forming:** stocks one step away from a signal, with the exact trigger level to watch.

SELL means exit / avoid longs; overnight shorting in the cash segment is not allowed in India (use F&O).
Tune thresholds in the `[swing]` section of `config.toml`.

## What the report contains
1. Market snapshot (Nifty, Sensex, Bank Nifty, VIX) 2. Sector ranking 3. Pivot / support / resistance, DMAs, RSI, MACD
4. Watchlist gainers & losers 5. Buy-watch and sell-watch screens 6. Portfolio P&L and allocation 7. Alerts
(stop-loss, target, near-stop, weak technicals, overbought, big move, over-concentration).

## Limits (read these)
- Data is Yahoo Finance via `yfinance`: free, unofficial, ~15 min delayed, occasionally missing. Check symbols that show "no data".
- Some sector-index tickers in `config.toml` may be unavailable on Yahoo; remove or fix any that show no data.
- FII/DII flows and news are not included (no reliable free API). Add them if you have a source.
- The buy/sell "watch" lists are a simple transparent score (see the note under table 5a). They are screens to research, not advice.
- P&L ignores brokerage, STT, charges and tax.

## Tests
    python tests/test_offline.py        # synthetic data, no internet needed
