#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          EDGE CALLS — Daily Pick Generator                       ║
║  Runs automatically via GitHub Actions every weekday 8:30 AM ET  ║
║  Scans watchlist · runs TA · fetches real options · updates site  ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO CUSTOMIZE:
  · Add/remove tickers in WATCHLIST below
  · Change NUM_PICKS to show more or fewer picks
  · Adjust RISK_THRESHOLDS to tune scoring sensitivity
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta
import pytz

# ─────────────────────────────────────────────────────────────────
#  CONFIG  —  edit this section to customize
# ─────────────────────────────────────────────────────────────────

WATCHLIST = [
    "NVDA", "TSLA", "AAPL", "AMZN", "META",
    "AMD",  "MSFT", "GOOGL", "SPY",  "QQQ",
    "SOFI", "PLTR", "COIN",  "MSTR", "SMCI",
    "ARM",  "HOOD", "RIVN",  "UBER", "SHOP",
    "NFLX", "DIS",  "BAC",   "JPM",  "GS",
]

NUM_PICKS        = 4    # how many picks to surface each day
RISK_HIGH_SCORE  = 7    # score >= this → LOW risk
RISK_MED_SCORE   = 4    # score >= this → MEDIUM risk (else HIGH)

# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

ET = pytz.timezone("America/New_York")

def next_options_expiry():
    """Return the next Friday (or this Friday if today < Friday market close)."""
    today = datetime.now(ET)
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def squeeze(series):
    """Ensure we have a 1-D Series even from multi-level yfinance columns."""
    if isinstance(series, pd.DataFrame):
        return series.iloc[:, 0]
    return series

# ─────────────────────────────────────────────────────────────────
#  STEP 1 — Technical Analysis Scoring
# ─────────────────────────────────────────────────────────────────

def analyze_ticker(symbol):
    try:
        raw = yf.download(symbol, period="90d", interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 30:
            return None

        close  = squeeze(raw["Close"])
        volume = squeeze(raw["Volume"])

        # ── Indicators ──────────────────────────────────────────
        ema9   = close.ewm(span=9,  adjust=False).mean()
        ema21  = close.ewm(span=21, adjust=False).mean()
        ema50  = close.ewm(span=50, adjust=False).mean()
        rsi    = calc_rsi(close)

        exp12  = close.ewm(span=12, adjust=False).mean()
        exp26  = close.ewm(span=26, adjust=False).mean()
        macd   = exp12 - exp26
        sig    = macd.ewm(span=9,  adjust=False).mean()

        vol_avg   = volume.rolling(20).mean()
        vol_ratio = float((volume / vol_avg).iloc[-1])

        # ── Last two candles ────────────────────────────────────
        lc   = float(close.iloc[-1])
        pc   = float(close.iloc[-2])
        lrsi = float(rsi.iloc[-1])
        lmac = float(macd.iloc[-1]);  lsig = float(sig.iloc[-1])
        pmac = float(macd.iloc[-2]);  psig = float(sig.iloc[-2])
        le21 = float(ema21.iloc[-1]); le50 = float(ema50.iloc[-1])

        # ── Score ───────────────────────────────────────────────
        score     = 0
        direction = "CALL"
        signals   = []

        # Trend
        if lc > le21:
            score += 2
            signals.append("price above 21-EMA")
        else:
            score -= 2
            direction = "PUT"
            signals.append("price below 21-EMA")

        if lc > le50:
            score += 1
        else:
            score -= 1

        # MACD
        if lmac > lsig and pmac <= psig:
            score += 4
            signals.append("fresh MACD bullish crossover")
        elif lmac < lsig and pmac >= psig:
            score -= 4
            direction = "PUT"
            signals.append("MACD bearish crossover")
        elif lmac > lsig:
            score += 1
        else:
            score -= 1

        # RSI
        if 40 < lrsi < 65:
            score += 2
            signals.append(f"RSI {lrsi:.0f} — healthy momentum")
        elif lrsi <= 35:
            score += 3
            signals.append(f"RSI {lrsi:.0f} — oversold bounce setup")
        elif lrsi >= 75:
            score -= 2
            signals.append(f"RSI {lrsi:.0f} — overbought, caution")

        # Volume spike
        if vol_ratio >= 1.8:
            score += 3
            signals.append(f"{vol_ratio:.1f}x average volume — strong interest")
        elif vol_ratio >= 1.3:
            score += 1
            signals.append(f"{vol_ratio:.1f}x average volume")

        # Intraday momentum alignment
        pct = (lc - pc) / pc * 100
        if direction == "CALL" and pct > 0:
            score += 1
        elif direction == "PUT" and pct < 0:
            score += 1

        # ── Risk tier ───────────────────────────────────────────
        if score >= RISK_HIGH_SCORE:
            risk = "low"
        elif score >= RISK_MED_SCORE:
            risk = "medium"
        else:
            risk = "high"

        return {
            "symbol":    symbol,
            "score":     score,
            "direction": direction,
            "price":     lc,
            "rsi":       lrsi,
            "signals":   signals[:3],
            "risk":      risk,
            "vol_ratio": vol_ratio,
            "pct":       pct,
        }

    except Exception as e:
        print(f"  ⚠  {symbol}: {e}")
        return None

# ─────────────────────────────────────────────────────────────────
#  STEP 2 — Real Options Chain Data
# ─────────────────────────────────────────────────────────────────

def fetch_options(symbol, direction, price, target_expiry):
    try:
        t    = yf.Ticker(symbol)
        exps = t.options
        if not exps:
            return None

        # Find nearest expiry on or after target Friday
        td = datetime.strptime(target_expiry, "%Y-%m-%d").date()
        chosen = next((e for e in exps
                       if datetime.strptime(e, "%Y-%m-%d").date() >= td), exps[0])

        chain = t.option_chain(chosen)
        opts  = chain.calls if direction == "CALL" else chain.puts
        if opts.empty:
            return None

        # Slightly OTM filter
        if direction == "CALL":
            pool = opts[opts["strike"] >= price * 1.01].head(5)
        else:
            pool = opts[opts["strike"] <= price * 0.99].tail(5)

        if pool.empty:
            pool = opts.iloc[:5] if direction == "CALL" else opts.iloc[-5:]

        # Best = highest open interest with a valid ask
        pool = pool[pool["ask"] > 0].copy()
        if pool.empty:
            return None

        row     = pool.sort_values("openInterest", ascending=False).iloc[0]
        strike  = float(row["strike"])
        premium = float(row["ask"])

        exp_obj = datetime.strptime(chosen, "%Y-%m-%d")
        ctype   = "C" if direction == "CALL" else "P"
        contract_name = (
            f"{symbol} {exp_obj.strftime('%m/%d')} "
            f"${int(strike)}{ctype}"
        )

        return {
            "contract": contract_name,
            "strike":   f"${strike:,.2f}",
            "expiry":   exp_obj.strftime("%b %d, %Y"),
            "premium":  f"${premium:.2f}",
        }

    except Exception as e:
        print(f"  ⚠  Options for {symbol}: {e}")
        return None

# ─────────────────────────────────────────────────────────────────
#  STEP 3 — Assemble Pick Object
# ─────────────────────────────────────────────────────────────────

def build_pick(analysis, exp_date):
    sym   = analysis["symbol"]
    price = analysis["price"]
    dir   = analysis["direction"]
    risk  = analysis["risk"]
    sigs  = analysis["signals"]

    # Price levels
    spread   = price * 0.004
    entry_lo = price - spread
    entry_hi = price + spread
    target   = price * (1.05  if dir == "CALL" else 0.95)
    stop     = price * (0.975 if dir == "CALL" else 1.025)

    # Rationale
    sig_text   = ", ".join(sigs) if sigs else "multi-indicator confluence"
    action_str = "bullish breakout above entry zone" if dir == "CALL" else "bearish breakdown through entry zone"
    rationale  = (
        f"{'Bullish' if dir == 'CALL' else 'Bearish'} setup driven by {sig_text}. "
        f"Look for {action_str} with above-average volume confirmation. "
        f"Manage size according to your account risk rules."
    )

    # Options — real data first, fallback estimate
    opts = fetch_options(sym, dir, price, exp_date)
    if opts:
        contract = opts["contract"]
        strike   = opts["strike"]
        expiry   = opts["expiry"]
        premium  = opts["premium"]
    else:
        otm_mult = 1.03 if dir == "CALL" else 0.97
        sv       = round(price * otm_mult / 5) * 5
        exp_obj  = datetime.strptime(exp_date, "%Y-%m-%d")
        ctype    = "C" if dir == "CALL" else "P"
        contract = f"{sym} {exp_obj.strftime('%m/%d')} ${int(sv)}{ctype}"
        strike   = f"${sv:.2f}"
        expiry   = exp_obj.strftime("%b %d, %Y")
        premium  = f"${price * 0.015:.2f}"

    return {
        "ticker":    sym,
        "direction": dir,
        "risk":      risk,
        "entry":     f"${entry_lo:,.2f} – ${entry_hi:,.2f}",
        "target":    f"${target:,.2f}",
        "stop":      f"${stop:,.2f}",
        "contract":  contract,
        "strike":    strike,
        "expiry":    expiry,
        "premium":   premium,
        "rationale": rationale,
        "chartUrl":  "",
    }

# ─────────────────────────────────────────────────────────────────
#  STEP 4 — Inject Picks into index.html
# ─────────────────────────────────────────────────────────────────

def picks_to_js(picks):
    lines = ["const PICKS = ["]
    for p in picks:
        lines.append("  {")
        for k, v in p.items():
            lines.append(f"    {k}: {json.dumps(v)},")
        lines.append("  },")
    lines.append("];")
    return "\n".join(lines)

def update_html(picks):
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()

    new_picks_js = picks_to_js(picks)
    html = re.sub(r"const PICKS = \[[\s\S]*?\];", new_picks_js, html)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    today = datetime.now(ET).strftime("%A, %B %d, %Y")
    print(f"\n✅  index.html updated — {len(picks)} picks for {today}")

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EDGE CALLS — Daily Pick Generator")
    print(f"  {datetime.now(ET).strftime('%A, %B %d, %Y  %I:%M %p ET')}")
    print("=" * 60)
    print(f"\nScanning {len(WATCHLIST)} tickers...\n")

    results = []
    for sym in WATCHLIST:
        print(f"  → {sym}", end="  ", flush=True)
        r = analyze_ticker(sym)
        if r:
            print(f"score={r['score']:+d}  dir={r['direction']}  rsi={r['rsi']:.0f}")
            results.append(r)
        else:
            print("skipped")

    # Sort by score, take top N
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:NUM_PICKS]

    print(f"\n🏆  Top {NUM_PICKS} picks: {[r['symbol'] for r in top]}")
    print("\nFetching live options chain data...\n")

    exp_date = next_options_expiry()
    picks    = [build_pich(r, exp_date) for r in top]

    for p in picks:
        print(f"  {p['ticker']:6s} {p['direction']}  {p['contract']}  premium={p['premium']}  risk={p['risk']}")

    update_html(picks)
    print("\n🚀  Done — site ready for market open!\n")

if __name__ == "__main__":
    main()
