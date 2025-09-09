"""
mo_dynamic_allocation.py: Dynamic Asset-Allocation Signal (simple regime detector 
    → new weights)

What this does:
Uses SPY (equities) and TLT (long bonds) total-return proxies to estimate a 
risk-on / risk-off regime from rolling volatility & trend, then outputs recommended 
weights (e.g., 70/30 vs 40/60) plus a rationale string you can send to clients.
"""
# mo_dynamic_allocation.py
import pandas as pd
import numpy as np
import yfinance as yf

tickers = ["SPY","TLT"]
data = yf.download(tickers, period="3y", interval="1d", auto_adjust=True)["Close"].dropna()

# Total-return proxy: simple pct change cumprod (for illustration)
ret = data.pct_change().dropna()
vol_60 = ret.rolling(60).std() * np.sqrt(252)
mom_120 = (data / data.shift(120) - 1.0)  # 6m momentum proxy

# Simple regime: risk-on if SPY momentum > 0 and vol below its 60d median
risk_on = (mom_120["SPY"] > 0) & (vol_60["SPY"] < vol_60["SPY"].median())

if risk_on.iloc[-1]:
    weights = {"SPY": 0.70, "TLT": 0.30}
    rationale = "Risk-on: positive equity momentum and contained volatility."
else:
    weights = {"SPY": 0.40, "TLT": 0.60}
    rationale = "Risk-off: weak equity momentum and/or elevated volatility."

print("Suggested strategic weights:", weights)
print("Rationale:", rationale)

# Save a tiny report for the FA
pd.Series(weights, name="weight").to_frame().to_csv("out/dynamic_allocation_weights.csv")
print("Saved: out/dynamic_allocation_weights.csv")
