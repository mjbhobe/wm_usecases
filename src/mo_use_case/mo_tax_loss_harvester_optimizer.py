"""
mo_tax_loss_harvester_optimizer.py: Tax-Loss Harvesting (TLH) Optimizer 
    (wash-sale aware, ETF proxies)

What this does:
Takes client tax lots (CSV), pulls current prices with yfinance, computes 
unrealized losses, and proposes swap candidates (e.g., SPY ↔ VOO/IVV; QQQ ↔ SCHG/VGT) 
to realize losses while keeping exposure. Avoids wash-sale windows for the 
same/similar lot-IDs and flags timing.
"""
# mo_tlh_optimizer.py
import datetime as dt
import pandas as pd
import yfinance as yf

# ---- Input: tax lots ----
# Columns: account_id, ticker, lot_id, trade_date, quantity, cost_basis
lots = pd.read_csv("data/tax_lots.csv", parse_dates=["trade_date"])
lots["ticker"] = lots["ticker"].str.upper()

# ---- Current prices ----
tickers = lots["ticker"].unique().tolist()
px = yf.download(tickers, period="6mo", interval="1d", auto_adjust=True)["Close"]
if isinstance(px, pd.Series): px = px.to_frame()
last = px.iloc[-1]

lots = lots.join(last.rename("last"), on="ticker")
lots["unrl_pl_%"] = (lots["last"]/lots["cost_basis"] - 1) * 100
candidates = lots[(lots["unrl_pl_%"] < -5.0) & (lots["quantity"] > 0)].copy()  # threshold loss

# ---- Proxy map (customize per shelf / compliance) ----
proxy_map = {
    "SPY": ["VOO","IVV"],
    "IVV": ["SPY","VOO"],
    "VOO": ["SPY","IVV"],
    "QQQ": ["SCHG","VGT"],
    "IWM": ["VTWO","IJR"],
    "EFA": ["IEFA","VEA"],
    "AGG": ["BND","SCHZ"],
}

wash_sale_window_days = 30
today = dt.date.today()

def within_wash_window(trade_date):
    return abs((today - trade_date.date()).days) < wash_sale_window_days

proposals = []
for _, row in candidates.iterrows():
    t = row["ticker"]
    lot_dt = row["trade_date"]
    if within_wash_window(lot_dt):
        status = "DEFER (inside wash-sale window)"
        proxy = []
    else:
        proxy = proxy_map.get(t, [])
        status = "OK" if proxy else "REVIEW (no proxy configured)"
    proposals.append({
        "account_id": row["account_id"],
        "ticker": t,
        "lot_id": row["lot_id"],
        "unrealized_loss_%": round(row["unrl_pl_%"],2),
        "qty": row["quantity"],
        "proxy_options": ",".join(proxy) if proxy else "-",
        "action": f"Sell {t}, buy {proxy[0]}" if proxy and status=="OK" else status
    })

out = pd.DataFrame(proposals).sort_values(["account_id","unrealized_loss_%"])
out.to_csv("out/tlh_proposals.csv", index=False)
print("Saved: out/tlh_proposals.csv")
