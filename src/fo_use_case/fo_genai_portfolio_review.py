"""
fo_genai_portfolio_review.py:  GenAI Portfolio Review Notes (personalized, 
    client-ready PDF/email text)

What this does:
Pulls a client’s holdings (CSV), downloads live market data with yfinance, 
computes 1M/3M/YTD perf, top movers, and asks Gemini 2.5 Flash to draft a 
client-ready review note with next-best-actions. You’ll get a markdown/HTML 
string to paste into email or PDF.
"""

# fo_genai_portfolio_review.py
import os, datetime as dt
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
assert GOOGLE_API_KEY, "Set GOOGLE_API_KEY"

# ---- 1) Load client holdings (CSV with columns: ticker, quantity, cost_basis) ----
# Example row: AAPL, 120, 145.20
holdings = pd.read_csv("data/client_holdings.csv")
holdings["ticker"] = holdings["ticker"].str.upper()

# ---- 2) Download prices & compute returns ----
tickers = holdings["ticker"].unique().tolist()
end = dt.date.today()
start = end - dt.timedelta(days=400)  # ~1y+ buffer

prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
if isinstance(prices, pd.Series):  # if single ticker, make DataFrame
    prices = prices.to_frame()

latest = prices.iloc[-1]
px_1m = prices.iloc[-22] if len(prices) > 22 else prices.iloc[0]
px_3m = prices.iloc[-66] if len(prices) > 66 else prices.iloc[0]
px_ytd = prices[prices.index >= pd.Timestamp(end.year, 1, 1)].iloc[0] if (prices.index >= pd.Timestamp(end.year,1,1)).any() else prices.iloc[0]

def ret(a, b): 
    try: 
        return (a/b - 1.0) * 100.0
    except Exception:
        return float("nan")

summary_rows = []
for t in tickers:
    r1m  = ret(latest[t], px_1m[t])
    r3m  = ret(latest[t], px_3m[t])
    rytd = ret(latest[t], px_ytd[t])
    summary_rows.append({"ticker": t, "ret_1m_%": r1m, "ret_3m_%": r3m, "ret_ytd_%": rytd})

perf = pd.DataFrame(summary_rows).sort_values("ret_1m_%", ascending=False)

# ---- 3) Compute portfolio snapshot ----
latest_prices = latest.reindex(holdings["ticker"])
holdings = holdings.join(latest_prices.rename("last"), on="ticker")
holdings["mv"] = holdings["quantity"] * holdings["last"]
holdings["unrealized_%"] = (holdings["last"]/holdings["cost_basis"] - 1) * 100
portfolio_mv = holdings["mv"].sum()
top_movers = perf.head(3).to_dict(orient="records"), perf.tail(3).to_dict(orient="records")

# ---- 4) Ask Gemini 2.5 Flash to draft a review note ----
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

client_profile = {
    "name": "Radhika Shah",
    "goals": ["Retirement income in 7 years", "Child education in 3 years"],
    "risk": "Moderate",
    "constraints": ["Avoid tobacco", "Tax-aware rebalancing"],
}

prompt = f"""
You are a senior Wealth Advisor. Draft a concise, client-ready portfolio review note.

Context:
- Client: {client_profile['name']}, risk: {client_profile['risk']}, goals: {client_profile['goals']}, constraints: {client_profile['constraints']}
- Portfolio market value: ₹{portfolio_mv:,.0f}
- Holdings (top 10 by MV) sample:
{holdings.sort_values('mv', ascending=False).head(10).to_string(index=False)}
- Performance snapshot (1M/3M/YTD):
{perf.to_string(index=False)}

Tasks:
1) Summarize portfolio performance in plain English (avoid jargon).
2) Explain key movers (winners/laggards) + likely drivers (macro/sector).
3) Give 2–3 next-best-actions: rebalancing, tax-loss harvesting, reduce cash drag, etc. Keep suggestions compliant and non-promissory.
4) Keep tone: calm, professional, no hype. 220–280 words. End with 3 bullet "What happens next".
"""

resp = model.invoke([SystemMessage(content="Be compliant, clear, concise."), HumanMessage(content=prompt)])
print(resp.content)

# Optionally, write to a file
with open("out/portfolio_review_note.md", "w", encoding="utf-8") as f:
    f.write(resp.content)
print("Saved: out/portfolio_review_note.md")
