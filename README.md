# Examples in Wealth Management
Here are 6 concrete, working Python mini-projects (2 each for Front Office, Middle Office, Back Office) in the Wealth Management Space. They use real libraries (yfinance, scikit-learn, pandas, etc.). For GenAI, I’ve used Gemini 2.5 Flash via LangChain and one example with a tiny LangGraph agentic flow.

## Setup
Packages needed
```bash
$> pip install pandas numpy yfinance scikit-learn langchain langgraph langchain-google-genai python-dotenv matplotlib
```
NOTE: these can be included as part of requirements.txt file also, which can be managed using `uv` or `conda` or similar Python env managers.

### Examples Included
#### Front Office
1. GenAI Portfolio Review Notes (personalized, client-ready PDF/email text):

    *What this does:* Pulls a client’s holdings (CSV), downloads live market data with yfinance, computes 1M/3M/YTD perf, top movers, and asks Gemini 2.5 Flash to draft a client-ready review note with next-best-actions. You’ll get a markdown/HTML string to paste into email or PDF.
2. Churn (Attrition) Early-Warning Model (real ML on real signals)

    *What this does:* Trains a logistic regression (or XGBoost if you prefer) on a labeled dataset of client behavior (logins, cash withdrawals, service tickets, response latency, meeting frequency, NPS), produces churn risk scores, and prints a ranked retention call-list for FAs.

#### Mid-Office
1. Tax-Loss Harvesting (TLH) Optimizer (wash-sale aware, ETF proxies)

    *What this does:* Takes client tax lots (CSV), pulls current prices with yfinance, computes unrealized losses, and proposes swap candidates (e.g., SPY ↔ VOO/IVV; QQQ ↔ SCHG/VGT) to realize losses while keeping exposure. Avoids wash-sale windows for the same/similar lot-IDs and flags timing.

2. Dynamic Asset-Allocation Signal (simple regime detector → new weights)

    *What this does:* Uses SPY (equities) and TLT (long bonds) total-return proxies to estimate a risk-on / risk-off regime from rolling volatility & trend, then outputs recommended weights (e.g., 70/30 vs 40/60) plus a rationale string you can send to clients.

### Back-Office
1. Statement Anomaly Detector (catch bad numbers before clients do)

    *What this does:* Runs IsolationForest across prepared statement lines (amounts, fees, accruals, valuations) to flag outliers and emits a QA checklist for ops, so your FAs don’t send error-ridden reports.

2. GenAI Exception-Triage Copilot (turns breaks into clear, client-safe narratives)

    *What this does:* Takes recon/settlement breaks (CSV), clusters them by cause (simple text TF-IDF + KMeans), and asks Gemini 2.5 Flash to produce plain-English explanations + suggested fixes FAs can share (client-safe wording).

#### Bonus
1. (Mini) LangGraph Agent for FO-1 (optional drop-in)

    An agentic orchestration usecase: here is a minimal LangGraph version of the Portfolio Review Notes flow that:

    * Pulls prices with a tool,
    * Formats analytics, and
    * Calls Gemini for the final note.