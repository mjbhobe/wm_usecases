"""
fo_review_langgraph_agent.py: 
If you prefer agentic orchestration, here is a minimal LangGraph version of the 
Portfolio Review Notes flow that:
 * pulls prices with a tool,
 * formats analytics, and
 * calls Gemini for the final note.
"""
# fo_review_langgraph_agent.py
import os, datetime as dt, pandas as pd, yfinance as yf
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()
assert os.environ.get("GOOGLE_API_KEY")

class State(TypedDict):
    holdings: pd.DataFrame
    analytics: Dict[str, Any]
    note: str

def fetch_prices(state: State) -> State:
    df = state["holdings"]
    tickers = df["ticker"].unique().tolist()
    end = dt.date.today(); start = end - dt.timedelta(days=380)
    px = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(px, pd.Series): px = px.to_frame()
    state["analytics"] = {"prices": px}
    return state

def compute_analytics(state: State) -> State:
    df = state["holdings"].copy()
    px = state["analytics"]["prices"]
    latest = px.iloc[-1]
    df = df.join(latest.rename("last"), on="ticker")
    df["mv"] = df["quantity"] * df["last"]
    port_mv = df["mv"].sum()
    perf = (px.iloc[-1] / px.iloc[-22] - 1.0) * 100.0 if len(px) > 22 else (px.iloc[-1]/px.iloc[0]-1)*100
    top = perf.sort_values(ascending=False).head(3).round(2).to_dict()
    lag = perf.sort_values(ascending=True).head(3).round(2).to_dict()
    state["analytics"].update({"portfolio_mv": port_mv, "top": top, "lag": lag})
    return state

def draft_note(state: State) -> State:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    df = state["holdings"]; an = state["analytics"]
    prompt = f"""Create a client-ready portfolio review note (220–280 words).
Portfolio MV: ₹{an['portfolio_mv']:,.0f}
Top movers (1M %): {an['top']}
Laggards (1M %): {an['lag']}
Holdings sample:
{df.sort_values('mv', ascending=False).head(8).to_string(index=False)}
Tasks: summarize performance; explain movers; 2–3 next best actions; compliant tone."""
    resp = llm.invoke([SystemMessage(content="Be compliant, clear."), HumanMessage(content=prompt)])
    state["note"] = resp.content
    return state

# Build the graph
graph = StateGraph(State)
graph.add_node("fetch_prices", fetch_prices)
graph.add_node("compute_analytics", compute_analytics)
graph.add_node("draft_note", draft_note)
graph.set_entry_point("fetch_prices")
graph.add_edge("fetch_prices", "compute_analytics")
graph.add_edge("compute_analytics", "draft_note")
graph.add_edge("draft_note", END)
app = graph.compile()

if __name__ == "__main__":
    # Input holdings
    holdings = pd.read_csv("data/client_holdings.csv")
    out = app.invoke({"holdings": holdings})
    print(out["note"])
