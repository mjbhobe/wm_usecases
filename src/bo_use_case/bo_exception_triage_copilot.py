"""
bo_exception_triage_copilot.py: GenAI Exception-Triage Copilot (turns breaks into 
    clear, client-safe narratives)

What this does:
Takes recon/settlement breaks (CSV), clusters them by cause (simple text TF-IDF + KMeans),
and asks Gemini 2.5 Flash to produce plain-English explanations + suggested fixes 
FAs can share (client-safe wording).
"""
# bo_exception_triage_copilot.py
import os, json
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()
assert os.environ.get("GOOGLE_API_KEY"), "Set GOOGLE_API_KEY"

# Input: breaks.csv
# Columns: break_id, account_id, break_text (free text from recon/settlement systems)
df = pd.read_csv("data/breaks.csv")

# ---- 1) Quick & dirty unsupervised grouping ----
tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
X = tfidf.fit_transform(df["break_text"].fillna(""))

k = min(8, max(2, len(df)//10))  # heuristic clusters
km = KMeans(n_clusters=k, random_state=42, n_init="auto")
df["cluster"] = km.fit_predict(X)

# ---- 2) Summarize each cluster with Gemini ----
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

cluster_summaries = {}
for c in sorted(df["cluster"].unique()):
    sample = df[df["cluster"]==c].head(12)["break_text"].tolist()
    prompt = f"""You are an operations SME. 
Given these exception texts, produce:
1) A one-paragraph plain-English explanation suitable for a client update email.
2) The likely root cause(s) (bullet list).
3) A recommended remediation plan (steps).
Keep it non-defensive, factual, and avoid internal jargon.

Exceptions:
{json.dumps(sample, ensure_ascii=False, indent=2)}
"""
    resp = llm.invoke([SystemMessage(content="Be client-safe, clear and concise."), HumanMessage(content=prompt)])
    cluster_summaries[c] = resp.content.strip()

# ---- 3) Join back & save ----
df["cluster_summary"] = df["cluster"].map(cluster_summaries)
df.to_csv("out/exception_triage_with_summaries.csv", index=False)
print("Saved: out/exception_triage_with_summaries.csv")
