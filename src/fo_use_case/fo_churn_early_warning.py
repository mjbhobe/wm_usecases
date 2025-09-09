"""
fo_churn.py: Churn (Attrition) Early-Warning Model 
    (real ML on real signals)

What this does:
Trains a logistic regression (or XGBoost if you prefer) on a labeled dataset 
of client behavior (logins, cash withdrawals, service tickets, response latency, 
meeting frequency, NPS), produces churn risk scores, and prints a ranked retention 
call-list for FAs.
"""
# fo_churn_early_warning.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# ---- 1) Load your labeled dataset ----
# Columns (example): client_id, churned(0/1), logins_30d, cash_outflows_30d, svc_tickets_90d,
# meeting_gap_days, response_latency_hours, nps_last, portfolio_drawdown_30d
df = pd.read_csv("data/client_behavior_labeled.csv")

X = df[["logins_30d","cash_outflows_30d","svc_tickets_90d",
        "meeting_gap_days","response_latency_hours","nps_last","portfolio_drawdown_30d"]]
y = df["churned"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

# ---- 2) Simple, strong baseline: scaled logistic regression ----
pipe = Pipeline([
    ("std", StandardScaler(with_mean=False)),  # handles zeros & mixed scales well
    ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
])
pipe.fit(X_train, y_train)

# ---- 3) Evaluate ----
probs = pipe.predict_proba(X_test)[:,1]
print("Test AUC:", roc_auc_score(y_test, probs))
print(classification_report(y_test, (probs>0.5).astype(int)))

# ---- 4) Score full book & create FA call list ----
df["churn_score"] = pipe.predict_proba(X)[:,1]
call_list = df[["client_id","churn_score","nps_last","meeting_gap_days","response_latency_hours"]]\
             .sort_values("churn_score", ascending=False)
call_list.head(50).to_csv("out/fa_retention_calllist.csv", index=False)
print("Saved: out/fa_retention_calllist.csv")
