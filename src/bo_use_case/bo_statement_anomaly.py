"""
bo_statement_anomaly.py: Statement Anomaly Detector (catch bad numbers before clients do)

What this does:
Runs IsolationForest across prepared statement lines (amounts, fees, accruals, 
valuations) to flag outliers and emits a QA checklist for ops, so your FAs don’t 
send error-ridden reports.
"""
# bo_statement_anomaly.py
import pandas as pd
from sklearn.ensemble import IsolationForest

# Input: monthly_statement_lines.csv
# Columns: account_id, line_type, amount, fee, valuation, cash_flow, month_start_mv, month_end_mv
df = pd.read_csv("data/monthly_statement_lines.csv")

numeric_cols = ["amount","fee","valuation","cash_flow","month_start_mv","month_end_mv"]
df[numeric_cols] = df[numeric_cols].fillna(0)

model = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
df["anomaly_score"] = model.fit_predict(df[numeric_cols])  # -1 = anomaly

issues = df[df["anomaly_score"] == -1]\
    .sort_values(["account_id","valuation"], ascending=[True, False])

issues.to_csv("out/statement_anomalies.csv", index=False)
print("Saved: out/statement_anomalies.csv")

# Quick summary for FA Ops
summary = issues.groupby("account_id").size().reset_index(name="num_anomalies")\
                .sort_values("num_anomalies", ascending=False)
summary.to_csv("out/statement_anomalies_summary.csv", index=False)
print("Saved: out/statement_anomalies_summary.csv")
