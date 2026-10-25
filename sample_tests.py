"""Step 7: Sample normal / leak test results (put these in your report)."""
import joblib, pandas as pd
from features import engineer, FEATURES
rf = joblib.load("models/random_forest.joblib")
tests = pd.DataFrame({
    "case": ["Normal flow", "Normal peak demand", "Small leak", "Large leak"],
    "flow_in": [50, 72, 52, 55], "flow_out": [49.5, 71.5, 46, 40],
    "pressure": [60, 55, 52, 44], "water_usage": [49.5, 71.5, 46, 40]})
f = engineer(tests.drop(columns="case"))
tests["leak_prob"] = rf.predict_proba(f[FEATURES])[:, 1].round(2)
tests["status"] = tests.leak_prob.apply(lambda p: "LEAK" if p >= .5 else "Normal")
print(tests.to_string(index=False)); tests.to_csv("outputs/sample_test_results.csv", index=False)
