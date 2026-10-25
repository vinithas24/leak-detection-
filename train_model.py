"""Step 2-5: EDA, feature engineering, training, evaluation."""
import joblib, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
from features import engineer, FEATURES

df = pd.read_csv("data/pipeline_data.csv", parse_dates=["timestamp"])
print("Missing values:\n", df.isna().sum(), "\n")
df = engineer(df)

# ---------------- EDA ----------------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for a, c in zip(ax, ["flow_in", "pressure", "flow_diff"]):
    a.plot(df.timestamp, df[c], lw=.7)
    a.fill_between(df.timestamp, df[c].min(), df[c].max(),
                   where=df.leak == 1, color="red", alpha=.2, label="leak")
    a.set_ylabel(c)
ax[0].legend(); plt.tight_layout(); plt.savefig("outputs/eda_timeseries.png"); plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df[FEATURES + ["leak"]].corr(), cmap="coolwarm", center=0)
plt.title("Correlation heatmap"); plt.tight_layout()
plt.savefig("outputs/eda_correlation.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, c in zip(ax, ["pressure", "flow_diff", "flow_ratio"]):
    sns.boxplot(x="leak", y=c, data=df, ax=a)
plt.tight_layout(); plt.savefig("outputs/eda_boxplots.png"); plt.close()

# ------------- split (chronological, no leakage) -------------
cut = int(len(df) * 0.8)
Xtr, Xte = df[FEATURES][:cut], df[FEATURES][cut:]
ytr, yte = df.leak[:cut], df.leak[cut:]

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=0, n_jobs=-1),
}
results, preds = {}, {}
for name, m in models.items():
    m.fit(Xtr, ytr); preds[name] = m.predict(Xte)

# Isolation Forest: trained on NORMAL data only
iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=0)
iso.fit(Xtr[ytr == 0])
preds["Isolation Forest"] = (iso.predict(Xte) == -1).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax_, (name, p) in zip(axes, preds.items()):
    cm = confusion_matrix(yte, p)
    tn, fp, fn, tp = cm.ravel()
    results[name] = dict(precision=precision_score(yte, p), recall=recall_score(yte, p),
                         f1=f1_score(yte, p), false_alarm_rate=fp / (fp + tn),
                         missed_leak_rate=fn / (fn + tp))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_,
                xticklabels=["Normal", "Leak"], yticklabels=["Normal", "Leak"])
    ax_.set_title(name); ax_.set_xlabel("Predicted"); ax_.set_ylabel("Actual")
    print(f"\n=== {name} ===\n", classification_report(yte, p, target_names=["Normal", "Leak"]))
plt.tight_layout(); plt.savefig("outputs/confusion_matrices.png"); plt.close()

res = pd.DataFrame(results).T.round(3)
print("\nComparison:\n", res); res.to_csv("outputs/metrics.csv")

# feature importance
imp = pd.Series(models["Random Forest"].feature_importances_, index=FEATURES).sort_values()
imp.plot.barh(figsize=(7, 5), title="Random Forest feature importance")
plt.tight_layout(); plt.savefig("outputs/feature_importance.png"); plt.close()

joblib.dump(models["Random Forest"], "models/random_forest.joblib")
joblib.dump(iso, "models/isolation_forest.joblib")
df.iloc[cut:].to_csv("data/test_set.csv", index=False)   # for dashboard demo
print("\nSaved models and plots.")
