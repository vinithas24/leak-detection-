"""Step 6: Streamlit dashboard.  Run:  streamlit run app.py"""
import time, joblib
import numpy as np, pandas as pd, streamlit as st
from features import engineer, FEATURES

st.set_page_config(page_title="Pipeline Leak Detection", page_icon="💧", layout="wide")
st.title("💧 AI-Based Water Pipeline Leakage Detection")

@st.cache_resource
def load():
    return joblib.load("models/random_forest.joblib"), joblib.load("models/isolation_forest.joblib")
rf, iso = load()

st.sidebar.header("Settings")
model_name = st.sidebar.radio("Model", ["Random Forest", "Isolation Forest"])
threshold = st.sidebar.slider("Alert probability threshold (RF)", 0.1, 0.9, 0.5, 0.05)
mode = st.sidebar.radio("Mode", ["Replay test data", "Upload CSV", "Manual input"])

def predict(df):
    f = engineer(df)
    if model_name == "Random Forest":
        p = rf.predict_proba(f[FEATURES])[:, 1]
        return f.assign(leak_score=p, pred=(p >= threshold).astype(int))
    s = -iso.score_samples(f[FEATURES])
    return f.assign(leak_score=s, pred=(iso.predict(f[FEATURES]) == -1).astype(int))

def status(pred):
    if pred: st.error("🚨 ALERT: SUSPECTED LEAK DETECTED - inspect pipeline!")
    else:    st.success("✅ Pipeline status: NORMAL")

if mode == "Manual input":
    c = st.columns(4)
    fi = c[0].number_input("Flow in", 0.0, 200.0, 50.0)
    fo = c[1].number_input("Flow out", 0.0, 200.0, 49.5)
    pr = c[2].number_input("Pressure", 0.0, 150.0, 60.0)
    us = c[3].number_input("Water usage", 0.0, 200.0, 49.5)
    if st.button("Check"):
        r = predict(pd.DataFrame({"flow_in": [fi], "flow_out": [fo],
                                  "pressure": [pr], "water_usage": [us]})).iloc[0]
        status(r.pred); st.metric("Leak score", f"{r.leak_score:.2f}")

else:
    if mode == "Upload CSV":
        up = st.file_uploader("CSV with flow_in, flow_out, pressure, water_usage")
        data = pd.read_csv(up) if up else None
    else:
        data = pd.read_csv("data/test_set.csv")
    if data is None:
        st.info("Upload a file to begin."); st.stop()

    res = predict(data.reset_index(drop=True))
    n = len(res)
    start = st.sidebar.slider("Start row", 0, n - 1, 0)
    speed = st.sidebar.slider("Replay speed (sec/step)", 0.0, 1.0, 0.05)

    c1, c2, c3 = st.columns(3)
    c3.metric("Total alerts", int(res.pred.sum()))
    if st.button("▶ Start live replay"):
        box, k1, k2, chart = st.empty(), c1.empty(), c2.empty(), st.empty()
        for i in range(start + 20, n, 5):
            w = res.iloc[max(0, i - 120):i]; last = w.iloc[-1]
            with box.container(): status(last.pred)
            k1.metric("Pressure", f"{last.pressure:.1f}")
            k2.metric("Flow diff (in-out)", f"{last.flow_diff:.1f}")
            chart.line_chart(w[["pressure", "flow_diff", "leak_score"]].reset_index(drop=True))
            time.sleep(speed)
    else:
        status(res.pred.iloc[-1])
        st.line_chart(res[["pressure", "flow_diff", "leak_score"]].iloc[:600])
    st.subheader("Detected leak events")
    cols = [c for c in ["timestamp", "flow_in", "flow_out", "pressure", "leak_score"] if c in res]
    st.dataframe(res[res.pred == 1][cols].head(200))
