"""Shared preprocessing + feature engineering (used by training AND dashboard)."""
import pandas as pd

RAW = ["flow_in", "flow_out", "pressure", "water_usage"]

def preprocess(df):
    df = df.copy()
    df[RAW] = df[RAW].interpolate(limit_direction="both")      # fill missing
    for c in RAW:                                               # clip outliers
        lo, hi = df[c].quantile([0.001, 0.999])
        df[c] = df[c].clip(lo, hi)
    return df

def engineer(df, w=10):
    df = preprocess(df)
    df["flow_diff"] = df["flow_in"] - df["flow_out"]
    df["flow_ratio"] = df["flow_out"] / (df["flow_in"] + 1e-6)
    df["pressure_mean"] = df["pressure"].rolling(w, min_periods=1).mean()
    df["pressure_std"] = df["pressure"].rolling(w, min_periods=1).std().fillna(0)
    df["pressure_change"] = df["pressure"].diff().fillna(0)
    df["flow_diff_mean"] = df["flow_diff"].rolling(w, min_periods=1).mean()
    df["usage_gap"] = df["flow_in"] - df["water_usage"]
    return df

FEATURES = RAW + ["flow_diff", "flow_ratio", "pressure_mean", "pressure_std",
                  "pressure_change", "flow_diff_mean", "usage_gap"]
