"""Step 1: Create a synthetic labelled pipeline dataset (normal vs leak).
Replace with a real dataset or ESP32 readings if you have one."""
import numpy as np
import pandas as pd

def generate(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    hour = (t // 60) % 24                      # 1 reading per minute

    # daily demand pattern: peaks morning and evening
    demand = 1 + 0.5*np.exp(-((hour-7)**2)/6) + 0.6*np.exp(-((hour-19)**2)/6)
    flow_in = 50*demand + rng.normal(0, 1.5, n)
    pressure = 60 - 4*(demand-1) + rng.normal(0, 0.8, n)
    flow_out = flow_in - rng.normal(0.5, 0.4, n)      # tiny normal loss

    # leak episodes
    label = np.zeros(n, dtype=int)
    i = 300
    while i < n - 200:
        if rng.random() < 0.5:
            dur = int(rng.integers(40, 150))
            sev = rng.uniform(0.6, 1.6)          # leak severity
            end = min(i + dur, n)
            label[i:end] = 1
            pressure[i:end] -= rng.uniform(5, 14) * sev
            flow_out[i:end] -= rng.uniform(4, 12) * sev
            i = end + int(rng.integers(150, 400))
        else:
            i += 200

    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="min"),
        "flow_in": flow_in.round(2),
        "flow_out": flow_out.round(2),
        "pressure": pressure.round(2),
        "water_usage": (flow_out + rng.normal(0, 0.5, n)).round(2),
        "leak": label,
    })
    # inject a few missing values to practise preprocessing
    for col in ["flow_in", "pressure"]:
        df.loc[rng.choice(n, 25, replace=False), col] = np.nan
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv("data/pipeline_data.csv", index=False)
    print(df.head()); print("\nLeak share:", df.leak.mean().round(3))
