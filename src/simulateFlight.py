import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from turbo import model, df, feature_cols
# thresholds
WATCH   = 30
WARNING = 15
def bar(rul, w=20):
    # makes a progress bar to show how much life is left
    filled = int(max(0, min(rul / 50, 1)) * w)
    return f"[{'#' * filled}{'.' * (w - filled)}]"
def get_pred(engine, cycle):
    # finds the row for a specific flight cycle and asks the model what it thinks the remaining life is and the real value
    row = engine[engine["timecycle"] == cycle]
    if row.empty:
        return None
    pred = float(model.predict(row[feature_cols].iloc[[0]])[0])
    actual = int(row["RUL"].iloc[0])
    return pred, actual




def find_sample_cycles(engine):
    results = []
    total = len(engine)
    # run the model on every single cycle per engine and sort them into buckets based on the predicted RUL
    buckets = {"healthy": [], "watch": [], "warning": [], "critical": []}
    for _, row in engine.iterrows():
        pred = float(model.predict(row[feature_cols].values.reshape(1, -1))[0])
        actual = int(row["RUL"])
        cycle = int(row["timecycle"])

        if pred > WATCH:
            buckets["healthy"].append((cycle, pred, actual))
        elif pred > WARNING:
            buckets["watch"].append((cycle, pred, actual))
        else:
            buckets["critical"].append((cycle, pred, actual))


    # grab one example from each bucket->the middle one
    for zone in ["healthy", "watch", "warning", "critical"]:
        items = buckets[zone]
        if items:
            mid = len(items) // 2
            results.append(items[mid])

    return results
# engines to simulate
engines = [1, 10, 30, 50, 80]
print("\nflight sim engine health monitor")
for eid in engines:
    engine = df[df["unitnum"] == eid]
    total = len(engine)
    samples = find_sample_cycles(engine)

    print(f"\nengine #{eid} ({total} total cycles)")
    for cycle, pred, actual in samples:
        if pred > WATCH:
            status = "ok"
        elif pred > WARNING:
            status = "watch"
        else:
            status = "service needed"
        print(
            f"  cycle {cycle:>3d}  |  pred: {pred:5.1f}  actual: {actual:3d}"
            f"  {bar(pred)}  {status}"
        )

    # final recommendation
    final = float(model.predict(engine[feature_cols].iloc[[-1]])[0])
    if final <= WARNING:
        print(f"  -> recommend servicing engine #{eid}")
    elif final <= WATCH:
        print(f"  -> schedule inspection for engine #{eid}")
    else:
        print(f"  -> engine #{eid} is fine")
# accuracy
print("model accuracy")
sample = df.sample(200, random_state=42)
preds = model.predict(sample[feature_cols])
actuals = sample["RUL"].values
errors = np.abs(preds - actuals)
mae = errors.mean()
within_10 = (errors <= 10).mean() * 100
within_20 = (errors <= 20).mean() * 100
print(f"  sample size       -> 200 random readings")
print(f"  mean error        -> {mae:.1f} cycles")
print(f"  within 10 cycles  -> {within_10:.1f}%")
print(f"  within 20 cycles  -> {within_20:.1f}%")
print()
print(f"  {within_10:.1f}% of predictions were within 10 cycles of the actual remaining life")
print()
