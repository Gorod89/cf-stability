"""Spacing band of the near-steady training samples per grid speed, and the equilibria of the audited models against it."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
from cf_stability.data.schema import EventSet  # noqa: E402
from cf_stability.data.splits import fold_ids, load_split  # noqa: E402

events = EventSet.from_parquet(REPO / "data/events/follownet_highd")
split = load_split(REPO / "data/splits/follownet_highd_driver.json")
train = events.subset(fold_ids(split, 0)[0]).events
s = np.concatenate([ev.s for ev in train])
dv = np.concatenate([ev.dv for ev in train])
v = np.concatenate([ev.v for ev in train])
a = np.concatenate([ev.a for ev in train])
print(f"training samples {len(s)}; near steady (|dv| < 0.5 m/s, |a| < 0.3 m/s^2): {100 * np.mean((np.abs(dv) < 0.5) & (np.abs(a) < 0.3)):.0f} %")
band = {}
print("v    n_near   q01   q05   q50   q95   q99   (time gap of the median, s)")
for vg in range(5, 31):
    near = (np.abs(v - vg) < 0.5) & (np.abs(dv) < 0.5) & (np.abs(a) < 0.3)
    if near.sum() < 200:
        print(f"{vg:>3}  {near.sum():>6}   too few samples")
        continue
    q = np.quantile(s[near], (0.01, 0.05, 0.5, 0.95, 0.99))
    band[vg] = q
    print(f"{vg:>3}  {near.sum():>6}  " + " ".join(f"{x:5.1f}" for x in q) + f"   {q[2] / vg:4.2f}")
Path(sys.argv[2]).write_text(json.dumps({str(k): list(map(float, q)) for k, q in band.items()}), encoding="utf-8")

RUNS = [("m2", m) for m in ("ovm", "idm", "knn", "mlp", "pidl", "residual_idm", "gru", "lstm", "perl")]
RUNS += [("m2_acc0", m) for m in ("mlp", "residual_idm", "gru")]
RUNS += [("m3_jacobian", "mlp"), ("m3_jacobian", "residual_idm"), ("m3_linear_gain", "mlp"), ("m3_linear_gain", "gru"), ("m3_linear_gain", "lstm")]
print()
print(f"Grid speeds with a band: {sorted(band)}")
print("| Run | Model | Speeds | Stable, inside q05-q95 | Unstable, inside | Equilibrium outside the band | No equilibrium | Sign: stable inside | Sign: unstable inside | Inside q01-q99 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for exp, model in RUNS:
    a_ = json.loads((REPO / "runs" / exp / "follownet_highd" / model / "driver_fold0_seed0" / "stability.json").read_text(encoding="utf-8"))["audit"]
    rows = [e for e in a_["equilibria"] if e["in_support"] and int(e["v"]) in band]
    n = len(rows)
    inside = [e for e in rows if e["s"] is not None and band[int(e["v"])][1] <= e["s"] <= band[int(e["v"])][3]]
    wide = [e for e in rows if e["s"] is not None and band[int(e["v"])][0] <= e["s"] <= band[int(e["v"])][4]]
    none = [e for e in rows if e["s"] is None]
    outside = n - len(inside) - len(none)
    st = sum(not e["unstable"] for e in inside)
    sst = sum(e["margin"] >= 0 and bool(e["local_stable"]) for e in inside)
    print(f"| {exp} | {model} | {n} | {100 * st / n:.0f} | {100 * (len(inside) - st) / n:.0f} | {100 * outside / n:.0f} | {100 * len(none) / n:.0f} "
          f"| {100 * sst / n:.0f} | {100 * (len(inside) - sst) / n:.0f} | {100 * len(wide) / n:.0f} |")
