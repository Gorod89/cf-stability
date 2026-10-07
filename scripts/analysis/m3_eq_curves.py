"""Equilibrium spacing of every audited model over the grid speeds; shares with the equilibrium inside the spacing range of the data."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
RUNS = [("m2", m) for m in ("ovm", "idm", "knn", "mlp", "pidl", "residual_idm", "gru", "lstm", "perl")]
RUNS += [("m2_acc0", m) for m in ("mlp", "residual_idm", "gru")]
RUNS += [("m3_jacobian", "mlp"), ("m3_jacobian", "residual_idm"), ("m3_linear_gain", "mlp"), ("m3_linear_gain", "gru"), ("m3_linear_gain", "lstm")]
audits = {}
for exp, model in RUNS:
    d = root / "runs" / exp / "follownet_highd" / model / "driver_fold0_seed0"
    audits[(exp, model)] = json.loads((d / "stability.json").read_text(encoding="utf-8"))["audit"]
    ctx = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["context"]
s_lo, s_hi = ctx["box_low"][0], ctx["box_high"][0]
print(f"spacing range of the training data (1 % and 99 % quantiles): {s_lo:.1f} .. {s_hi:.1f} m; speed {ctx['box_low'][2]:.1f} .. {ctx['box_high'][2]:.1f} m/s")
names = [f"{e.replace('m3_', '').replace('m2_', '').replace('m2', 'ref')}:{m[:8]}" for e, m in RUNS]
print("v    " + " ".join(f"{n[:13]:>13}" for n in names))
speeds = [e["v"] for e in audits[RUNS[0]]["equilibria"]]
for i, v in enumerate(speeds):
    cells = []
    for key in RUNS:
        e = audits[key]["equilibria"][i]
        if e["s"] is None:
            cells.append(e["status"][:6])
        else:
            mark = "" if s_lo <= e["s"] <= s_hi else "*"
            multi = f"/{e['n_crossings']}" if e["n_crossings"] > 1 else ""
            cells.append(f"{e['s']:.1f}{multi}{mark}")
    sup = "" if audits[RUNS[0]]["equilibria"][i]["in_support"] else "  (speed outside)"
    print(f"{v:>4.0f} " + " ".join(f"{c:>13}" for c in cells) + sup)

print()
print("Shares over the grid speeds inside the speed range of the data (22 speeds), numerical rule / sign criterion:")
print("| Run | Model | Equilibrium inside the spacing range | stable, inside | unstable, inside | equilibrium outside the spacing range | no equilibrium | sign: stable, inside | sign: unstable, inside |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for key in RUNS:
    rows = [e for e in audits[key]["equilibria"] if e["in_support"]]
    n = len(rows)
    inside = [e for e in rows if e["s"] is not None and s_lo <= e["s"] <= s_hi]
    outside = [e for e in rows if e["s"] is not None and not s_lo <= e["s"] <= s_hi]
    none = [e for e in rows if e["s"] is None]
    st = sum(not e["unstable"] for e in inside)
    un = sum(bool(e["unstable"]) for e in inside)
    sst = sum(e["margin"] >= 0 and bool(e["local_stable"]) for e in inside)
    sun = len(inside) - sst
    print(f"| {key[0]} | {key[1]} | {len(inside)} | {100 * st / n:.0f} | {100 * un / n:.0f} | {100 * len(outside) / n:.0f} | {100 * len(none) / n:.0f} | {100 * sst / n:.0f} | {100 * sun / n:.0f} |")
