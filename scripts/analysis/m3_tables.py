"""M3 final tables: audit of the reference models and the penalty preview, with the grid shares (D75)."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
ORDER = ["persistence", "newell", "ovm", "idm", "knn", "mlp", "pidl", "residual_idm", "gru", "lstm", "perl"]


def load(exp, model):
    d = root / "runs" / exp / "follownet_highd" / model / "driver_fold0_seed0"
    if not (d / "stability.json").exists():
        return None, None
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    return metrics, json.loads((d / "stability.json").read_text(encoding="utf-8"))["audit"]


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}"


def audit_cells(a):
    s = a["summary"]
    sup = [e for e in a["equilibria"] if e["in_support"]]
    found = [e for e in sup if e.get("max_gain") is not None]
    gn, gs = s["grid_numerical"]["support"], s["grid_sign"]["support"]
    top = max(found, key=lambda e: e["max_gain"]) if found else None
    gains = sorted(e["max_gain"] for e in found)
    return {
        "n_sup": len(sup), "n_found": len(found),
        "num": f"{pct(gn['stable'])} / {pct(gn['unstable'])} / {pct(gn['none'] + gn['indifferent'] + gn['undefined'])}",
        "sign": f"{pct(gs['stable'])} / {pct(gs['unstable'])} / {pct(gs['none'] + gs['indifferent'] + gs['undefined'])}",
        "gn": gn, "gs": gs,
        "local": pct(s["share_locally_unstable"]["support"]),
        "top": "-" if top is None else f"{top['max_gain']:.3f} ({top['omega_at_max']:.3f}; {top['v']:.0f})",
        "median": "-" if not gains else f"{gains[len(gains) // 2]:.3f}",
        "agree": pct(s["agreement_gain"]["support"]),
        "collided": s["n_collided"]["support"], "stopped": s["n_stopped"]["support"], "clipped": s["n_clipped"]["support"],
    }


print("support_v:", load("m2", "mlp")[1]["support_v"])
print()
print("AUDIT (speeds inside the data range)")
print("| Run | Model | Test RMSE s (m) | Grid speeds | Numerical: stable / unstable / no equilibrium (%) | Sign criterion: stable / unstable / no eq. (%) "
      "| Locally unstable (%) | Largest gain (rad/s; m/s) | Median largest gain | Numerical = analytic (%) | Collided / stopped / clipped |")
print("|---|---|---:|---:|---|---|---:|---|---:|---:|---|")
for exp in ("m2", "m2_acc0"):
    for model in ORDER:
        m, a = load(exp, model)
        if a is None:
            continue
        c = audit_cells(a)
        print(f"| {exp} | {model} | {m['test']['rmse_s_mean']:.2f} | {c['n_sup']} | {c['num']} | {c['sign']} | {c['local']} | {c['top']} | {c['median']} "
              f"| {c['agree']} | {c['collided']} / {c['stopped']} / {c['clipped']} |")

print()
print("PENALTY PREVIEW")
ROWS = [
    ("mlp", "m2", "none"), ("mlp", "m3_jacobian", "jacobian"), ("mlp", "m3_linear_gain", "linear_gain"),
    ("residual_idm", "m2_acc0", "none"), ("residual_idm", "m3_jacobian", "jacobian"),
    ("gru", "m2", "none"), ("gru", "m3_linear_gain", "linear_gain"),
    ("lstm", "m2", "none"), ("lstm", "m3_linear_gain", "linear_gain"),
]
print("| Model | Penalty | Epochs (best) | Test RMSE s mean / median (m) | Change of the mean | Collisions (%) | Numerical: stable / unstable / no eq. (%) "
      "| Sign: stable / unstable / no eq. (%) | Locally unstable (%) | Largest gain (rad/s; m/s) | Median largest gain | Num. = analytic (%) | Penalty at best epoch | s/epoch |")
print("|---|---|---|---:|---:|---:|---|---|---:|---|---:|---:|---:|---:|")
base = {}
for model, exp, kind in ROWS:
    m, a = load(exp, model)
    c = audit_cells(a)
    tr = m.get("training") or {}
    hist = tr.get("history") or []
    best = m.get("best_epoch") or 0
    pen = hist[best - 1].get("loss_penalty") if best and hist else None
    if pen is None and m.get("recovered"):
        pen = m["recovered"]["penalty_of_saved_weights"].get("value")
    sec = sum(h["time_train_s"] + h["time_val_s"] for h in hist) / len(hist) if hist else None
    rmse = m["test"]["rmse_s_mean"]
    if kind == "none":
        base[model] = rmse
    change = "" if kind == "none" else f"{100 * (rmse / base[model] - 1):+.1f} %"
    epochs = f"{tr.get('epochs')} ({best})" if tr else "lost"
    print(f"| {model} | {kind} | {epochs} | {rmse:.2f} / {m['test']['rmse_s_median']:.2f} | {change} | {100 * m['test']['collision_rate']:.1f} | {c['num']} | {c['sign']} "
          f"| {c['local']} | {c['top']} | {c['median']} | {c['agree']} | {'-' if pen is None else f'{pen:.4f}'} | {'-' if sec is None else f'{sec:.1f}'} |")
    if kind != "none" and hist:
        last = hist[best - 1] if best else hist[-1]
        print("    best epoch:", {k: round(v, 4) for k, v in last.items() if k.startswith("penalty_") or k.startswith("loss_")})
