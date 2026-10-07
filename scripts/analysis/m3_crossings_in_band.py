"""All upward zero crossings of f(s, 0, v) on the scan grid of the audit: how many grid speeds have one inside the band of the data."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
from cf_stability.models import load_model  # noqa: E402
from cf_stability.stability.equilibrium import steady_acc  # noqa: E402

band = {int(k): q for k, q in json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")).items()}
RUNS = [("m2", m) for m in ("idm", "knn", "mlp", "pidl", "residual_idm", "gru", "lstm", "perl")]
RUNS += [("m2_acc0", m) for m in ("mlp", "residual_idm", "gru")]
RUNS += [("m3_jacobian", "mlp"), ("m3_jacobian", "residual_idm"), ("m3_linear_gain", "mlp"), ("m3_linear_gain", "gru"), ("m3_linear_gain", "lstm")]
speeds = [v for v in range(5, 27) if v in band]
grid = torch.linspace(1.0, 200.0, 400, dtype=torch.float64)
print("| Run | Model | Speeds with an upward crossing inside q05-q95 | ... inside q01-q99 | Speeds where the first crossing is outside but another is inside (q05-q95) | Mean f_s-like slope at the band median (1/s^2) |")
print("|---|---|---:|---:|---:|---:|")
torch.backends.cudnn.enabled = False
for exp, model_name in RUNS:
    model = load_model(REPO / "runs" / exp / "follownet_highd" / model_name / "driver_fold0_seed0" / "model.pt").double()
    n5 = n1 = rescued = 0
    slopes = []
    with torch.no_grad():
        for v in speeds:
            f = steady_acc(model, grid, torch.full_like(grid, float(v))).numpy()
            up = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]
            roots = [float(grid[i] + (grid[i + 1] - grid[i]) * (-f[i]) / (f[i + 1] - f[i])) for i in up]
            q = band[v]
            in5 = [r for r in roots if q[1] <= r <= q[3]]
            in1 = [r for r in roots if q[0] <= r <= q[4]]
            n5 += bool(in5)
            n1 += bool(in1)
            rescued += bool(in5) and not (q[1] <= roots[0] <= q[3])
            lo, hi = torch.tensor([q[2] - 2.0, q[2] + 2.0], dtype=torch.float64)
            pair = steady_acc(model, torch.stack((lo, hi)), torch.full((2,), float(v), dtype=torch.float64)).numpy()
            slopes.append((pair[1] - pair[0]) / 4.0)
    print(f"| {exp} | {model_name} | {n5} / {len(speeds)} | {n1} / {len(speeds)} | {rescued} | {np.mean(slopes):+.4f} |")
