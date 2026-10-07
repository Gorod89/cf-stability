"""Platoon test of the penalised models of the second pass (time-gap start), with their references."""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(REPO))
from cf_stability.models import load_model  # noqa: E402
from cf_stability.stability.platoon import platoon_test  # noqa: E402

MODELS = {
    "mlp|jacobian": "runs/m3_jacobian/follownet_highd/mlp",
    "mlp|linear_gain": "runs/m3_linear_gain/follownet_highd/mlp",
    "residual_idm|jacobian": "runs/m3_jacobian/follownet_highd/residual_idm",
    "gru|linear_gain": "runs/m3_linear_gain/follownet_highd/gru",
    "lstm|linear_gain": "runs/m3_linear_gain/follownet_highd/lstm",
}
cfg = yaml.safe_load((REPO / "configs/stability/platoon.yaml").read_text(encoding="utf-8"))
results = {}
for label, rel in MODELS.items():
    model = load_model(REPO / rel / "driver_fold0_seed0" / "model.pt")
    for start_gap in ("time_gap", "equilibrium"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        res = platoon_test(model, {**cfg, "start_gap": start_gap})
        wall = time.perf_counter() - t
        results[f"{label}|{start_gap}"] = {"wall_s": wall, **res}
        print(f"{label:24s} {start_gap:12s} {wall:6.1f} s", flush=True)
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
print("platoon: job ended", flush=True)
