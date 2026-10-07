"""Follow-up: batched platoon test on the GPU, six reference models, both start gaps."""
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
from cf_stability.stability.platoon import openacc_profile, platoon_test  # noqa: E402
from cf_stability.utils import resolve_path  # noqa: E402

MODELS = {
    "idm": "runs/m2/follownet_highd/idm",
    "mlp": "runs/m2/follownet_highd/mlp",
    "gru": "runs/m2/follownet_highd/gru",
    "lstm": "runs/m2/follownet_highd/lstm",
    "residual_idm_acc0": "runs/m2_acc0/follownet_highd/residual_idm",
    "mlp_acc0": "runs/m2_acc0/follownet_highd/mlp",
}
cfg = yaml.safe_load((REPO / "configs/stability/platoon.yaml").read_text(encoding="utf-8"))
t = time.perf_counter()
for name in cfg["profiles"]:
    openacc_profile(resolve_path(cfg["raw_dir"]) / name)
print(f"loading the 5 OpenACC files: {time.perf_counter() - t:.1f} s", flush=True)
results = {}
for label, rel in MODELS.items():
    model = load_model(REPO / rel / "driver_fold0_seed0" / "model.pt")
    for start_gap in ("time_gap", "equilibrium"):
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = platoon_test(model, {**cfg, "start_gap": start_gap})
        wall = time.perf_counter() - t
        results[f"{label}|{start_gap}"] = {"wall_s": wall, **res}
        print(f"{label:18s} {start_gap:12s} {wall:6.1f} s (incl. file loading)", flush=True)
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
