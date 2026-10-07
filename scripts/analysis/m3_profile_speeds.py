import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
from cf_stability.stability.platoon import openacc_profile  # noqa: E402
from cf_stability.utils import resolve_path  # noqa: E402

cfg = yaml.safe_load((REPO / "configs/stability/platoon.yaml").read_text(encoding="utf-8"))
for name in cfg["profiles"]:
    p = openacc_profile(resolve_path(cfg["raw_dir"]) / name)
    v = np.asarray(p["v_lead"], dtype=float)
    a = np.gradient(v, 0.1)
    print(f"{name:45s} n={len(v):5d}  v min {v.min():5.1f} p5 {np.quantile(v, 0.05):5.1f} median {np.median(v):5.1f} p95 {np.quantile(v, 0.95):5.1f} max {v.max():5.1f}  "
          f"below 5 m/s {100 * (v < 5).mean():4.1f} %  above 26.8 {100 * (v > 26.8).mean():4.1f} %  a min {a.min():5.2f} max {a.max():5.2f}  last 30 s: v {v[-300]:.1f} -> {v[-1]:.1f}")
