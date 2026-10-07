import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
for spec in sys.argv[2:]:
    exp, model = spec.split("/")
    a = json.loads((root / "runs" / exp / "follownet_highd" / model / "driver_fold0_seed0" / "stability.json").read_text(encoding="utf-8"))["audit"]
    rows = [e for e in a["equilibria"] if e["in_support"] and e["s"] is not None]
    out = []
    for k in ("f_s", "f_dv", "f_v", "margin"):
        x = np.array([e[k] for e in rows], dtype=float)
        out.append(f"{k} {np.min(x):+.3f} .. {np.max(x):+.3f} (median {np.median(x):+.3f})")
    print(f"{spec:28s} n={len(rows):2d}  " + ";  ".join(out))
