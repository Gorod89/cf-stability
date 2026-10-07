"""Per-speed audit records of chosen runs, compact."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for spec in sys.argv[2:]:
    exp, model = spec.split("/")
    a = json.loads((root / "runs" / exp / "follownet_highd" / model / "driver_fold0_seed0" / "stability.json").read_text(encoding="utf-8"))["audit"]
    print(f"-- {exp}/{model}  window {a['window']}  analytic {a['analytic_gain']}")
    for e in a["equilibria"]:
        tag = "" if e["in_support"] else " (outside)"
        if e.get("max_gain") is None:
            print(f"   v={e['v']:>4.0f}  {e['status']}{tag}")
            continue
        an = e["analytic_max_gain"]
        gains = " ".join(f"{k[:4]} {v:.3f}" for k, v in an.items() if v is not None)
        flag = "" if e["unstable"] == e["analytic_unstable"] else "  <-- DISAGREE"
        extra = "".join(f" {k}" for k in ("clipped", "stopped", "collided") if any(e[k]))
        print(f"   v={e['v']:>4.0f} s={e['s']:>6.1f} {e['status']:<8} n={e['n_crossings']} fs {e['f_s']:+.4f} fdv {e['f_dv']:+.3f} fv {e['f_v']:+.3f} "
              f"M {e['margin']:+.4f} loc {str(e['local_stable'])[0]} | num {e['max_gain']:.3f} @ {e['omega_at_max']:.3f} | {gains}{extra}{flag}{tag}")
