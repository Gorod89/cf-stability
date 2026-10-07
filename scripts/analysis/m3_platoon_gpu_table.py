import json
import sys
from pathlib import Path

res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
LABEL = {"ZalaZone/handling_part30.csv": "Z30h", "Vicolungo/JRC-VC_280219_part4_highway.csv": "V4h",
         "Vicolungo/JRC-VC_280219_part2.csv": "V2h", "ZalaZone/handling_part32.csv": "Z32a",
         "ZalaZone/handling_part13.csv": "Z13a", "pulse": "pulse"}
for key, r in res.items():
    wall = r.pop("wall_s")
    parts = []
    for prof, x in r.items():
        std = x["speed_std"]
        nums = "/".join(f"{std[p]:.1f}" if std[p] < 100 else f"{std[p]:.0f}" for p in (1, 5, 10, 25, 50))
        coll = f" C{x['collision_step']}@{x['collision_vehicle']}" if x["collided"] else ""
        ge = x.get("growth_error")
        parts.append(f"{LABEL[prof]} {nums}{coll}" + (f" e{ge:.2f}" if ge is not None else ""))
    eq = "".join("E" if x["start_equilibrium"] else "-" for x in r.values())
    print(f"{key} ({wall:.0f}s, eq {eq}): " + "; ".join(parts))
