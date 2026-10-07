"""Descriptive statistics of an extracted event set (data/events/<dataset>/describe.json).

    python scripts/describe_events.py openacc follownet_highd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cf_stability.data.schema import DT, EventSet  # noqa: E402
from cf_stability.utils import resolve_path, write_json  # noqa: E402

QUANTILES = (1, 5, 25, 50, 75, 95, 99)


def quantiles(x: np.ndarray) -> dict:
    out = {f"p{q:02d}": float(v) for q, v in zip(QUANTILES, np.percentile(x, QUANTILES))}
    out.update(mean=float(np.mean(x)), std=float(np.std(x)), min=float(np.min(x)), max=float(np.max(x)))
    return out


def describe(events: EventSet) -> dict:
    s = np.concatenate([e.s for e in events])
    v = np.concatenate([e.v for e in events])
    dv = np.concatenate([e.dv for e in events])
    a = np.concatenate([e.a for e in events])
    moving = v > 5.0
    # kinematic consistency of the stored series: d(s)/dt should equal -dv
    ds_res = np.concatenate([np.diff(e.s) / DT + 0.5 * (e.dv[1:] + e.dv[:-1]) for e in events])
    # consistency of the stored acceleration with the stored speed
    a_res = np.concatenate([(e.v[2:] - e.v[:-2]) / (2 * DT) - e.a[1:-1] for e in events])
    jerk = np.concatenate([np.diff(e.a) / DT for e in events])
    per_group: dict[str, int] = {}
    for e in events:
        key = f"{e.site}|{e.meta.get('driver_mode', '-')}"
        per_group[key] = per_group.get(key, 0) + 1
    return {
        "summary": events.summary(),
        "events_per_site_and_mode": dict(sorted(per_group.items())),
        "duration_s": quantiles(np.array([len(e) * DT for e in events])),
        "spacing_m": quantiles(s),
        "speed_mps": quantiles(v),
        "dv_mps": quantiles(dv),
        "acceleration_mps2": quantiles(a),
        "jerk_mps3": quantiles(jerk),
        "time_gap_s_for_v_gt_5": quantiles(s[moving] / v[moving]) if moving.any() else None,
        "share_standing_v_lt_0.1": float(np.mean(v < 0.1)),
        "share_abs_acc_gt_3": float(np.mean(np.abs(a) > 3.0)),
        "rms_ds_dt_plus_dv": float(np.sqrt(np.mean(ds_res**2))),
        "rms_central_diff_v_minus_a": float(np.sqrt(np.mean(a_res**2))),
        "a_source": sorted({str(e.meta.get("a_source")) for e in events}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--events-root", default="data/events")
    args = parser.parse_args()
    for name in args.datasets:
        directory = resolve_path(args.events_root) / name
        report = describe(EventSet.from_parquet(directory))
        write_json(directory / "describe.json", report)
        sm = report["summary"]
        print(f"\n[{name}] events={sm['n_events']} samples={sm['n_samples']} followers={sm['n_followers']} sites={sm['n_sites']}")
        for key in ("duration_s", "spacing_m", "speed_mps", "dv_mps", "acceleration_mps2", "time_gap_s_for_v_gt_5"):
            q = report[key]
            if q is not None:
                print(f"  {key:24s} p05={q['p05']:8.3f} p50={q['p50']:8.3f} p95={q['p95']:8.3f} mean={q['mean']:8.3f} std={q['std']:7.3f}")
        print(f"  standing share={report['share_standing_v_lt_0.1']:.3f}  |a|>3 share={report['share_abs_acc_gt_3']:.4f}  "
              f"rms(ds/dt+dv)={report['rms_ds_dt_plus_dv']:.3f} m/s  rms(dv/dt-a)={report['rms_central_diff_v_minus_a']:.3f} m/s2")
        print(f"  events per site|mode: {report['events_per_site_and_mode']}")


if __name__ == "__main__":
    main()
