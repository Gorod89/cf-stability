"""Persistence baseline on one event set (decision D13).

    python scripts/persistence_baseline.py dataset=highd

(a) open loop, stored acceleration series: ``a_hat[k+1] = a[k]``;
(b) open loop, ``Persistence`` model fed with the observed state history: ``a_hat[k]`` vs ``a[k]``;
(c) closed loop, ``rollout_model`` over the whole event from the observed samples ``0..W-1``.
Errors are evaluated on the samples ``k >= W - 1 = 1``. Writes
``runs/baselines/<dataset>/persistence/metrics.json``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from torch import Tensor  # noqa: E402

from cf_stability.data.schema import DT, EventSet  # noqa: E402
from cf_stability.models.base import CFModel  # noqa: E402
from cf_stability.models.persistence import Persistence  # noqa: E402
from cf_stability.train.calibration import resolve_device  # noqa: E402
from cf_stability.train.closed_loop import closed_loop_metrics, pad_events, rollout_model  # noqa: E402
from cf_stability.utils import config_hash, resolve_path, to_plain, write_json  # noqa: E402


def open_loop_predictions(model: CFModel, batch: dict[str, Tensor]) -> Tensor:
    """``a_hat[:, j]`` for ``k = j + W - 1`` from the observed ``(s, dv, v)`` up to ``k``."""
    states = torch.stack((batch["s"], batch["v"] - batch["v_lead"], batch["v"]), dim=-1)
    windows = states.unfold(1, model.window, 1).transpose(-1, -2)  # [B, T - W + 1, W, 3]
    return model(windows.reshape(-1, model.window, 3)).reshape(windows.shape[:2])


def error_stats(sse: np.ndarray, n: np.ndarray, name: str) -> dict[str, float]:
    """Pooled RMSE and mean / median of the per-event RMSE."""
    rmse = np.sqrt(sse / n)
    return {
        f"{name}_pooled": float(np.sqrt(sse.sum() / n.sum())),
        f"{name}_mean": float(rmse.mean()),
        f"{name}_median": float(np.median(rmse)),
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="persistence_baseline")
def main(cfg: DictConfig) -> None:
    t_start = time.perf_counter()
    plain = to_plain(cfg)
    header = {"config": plain, "config_hash": config_hash(plain), "dataset": cfg.dataset}
    device = resolve_device(cfg.device)
    events = EventSet.from_parquet(resolve_path(cfg.paths.events_root) / cfg.dataset).events
    model = Persistence(dt=DT)
    w = model.window
    parts: dict[str, list[np.ndarray]] = {}
    order = np.argsort([len(ev) for ev in events], kind="stable")
    for start in range(0, len(order), cfg.batch_events):
        batch = pad_events([events[i] for i in order[start : start + cfg.batch_events]], device=device)
        mask, a = batch["mask"], batch["a"]
        valid = mask[:, w - 1 :]
        stored = torch.where(mask[:, 1:], (a[:, :-1] - a[:, 1:]) ** 2, 0.0).sum(1)
        predicted = torch.where(valid, (open_loop_predictions(model, batch) - a[:, w - 1 :]) ** 2, 0.0).sum(1)
        res = rollout_model(model, batch["x_lead"], batch["v_lead"], batch["s"][:, :w], batch["v"][:, :w], dt=DT)
        m = closed_loop_metrics(res, batch["s"], batch["v"], mask, start=w - 1)
        values = {
            "sse_a_stored": stored,
            "n_a_stored": mask[:, 1:].sum(1),
            "sse_a_model": predicted,
            "n_a_model": valid.sum(1),
            "sse_s": m["rmse_s"] ** 2 * m["n_samples"],
            "sse_v": m["rmse_v"] ** 2 * m["n_samples"],
            "nrmse_s": m["nrmse_s"],
            "nrmse_v": m["nrmse_v"],
            "collided": m["collided"],
            "n_closed": m["n_samples"],
        }
        for key, value in values.items():
            parts.setdefault(key, []).append(value.cpu().numpy())
    cols = {key: np.concatenate(value) for key, value in parts.items()}

    metrics = {
        **header,
        "n_events": len(events),
        "device": str(device),
        "open_loop": {
            "stored_acceleration": {
                **error_stats(cols["sse_a_stored"], cols["n_a_stored"], "rmse_a"),
                "n_samples": int(cols["n_a_stored"].sum()),
            },
            "persistence_model": {
                **error_stats(cols["sse_a_model"], cols["n_a_model"], "rmse_a"),
                "n_samples": int(cols["n_a_model"].sum()),
            },
        },
        "closed_loop": {
            **error_stats(cols["sse_s"], cols["n_closed"], "rmse_s"),
            **error_stats(cols["sse_v"], cols["n_closed"], "rmse_v"),
            "nrmse_s_median": float(np.median(cols["nrmse_s"])),
            "nrmse_v_median": float(np.median(cols["nrmse_v"])),
            "collision_rate": float(cols["collided"].mean()),
            "start_index": w - 1,
            "n_samples": int(cols["n_closed"].sum()),
        },
        "wall_time_s": time.perf_counter() - t_start,
    }
    run_dir = resolve_path(cfg.paths.runs_root) / "baselines" / cfg.dataset / "persistence"
    path = write_json(run_dir / "metrics.json", metrics)
    ol, cl = metrics["open_loop"], metrics["closed_loop"]
    print(
        f"persistence_baseline  dataset={cfg.dataset}  events={len(events)}  device={device}  "
        f"hash={header['config_hash']}\n"
        f"  open loop RMSE a (pooled): stored a[k] {ol['stored_acceleration']['rmse_a_pooled']:.3f}  "
        f"model {ol['persistence_model']['rmse_a_pooled']:.3f} m/s^2\n"
        f"  closed loop RMSE s {cl['rmse_s_median']:.2f}/{cl['rmse_s_mean']:.2f} m  v {cl['rmse_v_median']:.2f}/"
        f"{cl['rmse_v_mean']:.2f} m/s (median/mean)  collisions {cl['collision_rate']:.1%}\n"
        f"  written: {path}"
    )


if __name__ == "__main__":
    main()
