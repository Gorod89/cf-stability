"""Synthetic data source: single-lane IDM platoons behind a head vehicle with a stop-and-go episode.

Used by the tests and for smoke runs without downloaded data. Every platoon is one site.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from cf_stability.data.extraction import ExtractionConfig, PairSeries, extract_events
from cf_stability.data.schema import DT, EventSet

# heterogeneous IDM parameters and vehicle lengths, uniform ranges
IDM_RANGES = {"v0": (25.0, 35.0), "T": (0.9, 1.8), "s0": (1.5, 3.0), "a_max": (0.8, 1.8), "b": (1.2, 2.5)}
LENGTH_RANGE = (4.0, 5.5)
ACC_CLIP = (-8.0, 4.0)


@dataclass
class Platoon:
    """Simulated platoon: vehicle 0 is the head, vehicle i follows vehicle i - 1."""

    t: np.ndarray  # [n_t]
    x: np.ndarray  # [n_vehicles, n_t] front bumper position
    v: np.ndarray  # [n_vehicles, n_t] true speed
    a: np.ndarray  # [n_vehicles, n_t] acceleration applied from step k to k + 1
    v_measured: np.ndarray  # [n_vehicles, n_t] speed with measurement noise, clipped at 0
    length: np.ndarray  # [n_vehicles]
    params: dict[str, np.ndarray]  # IDM parameters per vehicle, NaN for the head


def simulate_platoon(
    n_vehicles: int,
    duration: float,
    seed: int | Sequence[int],
    dt: float = DT,
    noise_speed: float = 0.0,
) -> Platoon:
    """Simulate a head vehicle and ``n_vehicles - 1`` IDM followers (semi-implicit Euler).

    Followers start in IDM equilibrium at the head's initial speed. ``noise_speed`` is the
    standard deviation of Gaussian noise on the measured speeds.
    """
    if n_vehicles < 2:
        raise ValueError("a platoon needs at least 2 vehicles")
    rng = np.random.default_rng(seed)
    n_t = int(round(duration / dt)) + 1
    t = np.arange(n_t) * dt
    params = {name: rng.uniform(lo, hi, n_vehicles) for name, (lo, hi) in IDM_RANGES.items()}
    for values in params.values():
        values[0] = np.nan
    length = rng.uniform(*LENGTH_RANGE, n_vehicles)
    head = _head_speed(t, rng)

    x = np.zeros((n_vehicles, n_t))
    v = np.zeros((n_vehicles, n_t))
    a = np.zeros((n_vehicles, n_t))
    v[0] = head
    v[1:, 0] = head[0]
    p = {name: values[1:] for name, values in params.items()}
    gap0 = (p["s0"] + head[0] * p["T"]) / np.sqrt(1.0 - (head[0] / p["v0"]) ** 4)
    x[1:, 0] = -np.cumsum(length[:-1] + gap0)
    for k in range(n_t):
        s = x[:-1, k] - length[:-1] - x[1:, k]
        idm = _idm(s, v[1:, k], v[1:, k] - v[:-1, k], p["v0"], p["T"], p["s0"], p["a_max"], p["b"])
        a[1:, k] = np.clip(idm, *ACC_CLIP)
        if k + 1 < n_t:
            v[1:, k + 1] = np.maximum(v[1:, k] + a[1:, k] * dt, 0.0)
            x[:, k + 1] = x[:, k] + v[:, k + 1] * dt
    a[0, :-1] = np.diff(head) / dt
    a[0, -1] = a[0, -2]

    v_measured = v.copy()
    if noise_speed > 0.0:
        v_measured = np.maximum(v + rng.normal(0.0, noise_speed, v.shape), 0.0)
    return Platoon(t=t, x=x, v=v, a=a, v_measured=v_measured, length=length, params=params)


def build_events(data_cfg: Mapping, extraction: ExtractionConfig, stats: Counter | None = None) -> EventSet:
    """Simulate ``n_platoons`` platoons (site ``platoon<i>``) and extract their events."""
    name = str(data_cfg.get("name", "synthetic"))
    n_platoons = int(data_cfg.get("n_platoons", 4))
    n_vehicles = int(data_cfg.get("n_vehicles", 6))
    duration = float(data_cfg.get("duration", 180.0))
    noise_speed = float(data_cfg.get("noise_speed", 0.0))
    seed = int(data_cfg.get("seed", 0))
    events = []
    for i in range(n_platoons):
        platoon = simulate_platoon(n_vehicles, duration, (seed, i), dt=extraction.dt, noise_speed=noise_speed)
        for pair in platoon_pairs(platoon, name, f"platoon{i}"):
            events.extend(extract_events(pair, extraction, stats))
    return EventSet(events)


def platoon_pairs(platoon: Platoon, dataset: str, site: str) -> list[PairSeries]:
    """One PairSeries per follower (raw ids are the platoon positions, lane 1)."""
    n_t = platoon.t.shape[0]
    lane = np.ones(n_t, dtype=np.int64)
    pairs = []
    for i in range(1, platoon.x.shape[0]):
        pairs.append(
            PairSeries(
                dataset=dataset,
                site=site,
                follower_id=f"{dataset}/{site}/{i}",
                t=platoon.t,
                x_follower=platoon.x[i],
                v=platoon.v_measured[i],
                leader_id=np.full(n_t, f"{dataset}/{site}/{i - 1}", dtype=object),
                x_lead=platoon.x[i - 1] - platoon.length[i - 1],
                v_lead=platoon.v_measured[i - 1],
                lane_follower=lane,
                lane_leader=lane,
                meta={
                    "leader_length": float(platoon.length[i - 1]),
                    "follower_length": float(platoon.length[i]),
                    "idm": {name: float(values[i]) for name, values in platoon.params.items()},
                },
            )
        )
    return pairs


def _idm(s, v, dv, v0, T, s0, a_max, b):
    """IDM of docs/data_contract.md, section 6 (dv = v - v_lead)."""
    s_star = s0 + np.maximum(0.0, v * T + v * dv / (2.0 * np.sqrt(a_max * b)))
    return a_max * (1.0 - (v / v0) ** 4 - (s_star / s) ** 2)


def _ramp(x: np.ndarray) -> np.ndarray:
    """Smooth step from 0 (x <= 0) to 1 (x >= 1)."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


def _head_speed(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cruise speed with slow waves and one full stop (brake 12 s, stand 3-8 s, go 15 s)."""
    cruise = rng.uniform(12.0, 18.0)
    periods = rng.uniform((30.0, 10.0), (60.0, 20.0))
    phases = rng.uniform(0.0, 2.0 * np.pi, 2)
    waves = 1.0 + 0.08 * np.sin(2.0 * np.pi * t / periods[0] + phases[0]) + 0.04 * np.sin(
        2.0 * np.pi * t / periods[1] + phases[1]
    )
    brake, stand, go = 12.0, rng.uniform(3.0, 8.0), 15.0
    start = rng.uniform(0.25, 0.45) * float(t[-1])
    factor = 1.0 - _ramp((t - start) / brake) + _ramp((t - start - brake - stand) / go)
    return cruise * waves * factor
