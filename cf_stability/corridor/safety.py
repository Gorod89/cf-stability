"""Safety filter of the corridor (docs/m11_contract.md, D130): a control-barrier clip of the law's acceleration.

Barrier of a follower at ``(s, dv, v)`` (``dv = v - v_lead``, so ``v_lead = v - dv``)::

    h = s - s_min - v tau_s - max(0, v^2 - v_lead^2) / (2 b)

(the gap beyond a standstill distance, a time headway and the extra distance the follower needs to brake to the
leader's speed at ``b``). Under a constant leader speed ``dh/dt = (v_lead - v) - a (tau_s + [v^2 > v_lead^2] v / b)``
(``[v^2 > v_lead^2]`` is ``[v > v_lead]`` for the speeds of the loop, which are never negative), so the largest
acceleration with ``dh/dt + alpha h >= 0`` is

    a_safe = ((v_lead - v) + alpha h) / (tau_s + [v^2 > v_lead^2] v / b),

and the filter replaces the law's acceleration ``a`` by ``min(a, a_safe)`` (before the clip of the loop to ``[a_min,
a_max]``). It is inactive wherever ``a_safe >= a``: at an equilibrium ``(s*, 0, v)`` of a law (``a = 0``) exactly when
``h = s* - s_min - v tau_s > 0``, and then in a neighbourhood of it, so the linearisation of the law there, and the
audit of the equilibria, is unchanged. Without a leader in range the loop's state ``(leader_range, 0, v)`` gives
``a_safe = alpha (leader_range - s_min - v tau_s) / tau_s``, far above any acceleration of the clip.

In the loop (``cf_stability/corridor/loop.py``) the filter is switched on by ``sim.safety_filter`` of
``configs/run_corridor.yaml`` (default ``null``: off, and the key stays out of the config hash); its values enter the
hash of a filtered run. Filtered runs live under the law name with the suffix :data:`LAW_SUFFIX` (law files that are
copies of the unfiltered laws, ``configs/corridor/laws.yaml``), so that they never replace the runs of the unfiltered
law: the loop refuses a filtered run of any other law and an unfiltered run of a law with the suffix.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import numpy as np

LAW_SUFFIX = "_filter"


@dataclasses.dataclass(frozen=True)
class SafetyConfig:
    """Parameters of the barrier (D130)."""

    s_min: float = 2.0  # m
    tau_s: float = 0.5  # s
    b: float = 4.0  # m/s^2, the braking of the extra distance
    alpha: float = 1.0  # 1/s, the class-K gain of the barrier condition

    def __post_init__(self) -> None:
        if not (self.tau_s > 0.0 and self.b > 0.0 and self.alpha > 0.0 and self.s_min >= 0.0):
            raise ValueError(f"safety filter: tau_s, b, alpha > 0 and s_min >= 0, got {dataclasses.asdict(self)}")

    @classmethod
    def from_mapping(cls, raw: Any) -> "SafetyConfig | None":
        """``None``/``False``: off (None); ``True``: the defaults; a mapping: its values (``enabled: false`` switches
        it off)."""
        if raw is None or raw is False:
            return None
        if raw is True:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError(f"sim.safety_filter: null, true or a mapping of {[f.name for f in dataclasses.fields(cls)]}, "
                             f"got {raw!r}")  # fmt: skip
        values = dict(raw)
        if not values.pop("enabled", True):
            return None
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown keys of sim.safety_filter: {sorted(unknown)}")
        return cls(**{key: float(value) for key, value in values.items()})

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def barrier(s: np.ndarray, dv: np.ndarray, v: np.ndarray, cfg: SafetyConfig) -> np.ndarray:
    """``h = s - s_min - v tau_s - max(0, v^2 - v_lead^2) / (2 b)``, ``v_lead = v - dv``."""
    s, dv, v = (np.asarray(x, dtype=np.float64) for x in (s, dv, v))
    v_lead = v - dv
    return s - cfg.s_min - v * cfg.tau_s - np.maximum(0.0, v**2 - v_lead**2) / (2.0 * cfg.b)


def bound(s: np.ndarray, dv: np.ndarray, v: np.ndarray, cfg: SafetyConfig) -> np.ndarray:
    """The largest acceleration with ``dh/dt + alpha h >= 0`` under a constant leader speed:
    ``((v_lead - v) + alpha h) / (tau_s + [v^2 > v_lead^2] v / b)``."""
    s, dv, v = (np.asarray(x, dtype=np.float64) for x in (s, dv, v))
    v_lead = v - dv
    closing = v**2 > v_lead**2  # the braking term of h is active (v > v_lead for speeds >= 0)
    return (-dv + cfg.alpha * barrier(s, dv, v, cfg)) / (cfg.tau_s + np.where(closing, v / cfg.b, 0.0))


def apply(acc: np.ndarray, s: np.ndarray, dv: np.ndarray, v: np.ndarray, cfg: SafetyConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(min(acc, bound), active)``: the filtered accelerations and where the bound was below the law's value."""
    acc = np.asarray(acc, dtype=np.float64)
    limit = bound(s, dv, v, cfg)
    active = limit < acc
    return np.where(active, limit, acc), active


def check_law_name(law: str, cfg: SafetyConfig | None) -> None:
    """A filtered run belongs to a law named ``<law>_filter`` and an unfiltered run to any other law (the run
    directories of the two never collide); raises ``ValueError`` otherwise."""
    filtered_name = str(law).endswith(LAW_SUFFIX)
    if cfg is not None and not filtered_name:
        raise ValueError(f"law {law}: a run with sim.safety_filter needs a law named <law>{LAW_SUFFIX} (a copy of the "
                         "law file), so that it does not replace the runs of the unfiltered law")  # fmt: skip
    if cfg is None and filtered_name:
        raise ValueError(f"law {law}: the laws <law>{LAW_SUFFIX} run with sim.safety_filter only")


def simulate_pair(
    law: Any, v_lead: np.ndarray, s0: float, v0: float, cfg: SafetyConfig | None, dt: float = 0.1,
    a_min: float = -8.0, a_max: float = 4.0,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Two-vehicle toy with the arithmetic of the loop: the follower's acceleration ``law(s, dv, v)`` (a callable of
    arrays), filtered when ``cfg`` is given, clipped to ``[a_min, a_max]``; ``v_next = max(0, v + a dt)`` and the
    position advanced by ``v_next dt`` (SUMO's Euler update), the leader by its prescribed speed ``v_lead[k + 1]``.
    Returns the series ``s``, ``v``, ``a`` and ``active``."""
    v_lead = np.asarray(v_lead, dtype=np.float64)
    n = len(v_lead)
    s, v = np.empty(n), np.empty(n)
    a, active = np.zeros(n - 1), np.zeros(n - 1, dtype=bool)
    s[0], v[0] = float(s0), float(v0)
    x_lead, x = float(s0), 0.0
    for k in range(n - 1):
        dv = v[k] - v_lead[k]
        acc = float(np.asarray(law(np.array([s[k]]), np.array([dv]), np.array([v[k]]))).reshape(-1)[0])
        if cfg is not None:
            filtered, on = apply(np.array([acc]), np.array([s[k]]), np.array([dv]), np.array([v[k]]), cfg)
            acc, active[k] = float(filtered[0]), bool(on[0])
        acc = min(max(acc, a_min), a_max)
        a[k] = acc
        v[k + 1] = max(v[k] + acc * dt, 0.0)
        x += v[k + 1] * dt
        x_lead += v_lead[k + 1] * dt
        s[k + 1] = x_lead - x
    return {"s": s, "v": v, "a": a, "active": active}
