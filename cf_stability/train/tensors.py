"""Padded tensors of an event list and statistics of a training part (docs/m2_contract.md, section 3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from cf_stability.data.schema import Event
from cf_stability.stability.equilibrium import V_GRID
from cf_stability.train.closed_loop import pad_events

BAND_SPEEDS: tuple[float, ...] = V_GRID  # the band is listed on the speed grid of the stability audit


@dataclass
class BandConfig:
    """``train.band``: spacing band of the near-steady training samples per grid speed (D78)."""

    dv_max: float = 0.5  # m/s
    a_max: float = 0.3  # m/s^2
    half_width: float = 0.5  # m/s around the grid speed
    quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95)  # lower edge, median, upper edge
    min_samples: int = 200  # a grid speed with fewer samples has no band

    def __post_init__(self) -> None:
        self.quantiles = tuple(float(q) for q in self.quantiles)
        if len(self.quantiles) != 3 or not 0.0 <= self.quantiles[0] < self.quantiles[1] < self.quantiles[2] <= 1.0:
            raise ValueError("band quantiles must be three increasing numbers in [0, 1]")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "BandConfig":
        values = dict(mapping or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown band keys: {sorted(unknown)}")
        return cls(**values)


class EventTensors:
    """Events stacked into padded ``[N, T]`` tensors on one device.

    ``state [N, T, 3]`` holds ``(s, dv, v)`` with ``dv = v - v_lead`` (as in the rollouts);
    ``x_lead`` is relative to the first leader position of each event, which keeps float32
    rollouts precise. Padding repeats the last valid sample, ``mask`` marks the valid ones.
    """

    def __init__(
        self, events: Sequence[Event], device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
    ) -> None:
        if not events:
            raise ValueError("no events")
        batch = pad_events(events, device=device)  # float64 until the derived series are formed
        x_lead, v_lead, s, v = (batch[name] for name in ("x_lead", "v_lead", "s", "v"))
        self.x_lead = (x_lead - x_lead[:, :1]).to(dtype)
        self.v_lead = v_lead.to(dtype)
        self.state = torch.stack((s, v - v_lead, v), dim=-1).to(dtype)
        self.a = batch["a"].to(dtype)
        self.lengths: Tensor = batch["lengths"]
        self.mask: Tensor = batch["mask"]
        self.event_ids = [ev.event_id for ev in events]
        self.follower_ids = [ev.follower_id for ev in events]
        self.sites = [ev.site for ev in events]

    def __len__(self) -> int:
        return len(self.event_ids)

    @property
    def s(self) -> Tensor:
        return self.state[..., 0]

    @property
    def v(self) -> Tensor:
        return self.state[..., 2]

    def window_ends(self, window: int) -> Tensor:
        """``[M, 2]`` (event index, sample index) of every sample with ``window - 1`` predecessors in its event."""
        steps = torch.arange(self.mask.shape[1], device=self.mask.device)
        return torch.nonzero(self.mask & (steps >= window - 1))

    def state_windows(self, index: Tensor, window: int) -> Tensor:
        """States of the samples ``j - window + 1 .. j`` of the window ends ``index [B, 2]`` -> ``[B, window, 3]``."""
        return self.state[index[:, :1], index[:, 1:] + torch.arange(1 - window, 1, device=index.device)]

    def targets(self, index: Tensor) -> Tensor:
        """Observed accelerations ``[B]`` at the window ends ``index [B, 2]``."""
        return self.a[index[:, 0], index[:, 1]]

    def segments(self, index: Tensor, window: int, horizon: int) -> dict[str, Tensor]:
        """Series ``[B, window + horizon]`` of the samples ``j - window + 1 .. j + horizon``.

        Samples past the end of an event are ``False`` in ``mask``; ``x_lead`` is relative to the
        leader position at ``j``.
        """
        cols = index[:, 1:] + torch.arange(1 - window, horizon + 1, device=index.device)
        mask = cols < self.lengths[index[:, 0], None]
        rows, cols = index[:, :1], cols.clamp(max=self.mask.shape[1] - 1)
        x_lead, state = self.x_lead[rows, cols], self.state[rows, cols]
        return {
            "x_lead": x_lead - x_lead[:, window - 1 : window],
            "v_lead": self.v_lead[rows, cols],
            "s": state[..., 0],
            "v": state[..., 2],
            "mask": mask,
        }


def spacing_band(
    s: np.ndarray, dv: np.ndarray, v: np.ndarray, a: np.ndarray, cfg: BandConfig | None = None,
    speeds: Sequence[float] = BAND_SPEEDS,
) -> dict[str, Any] | None:  # fmt: skip
    """Quantiles of the spacing of the near-steady samples (``|dv| < dv_max``, ``|a| < a_max``)
    within ``half_width`` of every speed of ``speeds``. Speeds with fewer than ``min_samples``
    samples are left out; ``None`` with fewer than two speeds. Plain lists and numbers, so that
    the band survives a JSON round trip unchanged."""
    cfg = cfg or BandConfig()
    steady = (np.abs(dv) < cfg.dv_max) & (np.abs(a) < cfg.a_max)
    s, v = s[steady], v[steady]
    rows = []
    for speed in speeds:
        near = s[np.abs(v - speed) < cfg.half_width]
        if near.size >= cfg.min_samples:
            rows.append((float(speed), *np.quantile(near, cfg.quantiles).tolist(), int(near.size)))
    if len(rows) < 2:
        return None
    columns = list(zip(*rows))
    keys = ("v", "s_low", "s_median", "s_high", "n")
    settings = {**asdict(cfg), "quantiles": list(cfg.quantiles)}
    return {**{k: list(c) for k, c in zip(keys, columns)}, "settings": settings}


def training_context(events: Sequence[Event], seed: int, band: BandConfig | None = None) -> dict[str, Any]:
    """Statistics of the training samples: scaler values and state box of ``ModelContext``
    (``center, scale, box_low, box_high, seed``), the variances of ``s`` and ``v`` and the
    spacing band of the data (:func:`spacing_band`)."""
    s, v, v_lead, a = (np.concatenate([getattr(ev, name) for ev in events]) for name in ("s", "v", "v_lead", "a"))
    states = np.stack((s, v - v_lead, v), axis=1)
    low, high = np.quantile(states, (0.01, 0.99), axis=0)
    return {
        "center": states.mean(axis=0).tolist(),
        "scale": states.std(axis=0).tolist(),
        "box_low": low.tolist(),
        "box_high": high.tolist(),
        "var_s": float(s.var()),
        "var_v": float(v.var()),
        "n_samples": int(s.size),
        "band": spacing_band(s, v - v_lead, v, a, band),
        "seed": int(seed),
    }
