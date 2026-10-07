"""Stability audit of a car-following model (docs/m3_contract.md, section 5; band: docs/m4_contract.md, 1.4).

Per speed of the grid: equilibrium (anchored to the spacing band of the training data when the
context has one), partial derivatives and the analytic criterion, the analytic gains on the
frequencies of the numerical response (continuous, discrete and, for windowed models, the
linearised window response) and the numerical frequency response. The "analytic gain" flag uses
the exact linearisation of the simulated law: the windowed response for windowed
(differentiable) models, the discrete one otherwise.
"""

from __future__ import annotations

import copy
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from cf_stability.models.base import CFModel
from cf_stability.stability.analytic import (
    criterion,
    history_jacobian,
    partials,
    transfer_continuous,
    transfer_discrete,
    transfer_windowed,
)
from cf_stability.stability.equilibrium import V_GRID, Band, find_equilibria, model_like
from cf_stability.stability.frequency import FLAGS, FrequencyConfig, frequency_response
from cf_stability.train.calibration import resolve_device

PARTIALS = ("f_s", "f_dv", "f_v")
CRITERION_FLAGS = ("local_stable", "rational", "string_stable")
GAINS = ("continuous", "discrete", "windowed")
V_INDEX = 2  # position of the speed in the state (s, dv, v)
FOUND = ("ok", "multiple", "outside")  # statuses with an equilibrium, inside or outside the band
BAND_CATEGORIES = ("stable", "unstable", "outside", "none", "indifferent", "undefined")
# spacing bands of other quantiles of the near-steady training samples (D119): name -> (low, median, high);
# the reference band of D78 is (0.05, 0.5, 0.95), whose audit is stability.json
BAND_VARIANTS: dict[str, tuple[float, float, float]] = {"q01_99": (0.01, 0.5, 0.99), "q10_90": (0.10, 0.5, 0.90)}


def band_variant(quantiles: Sequence[float]) -> str:
    """Name of the band of the quantiles ``(low, median, high)``: ``q01_99`` for (0.01, 0.5, 0.99) (D119)."""
    low, _, high = (float(q) for q in quantiles)
    return f"q{round(100 * low):02d}_{round(100 * high):02d}"


def band_output(quantiles: Sequence[float]) -> str:
    """Output file of an audit with the spacing band of other quantiles: ``stability_q01_99.json`` (D119)."""
    return f"stability_{band_variant(quantiles)}.json"


@dataclass
class AuditConfig:
    """Options of the audit (``configs/stability/default.yaml``)."""

    device: str = "auto"  # cuda when available
    speeds: tuple[float, ...] | None = None  # equilibrium speeds in m/s; None: V_GRID
    marginal_tol: float = 0.02  # marginal equilibrium: analytic maximum gain within this distance of 1
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AuditConfig":
        values = dict(mapping)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown audit keys: {sorted(unknown)}")
        values["frequency"] = FrequencyConfig.from_mapping(values.get("frequency") or {})
        if values.get("speeds") is not None:
            values["speeds"] = tuple(float(v) for v in values["speeds"])
        return cls(**values)


def _num(x: Any) -> float | None:
    """Finite float, else None (NaN is written as null)."""
    x = float(x)
    return x if math.isfinite(x) else None


def _nums(row: Tensor) -> list[float | None]:
    return [_num(x) for x in row.tolist()]


def _analytic(model: CFModel, s: Tensor, v: Tensor, omega: Tensor, windowed: bool) -> dict[str, Tensor]:
    """Partials, criterion and the analytic gains ``|G| [n, n_w]`` at the equilibria ``(s, v)``."""
    f = partials(model, s, v)
    out = {**dict(zip(PARTIALS, f)), **criterion(*f)}
    out["continuous"] = transfer_continuous(*f, omega).abs()
    out["discrete"] = transfer_discrete(*f, omega, model.dt).abs()
    if windowed:
        out["windowed"] = transfer_windowed(history_jacobian(model, s, v), omega, model.dt).abs()
    return out


def audit_model(model: CFModel, context: Mapping[str, Any] | None, cfg: AuditConfig | None = None) -> dict[str, Any]:
    """Equilibria, analytic criterion and numerical frequency response of ``model`` on the speed grid.

    The model is deep-copied, cast to float64 and put in eval mode on ``cfg.device``. ``context``
    (the ``context`` of ``metrics.json``) gives the support: speeds within the 1 % - 99 % range
    ``box_low[2] .. box_high[2]`` of the training states; without it ``in_support`` is None. Its
    ``band`` (absent or None: no band) anchors the equilibria (:func:`find_equilibria`); every
    record gets ``in_band`` (None at a speed without band) and the edges ``band_low``,
    ``band_high``. Returns a JSON-serialisable dict (NaN as None) with the per-equilibrium records
    under ``equilibria`` and :func:`summarise_audit` under ``summary``.
    """
    t_start = time.perf_counter()
    cfg = cfg or AuditConfig()
    model = copy.deepcopy(model).double().to(resolve_device(cfg.device)).eval()
    device, _ = model_like(model)  # a model without tensors (persistence) stays on the CPU
    freq = cfg.frequency
    band = Band.from_context(context, model)
    eq = find_equilibria(model, cfg.speeds or V_GRID, band=band)
    index = eq.usable.nonzero().squeeze(-1)
    omega = torch.as_tensor(freq.omegas(), dtype=torch.float64, device=device)
    windowed = model.window > 1 and getattr(model, "differentiable", True)
    analytic = _analytic(model, eq.s[index], eq.v[index], omega, windowed) if len(index) else {}
    exact = analytic.get("windowed" if windowed else "discrete")
    response = frequency_response(model, eq, freq)
    support = None if context is None else (float(context["box_low"][V_INDEX]), float(context["box_high"][V_INDEX]))
    low, high = ([], []) if band is None else (x.tolist() for x in band.at(eq.v)[:2])  # edges at every speed

    records = []
    row_of = {int(i): k for k, i in enumerate(index.tolist())}
    columns = (eq.v, eq.s, eq.n_crossings, eq.has_band, eq.in_band)
    for i, (v, s, n_cross, has_band, in_band, status) in enumerate(zip(*(x.tolist() for x in columns), eq.status)):
        record: dict[str, Any] = {
            "v": v, "s": _num(s), "status": status, "n_crossings": int(n_cross),
            "in_support": None if support is None else support[0] <= v <= support[1],
            "in_band": bool(in_band) if has_band else None,
            "band_low": _num(low[i]) if has_band else None,
            "band_high": _num(high[i]) if has_band else None,
        }  # fmt: skip
        k = row_of.get(i)
        if k is None:
            record.update(dict.fromkeys((*PARTIALS, "margin", "band_upper", *CRITERION_FLAGS), None))
            record.update(dict.fromkeys(("gain", "phase", "residual", *FLAGS), None))
            record.update(dict.fromkeys(("max_gain", "omega_at_max", "unstable"), None))
            record.update(analytic_max_gain=None, analytic_unstable=None, marginal=None)
            records.append(record)
            continue
        margin = _num(analytic["margin"][k])
        record.update({name: _num(analytic[name][k]) for name in (*PARTIALS, "margin", "band_upper")})
        record.update({name: None if margin is None else bool(analytic[name][k]) for name in CRITERION_FLAGS})
        record.update({name: _nums(response[name][i]) for name in ("gain", "phase", "residual")})
        record.update({name: [bool(x) for x in response[name][i].tolist()] for name in FLAGS})
        max_gain = _num(response["max_gain"][i])
        record.update(
            max_gain=max_gain,
            omega_at_max=_num(response["omega_at_max"][i]),
            unstable=None if max_gain is None else max_gain > freq.threshold,
            analytic_max_gain={name: _num(analytic[name][k].max()) if name in analytic else None for name in GAINS},
        )
        exact_max = _num(exact[k].max())
        record["analytic_unstable"] = None if exact_max is None else exact_max > freq.threshold
        record["marginal"] = None if exact_max is None else abs(exact_max - 1.0) <= cfg.marginal_tol
        records.append(record)

    audit = {
        "model": model.name,
        "window": model.window,
        "differentiable": bool(getattr(model, "differentiable", True)),
        "analytic_gain": "windowed" if windowed else "discrete",
        "device": str(device),
        "dtype": "float64",
        "config": asdict(cfg),
        "support_v": None if support is None else list(support),
        "omega": omega.tolist(),
        "equilibria": records,
    }
    audit["summary"] = summarise_audit(audit)
    audit["wall_time_s"] = time.perf_counter() - t_start
    return audit


def _sign_unstable(record: Mapping[str, Any]) -> bool | None:
    return None if record["string_stable"] is None else not record["string_stable"]


def _agree(a: bool | None, b: bool | None) -> bool | None:
    return None if a is None or b is None else a == b


def _band_category(record: Mapping[str, Any], flag: Callable[[Mapping[str, Any]], bool | None]) -> str:
    """Category of a speed that has a band: equilibrium inside it (stable, unstable or undefined
    flag), equilibrium outside it, none, or an indifferent law."""
    if record["status"] in ("outside", "none", "indifferent"):
        return record["status"]
    value = flag(record)
    return "undefined" if value is None else "unstable" if value else "stable"


def summarise_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Counts, shares of flagged equilibria and agreement of the flags, from the records of an audit.

    Shares are taken over the usable equilibria (status other than ``none``) with a defined flag,
    for ``all`` of them and for those in ``support`` (None without a support). Agreements:
    ``gain`` (a) numerical flag vs the analytic-gain flag, ``sign`` (b) numerical flag vs
    ``margin < 0``, ``sign_nonmarginal`` (c) as (b) without the marginal equilibria. The shares of
    the grid speeds (``grid_*``, any equilibrium) and of the speeds that have a band (``band_*``,
    equilibria inside the band only) are described in the nested functions.
    """
    records = audit["equilibria"]
    usable = [r for r in records if r["status"] != "none"]
    parts = {"all": usable, "support": None if audit["support_v"] is None else [r for r in usable if r["in_support"]]}
    in_support = None if audit["support_v"] is None else [r for r in records if r["in_support"]]
    grid_parts = {"all": records, "support": in_support}
    # records of an audit without band (or written before M4) have no band: in_band None or absent
    band_parts = {
        name: None if part is None else [r for r in part if r.get("in_band") is not None]
        for name, part in grid_parts.items()
    }

    def share(value: Callable[[Mapping[str, Any]], bool | None]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for name, part in parts.items():
            values = [] if part is None else [x for x in map(value, part) if x is not None]
            out[name] = sum(values) / len(values) if values else None
        return out

    def count(value: Callable[[Mapping[str, Any]], bool | None]) -> dict[str, int | None]:
        return {name: None if part is None else sum(bool(value(r)) for r in part) for name, part in parts.items()}

    statuses = [r["status"] for r in records]
    gains = [r for r in usable if r["max_gain"] is not None]
    top = max(gains, key=lambda r: r["max_gain"]) if gains else None

    def grid_shares(flag: Callable[[Mapping[str, Any]], bool | None]) -> dict[str, dict[str, float] | None]:
        """Shares of the grid speeds that are stable, unstable, without equilibrium or indifferent.

        A law can empty the share of unstable equilibria by having no equilibrium: here a speed
        without equilibrium is its own category and the shares sum to 1. Any equilibrium counts,
        inside or outside the band.
        """
        out: dict[str, dict[str, float] | None] = {}
        for name, part in grid_parts.items():
            if part is None:
                out[name] = None
                continue
            n = max(len(part), 1)
            none = sum(r["status"] == "none" for r in part)
            indifferent = sum(r["status"] == "indifferent" for r in part)
            unstable = sum(bool(flag(r)) for r in part if r["status"] in FOUND)
            undefined = sum(flag(r) is None for r in part if r["status"] in FOUND)
            stable = len(part) - none - indifferent - unstable - undefined
            out[name] = {
                "stable": stable / n,
                "unstable": unstable / n,
                "none": none / n,
                "indifferent": indifferent / n,
                "undefined": undefined / n,
            }
        return out

    def band_shares(flag: Callable[[Mapping[str, Any]], bool | None]) -> dict[str, dict[str, float] | None]:
        """Shares of the speeds that have a band (D79): ``stable`` and ``unstable`` count equilibria
        inside the band only, ``outside`` the equilibria outside it; the six shares sum to 1. None
        for a part without such speeds (no band in the context, or no support)."""
        out: dict[str, dict[str, float] | None] = {}
        for name, part in band_parts.items():
            counts = Counter(_band_category(r, flag) for r in part or ())
            out[name] = {key: counts[key] / len(part) for key in BAND_CATEGORIES} if part else None
        return out

    return {
        "grid_numerical": grid_shares(lambda r: r["unstable"]),
        "grid_sign": grid_shares(_sign_unstable),
        "band_numerical": band_shares(lambda r: r["unstable"]),
        "band_sign": band_shares(_sign_unstable),
        "n_band": {name: None if part is None else len(part) for name, part in band_parts.items()},
        "n_grid": len(records),
        "n_equilibria": sum(statuses.count(status) for status in FOUND),
        "n_multiple": statuses.count("multiple"),
        "n_none": statuses.count("none"),
        "n_indifferent": statuses.count("indifferent"),
        "n_usable": len(usable),
        "n_in_support": None if parts["support"] is None else len(parts["support"]),
        "n_marginal": count(lambda r: r["marginal"]),
        "share_unstable_numerical": share(lambda r: r["unstable"]),
        "share_unstable_analytic_gain": share(lambda r: r["analytic_unstable"]),
        "share_unstable_sign": share(_sign_unstable),
        "agreement_gain": share(lambda r: _agree(r["unstable"], r["analytic_unstable"])),
        "agreement_sign": share(lambda r: _agree(r["unstable"], _sign_unstable(r))),
        "agreement_sign_nonmarginal": share(
            lambda r: None if r["marginal"] is not False else _agree(r["unstable"], _sign_unstable(r))
        ),
        "share_locally_unstable": share(lambda r: None if r["local_stable"] is None else not r["local_stable"]),
        "share_non_rational": share(lambda r: None if r["rational"] is None else not r["rational"]),
        **{f"n_{name}": count(lambda r, name=name: any(r[name] or ())) for name in FLAGS},
        "max_gain": None if top is None else top["max_gain"],
        "max_gain_omega": None if top is None else top["omega_at_max"],
        "max_gain_v": None if top is None else top["v"],
    }
