"""Stability audit (cf_stability/stability/audit.py) and scripts/audit_stability.py."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

from cf_stability.data.schema import DT, Event, EventSet
from cf_stability.models import GRU, IDM, MLP, OVM, save_model
from cf_stability.models.base import CFModel, ModelContext
from cf_stability.models.idm import idm_acc, idm_equilibrium_spacing
from cf_stability.stability.audit import BAND_CATEGORIES, AuditConfig, audit_model, summarise_audit
from cf_stability.stability.frequency import FLAGS, FrequencyConfig
from cf_stability.train.tensors import BandConfig, training_context
from cf_stability.utils import REPO_ROOT, config_hash, read_json, write_json

CALIBRATED = {"v0": 29.57, "T": 0.676, "s0": 2.015, "a": 0.351, "b": 0.5}  # global calibration on highD
CONTEXT = {"box_low": [6.9, -2.6, 4.7], "box_high": [72.9, 3.4, 26.8], "center": [28.9, 0.08, 20.6], "scale": [14.4, 1.06, 5.14]}
SCALER = ModelContext(center=(30.0, 0.0, 17.0), scale=(15.0, 1.0, 7.0))
STATE_BOX = (torch.tensor([3.0, -2.0, 3.0], dtype=torch.float64), torch.tensor([100.0, 2.0, 30.0], dtype=torch.float64))
FULL = AuditConfig(device="cpu")
SMALL = AuditConfig(
    device="cpu",
    speeds=(8.0, 15.0, 22.0),
    frequency=FrequencyConfig(n_omega=4, omega_min=0.1, omega_max=1.0, min_discard_s=20.0, min_measure_s=20.0),
)
SMALL_OVERRIDES = (
    "stability.device=cpu", "stability.speeds=[10.0,20.0]", "stability.frequency.n_omega=3",
    "stability.frequency.omega_min=0.2", "stability.frequency.min_discard_s=20.0",
    "stability.frequency.min_measure_s=20.0",
)  # fmt: skip

TOP_KEYS = {
    "model", "window", "differentiable", "analytic_gain", "device", "dtype", "config", "support_v", "omega",
    "equilibria", "summary", "wall_time_s",
}  # fmt: skip
RECORD_KEYS = {
    "v", "s", "status", "n_crossings", "in_support", "in_band", "band_low", "band_high", "f_s", "f_dv", "f_v",
    "local_stable", "rational", "margin", "string_stable", "band_upper", "gain", "phase", "residual", "clipped",
    "stopped", "collided", "max_gain", "omega_at_max", "unstable", "analytic_max_gain", "analytic_unstable", "marginal",
}  # fmt: skip
SUMMARY_KEYS = {
    "n_grid", "n_equilibria", "n_multiple", "n_none", "n_indifferent", "n_usable", "n_in_support", "n_marginal",
    "share_unstable_numerical", "share_unstable_analytic_gain", "share_unstable_sign", "agreement_gain",
    "agreement_sign", "agreement_sign_nonmarginal", "share_locally_unstable", "share_non_rational", "n_clipped",
    "n_stopped", "n_collided", "max_gain", "max_gain_omega", "max_gain_v", "grid_numerical", "grid_sign",
    "n_band", "band_numerical", "band_sign",
}  # fmt: skip


def calibrated_spacing(v: float) -> float:
    v = torch.tensor(v, dtype=torch.float64)
    return idm_equilibrium_spacing(v, CALIBRATED["v0"], CALIBRATED["T"], CALIBRATED["s0"]).item()


def band_context(inside: list[float], above: list[float]) -> dict[str, Any]:
    """CONTEXT with a band listed at ``inside`` (s_e -+ 1 m of the calibrated IDM) and ``above``
    (s_e + 10 m .. s_e + 20 m) speeds."""
    speeds = sorted(inside + above)
    low = [calibrated_spacing(v) + (-1.0 if v in inside else 10.0) for v in speeds]
    high = [s + (2.0 if v in inside else 10.0) for s, v in zip(low, speeds)]
    median = [0.5 * (lo + hi) for lo, hi in zip(low, high)]
    band = {"v": speeds, "s_low": low, "s_median": median, "s_high": high, "n": [300] * len(speeds)}
    return {**CONTEXT, "band": band}


def fitted_to_idm(build: Callable[[], CFModel], steps: int, lr: float) -> CFModel:
    """A model after ``steps`` Adam steps on random constant histories with calibrated IDM accelerations as targets."""
    torch.manual_seed(0)
    model = build().double()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    low, high = STATE_BOX
    for _ in range(steps):
        state = low + (high - low) * torch.rand(1024, 3, dtype=torch.float64)
        target = idm_acc(state[:, 0], state[:, 1], state[:, 2], **CALIBRATED).clamp(-8.0, 4.0)
        loss = ((model(state[:, None].expand(-1, model.window, -1)) - target) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model


def usable(audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in audit["equilibria"] if r["status"] != "none"]


@pytest.fixture(scope="module")
def idm_audit() -> dict[str, Any]:
    return audit_model(IDM(CALIBRATED), CONTEXT, FULL)


@pytest.fixture(scope="module")
def mlp_audit() -> dict[str, Any]:
    return audit_model(fitted_to_idm(lambda: MLP(SCALER), steps=500, lr=3e-3), CONTEXT, FULL)


def test_calibrated_idm_numerical_flag_matches_the_analytic_flags(idm_audit):
    records = usable(idm_audit)
    assert [r["v"] for r in records] == [float(v) for v in range(5, 30)]  # no equilibrium at v >= v0
    assert all(r["unstable"] == r["analytic_unstable"] for r in records)
    assert all(r["unstable"] == (r["margin"] < 0) for r in records if not r["marginal"])
    assert any(r["unstable"] for r in records) and not all(r["unstable"] for r in records)
    assert all(abs(r["max_gain"] - r["analytic_max_gain"]["discrete"]) < 5e-3 for r in records)


def test_trained_mlp_numerical_and_analytic_gains_agree(mlp_audit):
    records = usable(mlp_audit)
    assert len(records) >= 20
    agreement = sum(r["unstable"] == r["analytic_unstable"] for r in records) / len(records)
    assert agreement > 0.95 and mlp_audit["summary"]["agreement_gain"]["all"] == pytest.approx(agreement)
    for r in records:
        exact = r["analytic_max_gain"]["discrete"]
        assert abs(r["max_gain"] - exact) <= 0.02 * exact


@pytest.mark.parametrize("name", ["idm", "ovm", "mlp", "gru"])
def test_audit_keys_json_and_summary(name, request):
    if name in ("idm", "mlp"):
        audit = request.getfixturevalue(f"{name}_audit")
    elif name == "ovm":
        audit = audit_model(OVM(ModelContext(ovm_params={"v0": 30.0, "tau": 1.2, "s0": 3.0})), None, SMALL)
    else:
        gru = fitted_to_idm(lambda: GRU(SCALER, hidden_size=8, window=5), steps=150, lr=1e-2)
        audit = audit_model(gru, CONTEXT, SMALL)
        assert audit["analytic_gain"] == "windowed"
        assert all(r["analytic_max_gain"]["windowed"] is not None for r in usable(audit))
    json.dumps(audit, allow_nan=False)
    assert set(audit) == TOP_KEYS and all(set(r) == RECORD_KEYS for r in audit["equilibria"])
    summary = summarise_audit(audit)
    assert summary == audit["summary"] and set(summary) == SUMMARY_KEYS

    records, statuses = usable(audit), [r["status"] for r in audit["equilibria"]]
    assert records, "the audited model needs equilibria"
    assert summary["n_grid"] == len(audit["equilibria"])
    assert summary["n_equilibria"] == statuses.count("ok") + statuses.count("multiple") + statuses.count("outside")
    # no band in the context: no record has one, and the band shares are None
    assert all(r["in_band"] is r["band_low"] is r["band_high"] is None for r in audit["equilibria"])
    assert summary["band_numerical"] == summary["band_sign"] == {"all": None, "support": None}
    assert summary["n_band"] == {"all": 0, "support": None if audit["support_v"] is None else 0}
    assert (summary["n_none"], summary["n_indifferent"]) == (statuses.count("none"), statuses.count("indifferent"))
    assert summary["n_usable"] == len(records)
    for key, flag in (("grid_numerical", "unstable"), ("grid_sign", None)):
        for part, rows in (("all", audit["equilibria"]), ("support", [r for r in audit["equilibria"] if r["in_support"]])):
            grid = summary[key][part]
            if part == "support" and audit["support_v"] is None:
                assert grid is None
                continue
            assert sum(grid.values()) == pytest.approx(1.0)  # the categories partition the grid speeds
            found = [r for r in rows if r["status"] in ("ok", "multiple", "outside")]
            unstable = [r[flag] if flag else r["margin"] < 0 for r in found]
            assert grid["unstable"] == pytest.approx(sum(unstable) / len(rows))
            assert grid["none"] == pytest.approx(sum(r["status"] == "none" for r in rows) / len(rows))
    shares = {
        "share_unstable_numerical": [r["unstable"] for r in records],
        "share_unstable_analytic_gain": [r["analytic_unstable"] for r in records],
        "share_unstable_sign": [r["margin"] < 0 for r in records],
        "agreement_gain": [r["unstable"] == r["analytic_unstable"] for r in records],
    }
    for key, flags in shares.items():
        assert summary[key]["all"] == pytest.approx(sum(flags) / len(flags))
    assert summary["max_gain"] == max(r["max_gain"] for r in records)
    for r in audit["equilibria"]:
        if r["status"] == "none":
            assert r["s"] is None and r["gain"] is None and r["unstable"] is None
        else:
            assert len(r["gain"]) == len(audit["omega"]) and r["max_gain"] == max(r["gain"])
    if name in ("ovm",):
        assert summary["n_in_support"] is None and summary["share_unstable_sign"]["support"] is None
        assert summary["share_unstable_sign"]["all"] == 1.0  # string unstable on the whole congested branch
    else:
        support = [r for r in records if CONTEXT["box_low"][2] <= r["v"] <= CONTEXT["box_high"][2]]
        assert summary["n_in_support"] == len(support) == sum(bool(r["in_support"]) for r in records)


def test_audit_with_band():
    # band listed at 8 m/s (holds the equilibrium) and 15 m/s (above it); 22 m/s has no band
    context = band_context(inside=[8.0], above=[15.0])
    audit = audit_model(IDM(CALIBRATED), context, SMALL)
    json.dumps(audit, allow_nan=False)
    assert set(audit) == TOP_KEYS and all(set(r) == RECORD_KEYS for r in audit["equilibria"])
    records, band = audit["equilibria"], context["band"]
    assert [r["status"] for r in records] == ["ok", "outside", "ok"]
    assert [r["in_band"] for r in records] == [True, False, None]
    assert [r["band_low"] for r in records] == [*band["s_low"], None]
    assert [r["band_high"] for r in records] == [*band["s_high"], None]
    for r in records:  # every equilibrium is analysed, the one outside the band as well
        assert r["s"] == pytest.approx(calibrated_spacing(r["v"]), rel=1e-9) and r["max_gain"] is not None
    summary = audit["summary"]
    assert summary == summarise_audit(audit) and summary["n_band"] == {"all": 2, "support": 2}
    assert summary["n_equilibria"] == 3 and summary["n_none"] == 0
    for key, flag in (("band_numerical", "unstable"), ("band_sign", None)):
        for part in ("all", "support"):
            shares = summary[key][part]
            assert set(shares) == set(BAND_CATEGORIES) and sum(shares.values()) == pytest.approx(1.0)
            unstable = records[0][flag] if flag else records[0]["margin"] < 0
            assert (shares["stable"], shares["unstable"]) == ((0.0, 0.5) if unstable else (0.5, 0.0))
            assert [shares[key] for key in ("outside", "none", "indifferent", "undefined")] == [0.5, 0.0, 0.0, 0.0]
        # the shares of the grid count any equilibrium, the one outside the band too
        assert summary[key.replace("band", "grid")]["all"]["none"] == 0.0
        grid = summary[key.replace("band", "grid")]["all"]
        assert grid["stable"] + grid["unstable"] == pytest.approx(1.0)


def hand_record(
    v: float, status: str, in_support: bool, in_band: bool | None, unstable: bool | None, string_stable: bool | None
) -> dict[str, Any]:
    """The keys of an audit record that ``summarise_audit`` reads."""
    found = status != "none"
    return {
        "v": v, "status": status, "in_support": in_support, "in_band": in_band, "unstable": unstable,
        "string_stable": string_stable, "analytic_unstable": unstable, "marginal": False if found else None,
        "local_stable": True if found else None, "rational": True if found else None,
        "max_gain": 1.0 + v / 100.0 if found else None, "omega_at_max": 0.1 if found else None,
        **{name: [False] if found else None for name in FLAGS},
    }  # fmt: skip


HAND_RECORDS = [
    hand_record(5.0, "none", False, None, None, None),
    hand_record(6.0, "ok", True, True, False, True),  # band: stable (numerical and sign)
    hand_record(7.0, "ok", True, True, True, False),  # band: unstable
    hand_record(8.0, "multiple", True, True, None, True),  # band: undefined numerical flag, stable sign
    hand_record(9.0, "outside", True, False, True, False),  # band: outside, whatever its flags
    hand_record(10.0, "none", True, False, None, None),  # band: none
    hand_record(11.0, "indifferent", True, True, False, True),  # band: indifferent
    hand_record(12.0, "ok", False, True, True, False),  # outside the support: unstable in "all" only
    hand_record(13.0, "ok", True, None, False, True),  # no band at this speed: in the grid shares only
]


def test_summary_of_hand_made_records():
    summary = summarise_audit({"support_v": [6.0, 13.0], "equilibria": HAND_RECORDS})
    assert summary["n_band"] == {"all": 7, "support": 6} and summary["n_equilibria"] == 6
    shares = dict(zip(BAND_CATEGORIES, (1, 2, 1, 1, 1, 1)))  # stable, unstable, outside, none, indifferent, undefined
    assert summary["band_numerical"]["all"] == pytest.approx({k: n / 7 for k, n in shares.items()})
    assert summary["band_numerical"]["support"] == pytest.approx({k: 1 / 6 for k in BAND_CATEGORIES})
    shares = dict(zip(BAND_CATEGORIES, (2, 2, 1, 1, 1, 0)))
    assert summary["band_sign"]["all"] == pytest.approx({k: n / 7 for k, n in shares.items()})
    shares = dict(zip(BAND_CATEGORIES, (2, 1, 1, 1, 1, 0)))
    assert summary["band_sign"]["support"] == pytest.approx({k: n / 6 for k, n in shares.items()})
    # grid shares: any equilibrium, also outside the band (stable 6, 13; unstable 7, 9, 12; undefined 8)
    grid = {"stable": 2 / 9, "unstable": 3 / 9, "none": 2 / 9, "indifferent": 1 / 9, "undefined": 1 / 9}
    assert summary["grid_numerical"]["all"] == pytest.approx(grid)
    grid = {"stable": 2 / 7, "unstable": 2 / 7, "none": 1 / 7, "indifferent": 1 / 7, "undefined": 1 / 7}
    assert summary["grid_numerical"]["support"] == pytest.approx(grid)
    for key in ("band_numerical", "band_sign", "grid_numerical", "grid_sign"):
        for part in ("all", "support"):
            assert sum(summary[key][part].values()) == pytest.approx(1.0)

    no_support = summarise_audit({"support_v": None, "equilibria": HAND_RECORDS})
    assert no_support["n_band"] == {"all": 7, "support": None} and no_support["band_numerical"]["support"] is None
    assert no_support["band_numerical"]["all"] == summary["band_numerical"]["all"]
    # records without band (a context without band, or an audit written before M4)
    without_band = [{**r, "in_band": None} for r in HAND_RECORDS]
    before_m4 = [{k: x for k, x in r.items() if k != "in_band"} for r in HAND_RECORDS]
    for records in (without_band, before_m4):
        old = summarise_audit({"support_v": [6.0, 13.0], "equilibria": records})
        assert old["n_band"] == {"all": 0, "support": 0} and old["band_sign"] == {"all": None, "support": None}
        assert old["grid_numerical"] == summary["grid_numerical"]


def quoted(path: Path) -> str:
    return f"'{path.as_posix()}'"  # hydra override grammar: non-ASCII paths must be quoted


def run_script(tmp_path: Path, *overrides: str) -> list[str]:
    args = [
        sys.executable, str(REPO_ROOT / "scripts" / "audit_stability.py"), *overrides, *SMALL_OVERRIDES,
        f"paths.runs_root={quoted(tmp_path / 'runs')}", f"hydra.run.dir={quoted(tmp_path / 'outputs')}",
    ]  # fmt: skip
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HYDRA_FULL_ERROR": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        args, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout.strip().splitlines()


def test_audit_script(tmp_path):
    runs = tmp_path / "runs" / "exp" / "synthetic"
    good, broken = runs / "idm" / "driver_fold0_seed0", runs / "broken" / "driver_fold0_seed0"
    banded = runs / "idm_band" / "driver_fold0_seed0"
    for run in (good, broken, banded):
        run.mkdir(parents=True)
    for run, context in ((good, CONTEXT), (banded, band_context(inside=[10.0], above=[20.0]))):
        save_model(IDM(CALIBRATED), run / "model.pt")
        write_json(run / "metrics.json", {"context": context})
    (broken / "model.pt").write_bytes(b"not a checkpoint")

    lines = run_script(tmp_path, "experiment=exp", "data=synthetic")
    assert len(lines) == 3 and "FAILED" in lines[0] and lines[1].startswith("idm/driver_fold0_seed0")
    assert "band n/a " in lines[1]  # metrics.json without band
    # one speed in support holds its equilibrium in the band, the other has it outside
    shares = re.search(r"band (\d+)/(\d+)/(\d+)/(\d+) % \(stable/unstable/outside/none, support\)", lines[2])
    assert lines[2].startswith("idm_band/driver_fold0_seed0") and shares is not None, lines[2]
    stable, unstable, outside, none = map(int, shares.groups())
    assert stable + unstable == 50 and (outside, none) == (50, 0)
    result = read_json(good / "stability.json")
    assert result["model"] == "idm" and result["config_hash"] == config_hash(result["config"]["stability"])
    assert result["audit"]["summary"]["n_grid"] == 2 and result["audit"]["support_v"] == [4.7, 26.8]
    assert result["audit"]["config"]["frequency"]["n_omega"] == 3 and "git_revision" in result
    assert read_json(banded / "stability.json")["audit"]["summary"]["n_band"] == {"all": 2, "support": 2}
    failed = read_json(broken / "stability.json")
    assert failed["model"] is None and "error" in failed and "audit" not in failed

    assert run_script(tmp_path, "experiment=exp", "data=synthetic") == []  # audits newer than model.pt are kept
    assert len(run_script(tmp_path, "experiment=exp", "data=synthetic", "force=true")) == 3
    lines = run_script(tmp_path, f"run={quoted(good)}")
    assert len(lines) == 1 and lines[0].startswith("idm/driver_fold0_seed0")


def steady_event(i: int, v: float, offset: float, n: int = 150) -> Event:
    """A follower at the speed ``v`` of its leader whose spacing swings by 1 m around the calibrated equilibrium
    plus ``offset`` (period 15 s: |dv| <= 0.42 m/s, |a| <= 0.18 m/s^2, every sample near-steady, D78)."""
    t = DT * np.arange(n)
    omega, phase = 2.0 * np.pi / 15.0, 0.7 * i
    s = calibrated_spacing(v) + offset + np.sin(omega * t + phase)
    v_lead = np.full(n, v)
    x_lead = 50.0 + v * t
    v_f = v_lead - omega * np.cos(omega * t + phase)  # ds/dt = v_lead - v
    return Event(
        event_id=f"synthetic/a/{i}|L|0", dataset="synthetic", site="a", follower_id=f"synthetic/a/{i}",
        leader_id=f"synthetic/a/L{i}", t=t, s=s, dv=v_f - v_lead, v=v_f, a=omega**2 * np.sin(omega * t + phase),
        v_lead=v_lead, x_lead=x_lead, x_follower=x_lead - s,
    )  # fmt: skip


def test_audit_script_with_the_band_of_other_quantiles(tmp_path):
    """D119: band_quantiles rebuilds the training part of the run from its metrics.json (data, split, fold), computes
    the band with the run's band settings and the given quantiles (tensors.training_context) and writes
    stability_q<low>_<high>.json; with 5/95 it is the band of the context, and the audit that of stability.json."""
    events = [steady_event(i, v, offset) for i, (v, offset) in enumerate(
        (v, offset) for offset in (-3.0, -1.5, 0.0, 1.5, 3.0) for v in (10.0, 20.0)
    )]  # fmt: skip
    EventSet(events).to_parquet(tmp_path / "events" / "synthetic")
    folds = {ev.event_id: i // 2 for i, ev in enumerate(events)}  # one event per speed and fold: training 2, 3, 4
    split = {"kind": "driver", "n_folds": 5, "seed": 0, "folds": folds}
    write_json(tmp_path / "splits" / "synthetic_driver.json", split)
    settings = {"dv_max": 0.5, "a_max": 0.3, "half_width": 0.5, "quantiles": [0.05, 0.5, 0.95], "min_samples": 20}
    train = [ev for ev in events if folds[ev.event_id] >= 2]
    band = training_context(train, 0, BandConfig.from_mapping(settings))["band"]
    assert band["v"] == [10.0, 20.0] and band["n"] == [450, 450]
    config = {
        "data": {"name": "synthetic"}, "split": "driver", "fold": 0, "seed": 3, "max_train_events": None,
        "paths": {"events_root": (tmp_path / "events").as_posix(), "splits_root": (tmp_path / "splits").as_posix()},
        "train": {"band": settings},
    }  # fmt: skip
    run = tmp_path / "runs" / "exp" / "synthetic" / "idm" / "driver_fold0_seed3"
    run.mkdir(parents=True)
    save_model(IDM(CALIBRATED), run / "model.pt")
    metrics = {"config": config, "train_config": {"band": settings}, "context": {**CONTEXT, "band": band}}
    write_json(run / "metrics.json", metrics)
    run_script(tmp_path, f"run={quoted(run)}")
    reference = read_json(run / "stability.json")
    run_script(tmp_path, f"run={quoted(run)}", "band_quantiles=[0.05,0.5,0.95]")
    same = read_json(run / "stability_q05_95.json")
    assert same["band_override"]["band"] == band and same["band_override"]["settings"]["min_samples"] == 20
    assert same["audit"]["equilibria"] == reference["audit"]["equilibria"]  # the band of D78: the same audit
    hashed = {"stability": same["config"]["stability"], "band_quantiles": [0.05, 0.5, 0.95]}
    assert same["config_hash"] == config_hash(hashed)

    lines = run_script(tmp_path, "experiment=exp", "data=synthetic", "band_quantiles=[0.01,0.5,0.99]")
    assert len(lines) == 1 and lines[0].startswith("idm/driver_fold0_seed3")
    wide = read_json(run / "stability_q01_99.json")
    expected = training_context(train, 0, BandConfig.from_mapping({**settings, "quantiles": [0.01, 0.5, 0.99]}))["band"]
    assert wide["band_override"]["band"] == expected and wide["band_override"]["quantiles"] == [0.01, 0.5, 0.99]
    assert all(a < b for a, b in zip(expected["s_low"], band["s_low"]))
    assert all(a > b for a, b in zip(expected["s_high"], band["s_high"]))
    records = wide["audit"]["equilibria"]
    assert [r["band_low"] for r in records] == pytest.approx(expected["s_low"], rel=1e-12)
    assert read_json(run / "stability.json") == reference  # the reference audit stays as it is
    # experiment mode skips the runs whose audit of this band is newer than model.pt
    assert run_script(tmp_path, "experiment=exp", "data=synthetic", "band_quantiles=[0.01,0.5,0.99]") == []
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "audit_stability.py"), f"run={quoted(run)}",
         "band_quantiles=[0.9,0.5,0.1]", f"hydra.run.dir={quoted(tmp_path / 'outputs')}"],
        cwd=tmp_path, env={**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=600,
    )  # fmt: skip
    assert proc.returncode != 0 and "three increasing numbers" in proc.stdout + proc.stderr
