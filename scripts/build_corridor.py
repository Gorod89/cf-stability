"""Scenarios of the corridor and export of its laws (docs/m5_contract.md, sections 2-3; docs/m7_contract.md, 4-7).

    python scripts/build_corridor.py                      # scenarios i80_p0..p2, us101_p0..p2 and all law files
    python scripts/build_corridor.py scenarios=false      # only the law files (again, e.g. after new member runs)
    python scripts/build_corridor.py laws=false sites=[us101] periods=[1] force=true
    python scripts/build_corridor.py laws=false periods=[1] variant=gain0.01 corridor.i80.boundary.gain=0.01

Scenarios: ``<paths.scenarios_root>/<name>_p<period>/`` for the sites of ``sites`` (``corridor.<site>``,
``configs/corridor/<site>.yaml``) with the SUMO network (netconvert as a child process), the routes, the
boundary speeds, the ground truth and ``scenario.json`` (``cf_stability/corridor/scenario.py``). A scenario
whose ``scenario.json`` holds the hash of the current config (and its ``metrics`` settings) is kept unless
``force=true``. Every scenario gets a line ``TRUTH``: the vehicles of the period and the throughput, mean
speed, wave speed by cross-correlation and number of waves of its ground truth in the analysis window (the
metric code of ``scripts/corridor_metrics.py``, which writes the ``macro.json`` files).

Variants (D113): ``variant=<name>`` with overrides of ``corridor.<site>.<key>`` on the command line builds
``<name>_p<period>_<variant>`` for the sites that those overrides change (and no other); ``scenario.json``
records the variant and its overrides (keys relative to the site). A key that the site config does not have
needs ``+`` (``+corridor.i80.vtype.lcSpeedGain=0.5``).

Laws: ``<paths.laws_root>/<law>.json`` (+ ``.npz``) for the laws of ``configs/corridor/laws.yaml`` whose
member runs all exist (``cf_stability/corridor/laws.py``); a law with a missing member run is not written
and its line names the missing runs. One printed line per scenario and law.

This is the build process of section 0: pandas and pyarrow, never libsumo.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cf_stability.corridor.laws import export_laws  # noqa: E402
from cf_stability.corridor.macro import (  # noqa: E402
    MacroConfig,
    corridor_metrics,
    geometry_from_scenario,
    load_npz,
    scenario_macro_config,
)
from cf_stability.corridor.scenario import (  # noqa: E402
    Corridor,
    build_scenario,
    read_tracks,
    scenario_config,
    scenario_metrics,
    scenario_name,
)
from cf_stability.utils import config_hash, read_json, resolve_path, to_plain  # noqa: E402


def is_current(directory: Path, expected_hash: str, metrics: Mapping[str, Any] | None = None) -> bool:
    """``scenario.json`` holds ``expected_hash`` and the settings of the metrics of the site now."""
    path = directory / "scenario.json"
    try:
        if not path.exists():
            return False
        payload = read_json(path)
        return payload.get("config_hash") == expected_hash and payload.get("metrics") == metrics
    except (OSError, ValueError):
        return False


def truth_line(directory: Path) -> str:
    """The ground truth of a scenario inside its analysis window: vehicles (entering on the auxiliary lane, routed
    to the off-ramp), throughput, mean speed, wave speed by cross-correlation and number of waves, computed as
    ``scripts/corridor_metrics.py`` does with the defaults of ``MacroConfig`` (those of
    ``configs/corridor_metrics.yaml``) and the scenario's ``metrics`` settings; no file is written (``macro.json``
    is the metrics script's)."""
    meta = read_json(directory / "scenario.json")
    cfg = scenario_macro_config(MacroConfig(), meta)
    vehicles = load_npz(directory / "vehicles_truth.npz")
    m = corridor_metrics(load_npz(directory / "ground_truth.npz"), vehicles, geometry_from_scenario(meta), cfg)
    corridor = Corridor.from_mapping(meta["config"])
    on_aux = int((corridor.network_lane(vehicles["entry_lane"].astype(int)) == corridor.aux_lane).sum())
    waves, (w0, w1) = m["waves"], m["window"]

    def number(value: float | None, spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    return (
        f"TRUTH {meta['scenario']}: {len(vehicles['vehicle_id'])} vehicles ({on_aux} enter on the auxiliary lane, "
        f"{meta['counts'].get('offramp', 0)} routed to the off-ramp); window {w0:g}-{w1:g} s: throughput "
        f"{number(m['throughput_vph'], '.0f')} veh/h at {cfg.throughput_x:g} m, mean speed {number(m['mean_speed'], '.2f')} "
        f"m/s, wave speed {number(waves['wave_speed_xcorr'], '.2f')} m/s (cross-correlation), {waves['n_waves']} waves"
    )


def site_overrides(task_overrides: Sequence[str], sites: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The command-line overrides of ``corridor.<site>.<key>`` of the ``sites``: ``{site: {key: value}}``."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    parser = OverridesParser.create()
    out: dict[str, dict[str, Any]] = {}
    for item in task_overrides:
        override = parser.parse_override(item)
        parts = override.key_or_group.split(".")
        if len(parts) > 2 and parts[0] == "corridor" and parts[1] in sites:
            out.setdefault(parts[1], {})[".".join(parts[2:])] = to_plain(override.value())
    return out


def build_scenarios(
    cfg: DictConfig, site_key: str, variant: str | None = None, overrides: Mapping[str, Any] | None = None
) -> None:
    site = to_plain(cfg.corridor[site_key])
    if not site.get("enabled", True):
        print(f"SCENARIO corridor.{site_key} disabled")
        return
    periods = [int(p) for p in (cfg.periods if cfg.periods is not None else site["periods"])]
    root = resolve_path(cfg.paths.scenarios_root)
    metrics = scenario_metrics(site)
    todo = []
    for period in periods:
        directory = root / scenario_name(site["name"], period, variant)
        expected = config_hash(scenario_config(site, period, variant))
        if not cfg.force and is_current(directory, expected, metrics):
            print(f"SCENARIO {directory.name}: current (config hash {expected})")
            print(truth_line(directory), flush=True)
        else:
            todo.append((period, directory))
    if not todo:
        return
    t_start = time.perf_counter()
    periods = [period for period, _ in todo]
    tracks = read_tracks(resolve_path(site["source"]), periods)
    seconds = time.perf_counter() - t_start
    print(f"SCENARIO data {site['name']}: {len(tracks)} samples of periods {periods} ({seconds:.0f} s)")
    for period, directory in todo:
        t_start = time.perf_counter()
        meta = build_scenario(tracks, site, period, directory, variant, overrides)
        counts = meta["counts"]
        lanes = " ".join(f"{lane}:{n}" for lane, n in counts["entry_lane"].items())
        ramp = f", {counts['offramp']} routed to the off-ramp" if meta["geometry"].get("offramp_edge") else ""
        made = f", variant {variant} {meta['overrides']}" if variant else ""
        print(
            f"SCENARIO {meta['scenario']}: {counts['vehicles']} vehicles of {counts['tracks']} tracks "
            f"(entry lanes {lanes}; {counts['entered_downstream_of_x_in']} enter downstream of x_in{ramp}), last "
            f"departure {meta['t_last_depart']:.1f} s, exits {counts['exited']} ({counts['exits_extrapolated']} "
            f"extrapolated), {counts['vehicle_types']} vehicle types, {counts['truth_rows']} truth rows{made}, hash "
            f"{meta['config_hash']} -> {directory} ({time.perf_counter() - t_start:.0f} s)"
        )
        print(truth_line(directory), flush=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="build_corridor")
def main(cfg: DictConfig) -> None:
    if cfg.corridor.i24.enabled:
        raise SystemExit("corridor.i24.enabled=true: the I-24 MOTION data are not on this machine (D101)")
    if cfg.scenarios:
        known = [key for key, value in cfg.corridor.items() if isinstance(value, Mapping) and "edges" in value]
        overrides = site_overrides(HydraConfig.get().overrides.task, known)
        if cfg.variant:
            if not overrides:
                raise SystemExit(
                    f"variant={cfg.variant}: a variant needs overrides of corridor.<site>.<key> on the command line "
                    f"(sites {known}), e.g. corridor.i80.boundary.gain=0.01 (D113)"
                )
            for key in sorted(overrides):
                build_scenarios(cfg, key, str(cfg.variant), overrides[key])
        else:
            for key in cfg.sites:
                build_scenarios(cfg, str(key))
    if cfg.laws:
        for line in export_laws(to_plain(cfg.corridor.laws), resolve_path(cfg.paths.laws_root)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
