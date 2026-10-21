"""Tables of the two-timescale certified hybrid with a persistent latent (docs/m12_contract.md, D135, hypothesis H-N2').

    python scripts/analysis/tt2_summary.py                 # pilot, highD folds, fine-tuning, the rule of H-N2'
    python scripts/analysis/tt2_summary.py --no-latent     # without the latent on the test events (no data read)

The sibling of ``scripts/analysis/tt_summary.py`` (D129) for ``residual_idm_tt2``: the run directories of
``e4_tt2_pilot_tau{15,30,60}``, ``e4_tt2`` and ``e4_tt2_ft`` with their ``metrics.json``, ``stability.json`` (the audit of
the persistent latent: equilibria of the steady state, rollouts carrying the latent), ``stability_z{-1,-0.5,0,0.5,1}.json``
(the frozen latents of D129) and ``stability_z-0.2.json`` (the lower end of the range, for information),
``platoon.json`` and ``certificate.json`` (the a priori certificate at the 21 latents of [-0.2, 1]), read with
``tt_summary.run_row``. References of the same fold, seed 0: the GRU of E1, the certified hybrid E4 (``e4_stable``,
``e4_stable_ft``) and the two-timescale law of D129 (``e4_tt``, ``e4_tt_ft``, the window-restarted latent).

Latent on the test events: the persistent latent of the teacher-forced pass along every observed test event (from z = 0
before its first window, advanced once per observed window: ``cf_stability/train/stateful.py``), at every 10th window end
(as tt_summary) and at the last window of every event: the 1 %, median and 99 % quantiles, the largest |z|, the share
above the window bound of D129 (0.182 at tau_z = 15 s) and the bound of an event of the part,
``(1 - (1 - dt / tau_z)^n) max(|z_lo|, z_hi)`` with ``n`` the windows of its longest event.

Choice of tau_z (rule of D129 with the online audit of H-N2'): the smallest validation RMSE among the pilot runs with no
unstable equilibrium at any frozen z of D129 and in the online audit; without such a run, the largest mean band share
stable at the frozen z. Rule of H-N2' (5-fold means on highD): confirmed when the test spacing RMSE is <= 1.05 x the
GRU's E1 RMSE, every frozen-z audit and the online audit have 100 % stable equilibria (every fold) and the certificate
holds at all 21 z of [-0.2, 1] after fine-tuning (every fold); partial when the RMSE is <= 1.15 x; refuted otherwise.

Writes ``runs/_tables/m12/tt2_pilot``, ``tt2_highd`` and ``tt2_ngsim`` (``.csv``, ``.md``) and prints the choice and the
branch.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cf_stability.eval.tables import Column, Table, write_table  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402
from tt_summary import FOLDS, RUN, TAUS, _read, frozen_columns, reference_rmse, run_row  # noqa: E402

MODEL = "residual_idm_tt2"
INFO_Z = "-0.2"  # the lower end of the range of D135, audited for information
WINDOW_BOUND_15 = 1.0 - (1.0 - 0.1 / 15.0) ** 30  # the bound of the window latent of D129 at tau_z = 15 s


def tt2_row(run: Path) -> dict[str, Any]:
    """``run_row`` of tt_summary plus the audit at z = -0.2 and the persistent range of the certificate."""
    row = run_row(run)
    audit = _read(run / f"stability_z{INFO_Z}.json")
    if audit and audit.get("audit"):
        s = audit["audit"]["summary"]
        row.update({"unstable_z-0.2": s["share_unstable_numerical"]["support"],
                    "band_stable_z-0.2": (s["band_numerical"]["support"] or {}).get("stable"),
                    "max_gain_z-0.2": s["max_gain"]})  # fmt: skip
    online = _read(run / "stability.json")
    if online and online.get("audit"):
        s = online["audit"]["summary"]
        row["online_state"] = online["audit"].get("state")
        row["online_stable"] = bool(s["share_unstable_numerical"]["support"] == 0.0 and s["n_equilibria"] > 0)
    row["stable_all"] = bool(row.get("frozen_stable_all")) and bool(row.get("online_stable"))
    metrics = _read(run / "metrics.json")
    if metrics:
        row["state_pass_overflow"] = ((metrics.get("training") or {}).get("state_pass") or {}).get("overflow_steps")
    return row


def latent_on_test(run: Path, stride: int = 10) -> dict[str, float]:
    """The persistent latent of the teacher-forced pass along the run's test events."""
    import torch

    from cf_stability.data.views import load_parts
    from cf_stability.models import load_model
    from cf_stability.train.stateful import teacher_forced_path
    from cf_stability.train.tensors import EventTensors

    metrics = read_json(run / "metrics.json")
    config = metrics["config"]
    paths = config["paths"]
    parts, _ = load_parts(config["data"], config["split"], int(config["fold"]), paths["events_root"], paths["splits_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(run / "model.pt", device)
    data = EventTensors(parts["test"], device, torch.float32)
    ends, _, after = teacher_forced_path(model, data)
    ends, z = ends.cpu().numpy(), after.cpu().numpy()
    w = model.window
    every = (ends[:, 1] - (w - 1)) % stride == 0
    last = ends[:, 1] == data.lengths.cpu().numpy()[ends[:, 0]] - 1
    q = np.quantile(z[every], [0.01, 0.5, 0.99])
    q_last = np.quantile(z[last], [0.01, 0.5, 0.99])
    n_windows = int(data.lengths.max()) - w + 1
    return {
        "z_windows": int(every.sum()), "z_q01": q[0], "z_median": q[1], "z_q99": q[2],
        "z_abs_max": float(np.abs(z).max()), "z_share_beyond_0182": float(np.mean(np.abs(z[every]) > WINDOW_BOUND_15)),
        "z_last_q01": q_last[0], "z_last_median": q_last[1], "z_last_q99": q_last[2],
        "z_event_bound": float(model.latent_reach(n_windows)), "z_range_lo": model.z_lo, "z_range_hi": model.z_hi,
    }  # fmt: skip


def latent_columns() -> list[Column]:
    return [
        Column("z_q01", "z 1 %", digits=4), Column("z_median", "z median", digits=4), Column("z_q99", "z 99 %", digits=4),
        Column("z_abs_max", "max \\|z\\|", digits=4), Column("z_share_beyond_0182", "share \\|z\\| > 0.182", digits=3),
        Column("z_last_q99", "z 99 % (event ends)", digits=4), Column("z_event_bound", "bound in an event", digits=3),
    ]  # fmt: skip


def stability_columns() -> list[Column]:
    return [
        *frozen_columns(), Column("unstable_z-0.2", "unstable eq. z=-0.2", digits=3),
        Column("frozen_stable_all", "stable at every frozen z", "flag"),
        Column("online_unstable", "unstable eq. (online)", digits=3), Column("online_band_stable", "band stable (online)", digits=3),
        Column("online_max_gain", "max gain (online)", digits=3), Column("stable_all", "stable frozen and online", "flag"),
    ]  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="runs/_tables/m12")
    parser.add_argument("--no-latent", action="store_true")
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    out = resolve_path(args.out)

    # pilot: the choice of tau_z
    pilot = []
    for tau in TAUS:
        run = resolve_path(RUN.format(experiment=f"e4_tt2_pilot_tau{tau}", data="follownet_highd", model=MODEL, fold=0))
        row = {"tau_z": float(tau), **tt2_row(run)}
        if not args.no_latent and (run / "model.pt").exists():
            row.update(latent_on_test(run))
        pilot.append(row)
    pilot = pd.DataFrame(pilot)
    stable = pilot.get("stable_all", pd.Series(False, index=pilot.index)).fillna(False).astype(bool)
    eligible = pilot[stable & pilot.get("val_rmse", pd.Series(np.nan, index=pilot.index)).notna()]
    if len(eligible):
        chosen = float(eligible.loc[eligible["val_rmse"].idxmin(), "tau_z"])
        rule = ("the smallest validation RMSE among the runs with no unstable equilibrium at any frozen z of D129 and in "
                "the online audit")
    elif len(pilot) and "frozen_band_stable_mean" in pilot and pilot["frozen_band_stable_mean"].notna().any():
        chosen = float(pilot.loc[pilot["frozen_band_stable_mean"].idxmax(), "tau_z"])
        rule = "no run stable at every frozen z and online: the largest mean band share stable at the frozen z"
    else:
        chosen, rule = math.nan, "no pilot runs"
    pilot["chosen"] = pilot["tau_z"] == chosen
    pilot_columns = [
        Column("tau_z", "τ_z (s)", digits=0), Column("chosen", "chosen", "flag"), Column("val_rmse", "val RMSE s (m)", digits=3),
        Column("test_rmse", "test RMSE s (m)", digits=3), Column("epochs", "epochs", "int"), Column("best_epoch", "best", "int"),
        Column("admissible", "admissible budget", digits=4), Column("lipschitz", "lipschitz", digits=4),
        *stability_columns(), Column("cert_n_z", "certificate: z of 21", "int"),
        Column("cert_reachable", "certificate on [-0.2, 1]", "flag"), *latent_columns(),
    ]  # fmt: skip
    pilot_notes = [
        "Pilot of D135: residual_idm_tt2 (the latent carried along the vehicle, z in [-0.2, 1]) on follownet_highd, fold 0, "
        "seed 0, tau_z 15, 30 and 60 s (configs/queue/e4_tt2_pilot.yaml). val/test RMSE s: closed-loop spacing RMSE, "
        "mean over the events (m), the latent from z = 0 at the first window of every event. admissible budget: the "
        "largest r_max x lipschitz of the a priori certificate over the 21 frozen latents of [-0.2, 1], lipschitz = 0.9 x "
        "it / r_max.",
        "unstable eq. z=..: share of the unstable equilibria (numerical rule, gain > 1.02) among the equilibria in the "
        "support of the speeds at the frozen latent (scripts/audit_stability.py frozen_z=<z>; the five z of D129, and "
        "z = -0.2, the lower end of the range, for information); band stable: the band share \"stable inside the band\" "
        "(D79). online: the audit of the persistent model (stability.json: the equilibria of the latent at its steady "
        "state, the frequency-response rollouts from it, the latent carried). certificate: the number of the 21 frozen "
        "latents of [-0.2, 1] at which the scanned a priori certificate holds at every grid speed (certificate.json).",
        "z: the persistent latent of the teacher-forced pass along the observed test events (from z = 0 before the first "
        "window, once per window), at every 10th window end; event ends: at the last window of every event; bound in an "
        "event: (1 - (1 - dt / tau_z)^n) x 1 with n = 121 windows (the 15-s events of highD): the latent cannot go further "
        "within one event; 0.182: the bound of the window latent of D129 at tau_z = 15 s.",
        f"Choice (rule of D129 with the online audit of H-N2'): {rule}: tau_z = {chosen:g} s.",
    ]
    write_table(Table("tt2_pilot", "Two-timescale certified hybrid with a persistent latent: pilot and choice of τ_z (D135)",
                      pilot_notes, pilot, pilot_columns), out)  # fmt: skip

    # five folds on highD
    rows = []
    for fold in FOLDS:
        run = resolve_path(RUN.format(experiment="e4_tt2", data="follownet_highd", model=MODEL, fold=fold))
        row = {"fold": fold, **tt2_row(run), "gru_e1_rmse": reference_rmse("e1", "follownet_highd", "gru", fold),
               "e4_rmse": reference_rmse("e4_stable", "follownet_highd", "residual_idm", fold),
               "tt_rmse": reference_rmse("e4_tt", "follownet_highd", "residual_idm_tt", fold)}  # fmt: skip
        if not args.no_latent and (run / "model.pt").exists():
            row.update(latent_on_test(run))
        rows.append(row)
    highd = pd.DataFrame(rows)
    for ref in ("gru_e1", "e4", "tt"):
        if "test_rmse" in highd:
            highd[f"ratio_{ref}"] = highd["test_rmse"] / highd[f"{ref}_rmse"]
    complete = highd.dropna(subset=["test_rmse"]) if "test_rmse" in highd else highd.iloc[:0]
    mean_row = {"fold": "mean", **{k: complete[k].mean() for k in complete.columns
                                   if k not in ("fold", "run", "cert_fails") and pd.api.types.is_numeric_dtype(complete[k])}}  # fmt: skip
    ratios = {ref: complete["test_rmse"].mean() / complete[f"{ref}_rmse"].mean() if len(complete) else math.nan
              for ref in ("gru_e1", "e4", "tt")}  # fmt: skip
    mean_row.update({f"ratio_{ref}": value for ref, value in ratios.items()})
    for flag in ("frozen_stable_all", "stable_all"):
        mean_row[flag] = bool(complete[flag].all()) if len(complete) and flag in complete else None

    # fine-tuning on NGSIM I-80
    ft_rows = []
    for fold in FOLDS:
        run = resolve_path(RUN.format(experiment="e4_tt2_ft", data="ngsim_i80", model=MODEL, fold=fold))
        ft_row = {"fold": fold, **tt2_row(run),
                  "e4_ft_rmse": reference_rmse("e4_stable_ft", "ngsim_i80", "residual_idm", fold),
                  "tt_ft_rmse": reference_rmse("e4_tt_ft", "ngsim_i80", "residual_idm_tt", fold)}  # fmt: skip
        if not args.no_latent and (run / "model.pt").exists():  # the I-80 events (15-217 s): the whole range
            ft_row.update(latent_on_test(run))
        ft_rows.append(ft_row)
    ngsim = pd.DataFrame(ft_rows)
    ft_complete = ngsim.dropna(subset=["test_rmse"]) if "test_rmse" in ngsim else ngsim.iloc[:0]
    cert_after = bool(len(ft_complete) == len(FOLDS) and "cert_all_z" in ft_complete
                      and ft_complete["cert_all_z"].fillna(False).astype(bool).all())

    stable_ok = bool(len(complete) == len(FOLDS) and "stable_all" in complete and complete["stable_all"].all())
    ratio = ratios["gru_e1"]
    if len(complete) < len(FOLDS) or not math.isfinite(ratio):
        branch = f"incomplete ({len(complete)} of {len(FOLDS)} folds)"
    elif ratio <= 1.05 and stable_ok and cert_after:
        branch = "confirmed"
    elif ratio <= 1.15:
        branch = "partial"
    else:
        branch = "refuted"
    mean_of = lambda key: complete[key].mean() if key in complete and len(complete) else math.nan  # noqa: E731
    summary = (f"H-N2' (D135), 5-fold means on highD ({len(complete)} of {len(FOLDS)} folds): test spacing RMSE "
               f"{mean_of('test_rmse'):.3f} m against the GRU of E1 {mean_of('gru_e1_rmse'):.3f} m: ratio "
               f"{ratio:.3f} (rule: <= 1.05 confirmed with the two stability conditions, <= 1.15 partial); every frozen-z "
               f"audit and the online audit 100 % stable equilibria (every fold): {'yes' if stable_ok else 'no'}; "
               f"certificate at all 21 z of [-0.2, 1] after fine-tuning (scan, every fold): {'yes' if cert_after else 'no'} "
               f"-> branch: {branch}. Against E4 (e4_stable, seed 0): ratio {ratios['e4']:.3f}; against the law of D129 "
               f"(e4_tt): ratio {ratios['tt']:.3f}.")  # fmt: skip
    highd = pd.concat([highd, pd.DataFrame([mean_row])], ignore_index=True)
    highd_columns = [
        Column("fold", "fold", "text"), Column("val_rmse", "val RMSE s (m)", digits=3), Column("test_rmse", "test RMSE s (m)", digits=3),
        Column("gru_e1_rmse", "GRU E1 (m)", digits=3), Column("ratio_gru_e1", "ratio to GRU", digits=3),
        Column("e4_rmse", "E4 (m)", digits=3), Column("ratio_e4", "ratio to E4", digits=3),
        Column("tt_rmse", "tt of D129 (m)", digits=3), Column("ratio_tt", "ratio to tt", digits=3),
        Column("epochs", "epochs", "int"), Column("admissible", "admissible budget", digits=4), *stability_columns(),
        Column("growth_error", "platoon growth error", digits=3), Column("cert_n_z", "certificate: z of 21", "int"),
        Column("cert_reachable", "certificate on [-0.2, 1]", "flag"), *latent_columns(),
    ]  # fmt: skip
    highd_notes = [
        f"e4_tt2: residual_idm_tt2 (tau_z chosen on the pilot) on follownet_highd, five folds, seed 0 (configs/queue/"
        "e4_tt2.yaml). References of the same fold, seed 0: GRU E1 (runs/e1/follownet_highd/gru), the certified hybrid E4 "
        "(runs/e4_stable/follownet_highd/residual_idm) and the two-timescale law of D129 with the window-restarted latent "
        "(runs/e4_tt/follownet_highd/residual_idm_tt); RMSE: closed-loop spacing RMSE on the test events, mean over the "
        "events (m). mean: the means over the folds; the ratios of the mean row are the ratios of the 5-fold means.",
        "Stability and latent columns as in tt2_pilot. platoon growth error: mean over the profiles (platoon.json).",
        summary,
    ]
    write_table(Table("tt2_highd", "Two-timescale certified hybrid with a persistent latent on highD (D135)", highd_notes,
                      highd, highd_columns), out)  # fmt: skip
    for ref in ("e4_ft", "tt_ft"):
        if "test_rmse" in ngsim:
            ngsim[f"ratio_{ref}"] = ngsim["test_rmse"] / ngsim[f"{ref}_rmse"]
    ngsim_mean = {"fold": "mean", **{k: ft_complete[k].mean() for k in ft_complete.columns
                                     if k not in ("fold", "run", "cert_fails") and pd.api.types.is_numeric_dtype(ft_complete[k])}}  # fmt: skip
    for ref in ("e4_ft", "tt_ft"):
        if len(ft_complete):
            ngsim_mean[f"ratio_{ref}"] = ft_complete["test_rmse"].mean() / ft_complete[f"{ref}_rmse"].mean()
    ngsim_mean["cert_all_z"] = cert_after
    ngsim = pd.concat([ngsim, pd.DataFrame([ngsim_mean])], ignore_index=True)
    ngsim_columns = [
        Column("fold", "fold", "text"), Column("val_rmse", "val RMSE s (m)", digits=3), Column("test_rmse", "test RMSE s (m)", digits=3),
        Column("e4_ft_rmse", "E4 fine-tuned (m)", digits=3), Column("ratio_e4_ft", "ratio to E4", digits=3),
        Column("tt_ft_rmse", "tt fine-tuned (m)", digits=3), Column("ratio_tt_ft", "ratio to tt", digits=3),
        Column("epochs", "epochs", "int"), *stability_columns(), Column("growth_error", "platoon growth error", digits=3),
        Column("cert_z0", "certificate z = 0", "flag"), Column("cert_n_z", "certificate: z of 21", "int"),
        Column("cert_all_z", "certificate at all 21 z", "flag"), Column("cert_reachable_min", "min guaranteed margin (s⁻²)", digits=4),
        Column("cert_fails", "z failing", "text"), Column("state_pass_overflow", "eager pass steps", "int"),
        *latent_columns(),
    ]  # fmt: skip
    ngsim_notes = [
        "e4_tt2_ft: e4_tt2 fine-tuned on ngsim_i80 (init_from, the checkpoint as it is: core, scalers, certified budget; "
        "configs/queue/e4_tt2.yaml), five folds, seed 0; the latent carried along the event in training (teacher-forced "
        "pass from the first window, events of 15-217 s) and evaluation. References: E4 fine-tuned the same way "
        "(runs/e4_stable_ft/ngsim_i80/residual_idm) and the law of D129 fine-tuned (runs/e4_tt_ft/ngsim_i80/residual_idm_tt), "
        "seed 0. The certificate is that of scripts/certificate.py after the fine-tuning (scanned a priori certificate at "
        "the 21 frozen latents of [-0.2, 1]); the exact one: table certificate_tt2. eager pass steps: training steps whose "
        "teacher-forced pass exceeded the capacity of the CUDA graph and ran eagerly.",
        "Stability columns as in tt2_pilot. z: the persistent latent along the I-80 test events (15-217 s, up to 2 096 "
        "windows: within an event it can reach its whole range), as in tt2_pilot.",
    ]
    write_table(Table("tt2_ngsim", "Two-timescale certified hybrid with a persistent latent fine-tuned on NGSIM I-80 (D135)",
                      ngsim_notes, ngsim, ngsim_columns), out)  # fmt: skip
    print(f"tt2_pilot: choice tau_z = {chosen:g} s ({rule})")
    print(summary)
    print(f"tt2_summary: {time.perf_counter() - t_start:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
