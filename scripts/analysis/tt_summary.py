"""Tables of the two-timescale certified hybrid (docs/m11_contract.md, D129, hypothesis H-N2).

    python scripts/analysis/tt_summary.py                 # pilot, highD folds, fine-tuning, the rule of H-N2
    python scripts/analysis/tt_summary.py --no-latent     # without the latent on the test windows (no data read)

Reads the run directories of ``e4_tt_pilot_tau{15,30,60}``, ``e4_tt`` and ``e4_tt_ft`` (``residual_idm_tt``): the
``metrics.json`` (validation and test spacing RMSE, mean over the events), ``stability.json`` (the audit of the online
latent), ``stability_z{-1,-0.5,0,0.5,1}.json`` (the audits at the frozen latents, ``scripts/audit_stability.py
frozen_z=<z>``), ``platoon.json`` and ``certificate.json`` (``scripts/certificate.py``: the a priori certificate at
21 frozen latents and on the reachable range). References: the GRU of E1 (``runs/e1/follownet_highd/gru``) and the
certified hybrid of E4 (``runs/e4_stable/follownet_highd/residual_idm`` and ``runs/e4_stable_ft/ngsim_i80/
residual_idm``), seed 0, the same folds. "stable" at a frozen latent: no unstable equilibrium among the equilibria in
the support of the speeds (``share_unstable_numerical.support`` = 0, the statistic of H1.1); the band shares (stable
inside the band, D79) are reported beside it. Latent on the test windows: the online latent of every fold on the
observed windows (30 states, every 10th window end) of its test events: quantiles and the largest ``|z|`` against the
reachable bound ``1 - (1 - dt / tau_z)^30``.

Writes ``runs/_tables/m11/tt_pilot``, ``tt_highd`` and ``tt_ngsim`` (``.csv``, ``.md``) and prints the branch of the
rule of D129 (5-fold means on highD: confirmed when the test spacing RMSE is <= 1.05 x the GRU's E1 RMSE, every
frozen-z audit has 100 % stable equilibria and the certificate holds at all 21 z after fine-tuning; partial when the
RMSE is <= 1.15 x; refuted otherwise).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cf_stability.eval.tables import Column, Table, write_table  # noqa: E402
from cf_stability.utils import read_json, resolve_path  # noqa: E402

FROZEN = ("-1", "-0.5", "0", "0.5", "1")
TAUS = (15, 30, 60)
FOLDS = (0, 1, 2, 3, 4)
RUN = "runs/{experiment}/{data}/{model}/driver_fold{fold}_seed0"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def run_row(run: Path) -> dict[str, Any]:
    """The numbers of one residual_idm_tt run (empty when it has no metrics.json)."""
    row: dict[str, Any] = {"run": run.as_posix()}
    metrics = _read(run / "metrics.json")
    if metrics is None:
        return row
    budget = metrics.get("certificate_budget") or {}
    training = metrics.get("training") or {}
    row.update(
        val_rmse=metrics["val"]["rmse_s_mean"], test_rmse=metrics["test"]["rmse_s_mean"],
        test_rmse_v=metrics["test"]["rmse_v_mean"], collision_rate=metrics["test"]["collision_rate"],
        epochs=training.get("epochs"), best_epoch=metrics.get("best_epoch"),
        tau_z=metrics["model_config"].get("tau_z"), lipschitz=metrics["model_config"].get("lipschitz"),
        admissible=budget.get("admissible"), reach=(budget.get("latent") or {}).get("reach"),
        wall_time_s=metrics.get("wall_time_s"),
    )  # fmt: skip
    online = _read(run / "stability.json")
    if online and online.get("audit"):
        s = online["audit"]["summary"]
        row.update(online_unstable=s["share_unstable_numerical"]["support"],
                   online_band_stable=(s["band_numerical"]["support"] or {}).get("stable"), online_max_gain=s["max_gain"])
    stable_all, band = True, []
    for z in FROZEN:
        audit = _read(run / f"stability_z{z}.json")
        if not audit or not audit.get("audit"):
            row[f"unstable_z{z}"] = np.nan
            stable_all = False
            continue
        s = audit["audit"]["summary"]
        unstable = s["share_unstable_numerical"]["support"]
        row[f"unstable_z{z}"] = unstable
        row[f"band_stable_z{z}"] = (s["band_numerical"]["support"] or {}).get("stable")
        row[f"max_gain_z{z}"] = s["max_gain"]
        stable_all = stable_all and unstable is not None and unstable == 0.0 and s["n_equilibria"] > 0
        band.append(row[f"band_stable_z{z}"])
    row["frozen_stable_all"] = stable_all
    row["frozen_band_stable_mean"] = float(np.mean(band)) if band else np.nan
    platoon = _read(run / "platoon.json")
    if platoon and platoon.get("summary"):
        row["growth_error"] = platoon["summary"].get("growth_error_mean")
    cert = _read(run / "certificate.json")
    if cert and cert.get("latent"):
        summary, reach = cert["latent"]["summary"], cert["latent"]["reachable"]
        row.update(cert_z0=cert["a_priori"]["holds"], cert_n_z=summary["n_a_priori_holds"],
                   cert_all_z=summary["a_priori_holds_all"], cert_reachable=reach["holds"],
                   cert_reachable_min=reach["guaranteed_margin_min"], cert_fails=" ".join(
                       f"{z:g}" for z in summary["z_a_priori_fails"]))  # fmt: skip
    return row


def reference_rmse(experiment: str, data: str, model: str, fold: int) -> float:
    metrics = _read(resolve_path(RUN.format(experiment=experiment, data=data, model=model, fold=fold)) / "metrics.json")
    return float(metrics["test"]["rmse_s_mean"]) if metrics else math.nan


def latent_on_test(run: Path, stride: int = 10) -> dict[str, float]:
    """The online latent on the observed windows of the run's test events: quantiles and the largest |z|."""
    import torch

    from cf_stability.data.views import load_parts
    from cf_stability.models import load_model

    metrics = read_json(run / "metrics.json")
    config = metrics["config"]
    paths = config["paths"]
    parts, _ = load_parts(config["data"], config["split"], int(config["fold"]), paths["events_root"], paths["splits_root"])
    model = load_model(run / "model.pt")
    window = int(model.config()["window"])
    batches = []
    for event in parts["test"]:
        states = np.stack((event.s, event.dv, event.v), axis=1)
        ends = np.arange(window - 1, len(states), stride)
        if len(ends):
            batches.append(np.stack([states[e - window + 1 : e + 1] for e in ends]))
    x = torch.as_tensor(np.concatenate(batches), dtype=torch.float32)
    with torch.no_grad():
        z = torch.cat([model.latent(x[k : k + 65536]) for k in range(0, len(x), 65536)]).double().numpy()
    q = np.quantile(z, [0.01, 0.5, 0.99])
    return {"z_windows": len(z), "z_q01": q[0], "z_median": q[1], "z_q99": q[2], "z_abs_max": float(np.abs(z).max()),
            "z_reach": float(model.latent_reach())}  # fmt: skip


def frozen_columns() -> list[Column]:
    return [Column(f"unstable_z{z}", f"unstable eq. z={z}", digits=3) for z in FROZEN] + \
        [Column(f"band_stable_z{z}", f"band stable z={z}", digits=3) for z in FROZEN]  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="runs/_tables/m11")
    parser.add_argument("--no-latent", action="store_true")
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    out = resolve_path(args.out)
    model = "residual_idm_tt"

    # pilot: the choice of tau_z
    pilot = []
    for tau in TAUS:
        run = resolve_path(RUN.format(experiment=f"e4_tt_pilot_tau{tau}", data="follownet_highd", model=model, fold=0))
        pilot.append({"tau_z": float(tau), **run_row(run)})
    pilot = pd.DataFrame(pilot)
    eligible = pilot[pilot.get("frozen_stable_all", pd.Series(False, index=pilot.index)).fillna(False).astype(bool)]
    if len(eligible):
        chosen = float(eligible.loc[eligible["val_rmse"].idxmin(), "tau_z"])
        rule = "the smallest validation RMSE among the runs with 100 % stable equilibria at every frozen z"
    elif len(pilot) and "frozen_band_stable_mean" in pilot:
        chosen = float(pilot.loc[pilot["frozen_band_stable_mean"].idxmax(), "tau_z"])
        rule = "no run with 100 % stable equilibria at every frozen z: the largest stable share"
    else:
        chosen, rule = math.nan, "no pilot runs"
    pilot["chosen"] = pilot["tau_z"] == chosen
    pilot_columns = [
        Column("tau_z", "τ_z (s)", digits=0), Column("chosen", "chosen", "flag"), Column("val_rmse", "val RMSE s (m)", digits=3),
        Column("test_rmse", "test RMSE s (m)", digits=3), Column("epochs", "epochs", "int"), Column("best_epoch", "best", "int"),
        Column("reach", "reach of z", digits=3), Column("admissible", "admissible budget", digits=4),
        Column("lipschitz", "lipschitz", digits=4), *frozen_columns(), Column("frozen_stable_all", "stable at every frozen z", "flag"),
        Column("online_unstable", "unstable eq. (online)", digits=3), Column("online_max_gain", "max gain (online)", digits=3),
        Column("cert_n_z", "certificate: z of 21", "int"), Column("cert_reachable", "certificate on the reachable range", "flag"),
    ]  # fmt: skip
    pilot_notes = [
        "Pilot of D129: residual_idm_tt on follownet_highd, fold 0, seed 0, tau_z 15, 30 and 60 s (configs/queue/"
        "e4_tt_pilot.yaml). val/test RMSE s: closed-loop spacing RMSE, mean over the events of the part (m). reach of z: "
        "1 - (1 - dt / tau_z)^30, the largest |z| of the online latent (z_0 = 0 at the start of the window of 30 states); "
        "admissible budget: the largest r_max x lipschitz of the a priori certificate over the frozen latents of the "
        "reachable range (21 values), lipschitz = 0.9 x it / r_max.",
        "unstable eq. z=..: share of the unstable equilibria (numerical rule, gain > 1.02) among the equilibria in the "
        "support of the speeds at the frozen latent (scripts/audit_stability.py frozen_z=<z>); band stable: the band share "
        "\"stable inside the band\" (D79; the rest: equilibria outside the band). stable at every frozen z: no unstable "
        "equilibrium at any of the five. online: the audit of the model as it runs (stability.json). certificate: the "
        "number of the 21 frozen latents -1, -0.9, ..., 1 at which the scanned a priori certificate holds at every grid "
        "speed, and whether it holds on the reachable range (certificate.json).",
        f"Choice (D129): {rule}: tau_z = {chosen:g} s.",
    ]
    write_table(Table("tt_pilot", "Two-timescale certified hybrid: pilot and choice of τ_z (D129)", pilot_notes, pilot,
                      pilot_columns), out)  # fmt: skip

    # five folds on highD
    rows = []
    for fold in FOLDS:
        run = resolve_path(RUN.format(experiment="e4_tt", data="follownet_highd", model=model, fold=fold))
        row = {"fold": fold, **run_row(run), "gru_e1_rmse": reference_rmse("e1", "follownet_highd", "gru", fold),
               "e4_rmse": reference_rmse("e4_stable", "follownet_highd", "residual_idm", fold)}  # fmt: skip
        if not args.no_latent and (run / "model.pt").exists():
            row.update(latent_on_test(run))
        rows.append(row)
    highd = pd.DataFrame(rows)
    if "test_rmse" in highd:
        highd["ratio_gru"] = highd["test_rmse"] / highd["gru_e1_rmse"]
        highd["ratio_e4"] = highd["test_rmse"] / highd["e4_rmse"]
    complete = highd.dropna(subset=["test_rmse"]) if "test_rmse" in highd else highd.iloc[:0]
    mean_row = {"fold": "mean", **{k: complete[k].mean() for k in complete.columns
                                   if k not in ("fold", "run", "cert_fails") and pd.api.types.is_numeric_dtype(complete[k])}}  # fmt: skip
    ratio = complete["test_rmse"].mean() / complete["gru_e1_rmse"].mean() if len(complete) else math.nan
    ratio_e4 = complete["test_rmse"].mean() / complete["e4_rmse"].mean() if len(complete) else math.nan
    mean_row.update(ratio_gru=ratio, ratio_e4=ratio_e4, frozen_stable_all=bool(complete["frozen_stable_all"].all())
                    if len(complete) else None)  # fmt: skip

    # fine-tuning on NGSIM I-80
    ft_rows = []
    for fold in FOLDS:
        run = resolve_path(RUN.format(experiment="e4_tt_ft", data="ngsim_i80", model=model, fold=fold))
        ft_rows.append({"fold": fold, **run_row(run),
                        "e4_ft_rmse": reference_rmse("e4_stable_ft", "ngsim_i80", "residual_idm", fold)})  # fmt: skip
    ngsim = pd.DataFrame(ft_rows)
    ft_complete = ngsim.dropna(subset=["test_rmse"]) if "test_rmse" in ngsim else ngsim.iloc[:0]
    cert_after = bool(len(ft_complete) == len(FOLDS) and ft_complete["cert_all_z"].fillna(False).astype(bool).all())

    stable_ok = bool(len(complete) == len(FOLDS) and complete["frozen_stable_all"].all())
    if not len(complete) or not math.isfinite(ratio):
        branch = "incomplete (runs missing)"
    elif ratio <= 1.05 and stable_ok and cert_after:
        branch = "confirmed"
    elif ratio <= 1.15:
        branch = "partial"
    else:
        branch = "refuted"
    folds_text = f"{len(complete)} of {len(FOLDS)} folds"
    summary = (f"H-N2 (D129), 5-fold means on highD ({folds_text}): test spacing RMSE {complete['test_rmse'].mean():.3f} m "
               f"against the GRU of E1 {complete['gru_e1_rmse'].mean():.3f} m: ratio {ratio:.3f} (rule: <= 1.05 confirmed "
               f"with the two stability conditions, <= 1.15 partial); every frozen-z audit 100 % stable equilibria: "
               f"{'yes' if stable_ok else 'no'}; certificate at all 21 z after fine-tuning (scan, every fold): "
               f"{'yes' if cert_after else 'no'} -> branch: {branch}. Against the certified hybrid of E4 (e4_stable, "
               f"seed 0): ratio {ratio_e4:.3f}.")  # fmt: skip
    highd = pd.concat([highd, pd.DataFrame([mean_row])], ignore_index=True)
    highd_columns = [
        Column("fold", "fold", "text"), Column("val_rmse", "val RMSE s (m)", digits=3), Column("test_rmse", "test RMSE s (m)", digits=3),
        Column("gru_e1_rmse", "GRU E1 test RMSE (m)", digits=3), Column("ratio_gru", "ratio to GRU", digits=3),
        Column("e4_rmse", "E4 test RMSE (m)", digits=3), Column("ratio_e4", "ratio to E4", digits=3),
        Column("epochs", "epochs", "int"), Column("admissible", "admissible budget", digits=4),
        *frozen_columns(), Column("frozen_stable_all", "stable at every frozen z", "flag"),
        Column("online_unstable", "unstable eq. (online)", digits=3), Column("online_band_stable", "band stable (online)", digits=3),
        Column("online_max_gain", "max gain (online)", digits=3), Column("growth_error", "platoon growth error", digits=3),
        Column("cert_n_z", "certificate: z of 21", "int"), Column("cert_reachable", "certificate on the reachable range", "flag"),
        Column("z_q01", "z 1 %", digits=4), Column("z_median", "z median", digits=4), Column("z_q99", "z 99 %", digits=4),
        Column("z_abs_max", "max \\|z\\|", digits=4), Column("z_reach", "reach", digits=4),
    ]  # fmt: skip
    highd_notes = [
        "e4_tt: residual_idm_tt (tau_z 15 s, chosen on the pilot) on follownet_highd, five folds, seed 0 "
        "(configs/queue/e4_tt.yaml). References of the same fold, seed 0: GRU E1 (runs/e1/follownet_highd/gru) and the "
        "certified hybrid E4 (runs/e4_stable/follownet_highd/residual_idm); RMSE: closed-loop spacing RMSE on the test "
        "events, mean over the events (m). mean: the means over the folds; the ratios of the mean row are the ratios of "
        "the 5-fold means (the statistic of H-N2).",
        "Stability columns as in tt_pilot. platoon growth error: mean over the profiles (platoon.json). z: the online "
        "latent on the observed windows of the test events (30 states, every 10th window end): 1 %, median and 99 % "
        "quantiles and the largest |z|, against its reach 1 - (1 - dt / tau_z)^30.",
        summary,
    ]
    write_table(Table("tt_highd", "Two-timescale certified hybrid on highD: accuracy, frozen-latent audits, latent (D129)",
                      highd_notes, highd, highd_columns), out)  # fmt: skip
    if "test_rmse" in ngsim:
        ngsim["ratio_e4_ft"] = ngsim["test_rmse"] / ngsim["e4_ft_rmse"]
    ngsim_mean = {"fold": "mean", **{k: ft_complete[k].mean() for k in ft_complete.columns
                                     if k not in ("fold", "run", "cert_fails") and pd.api.types.is_numeric_dtype(ft_complete[k])}}  # fmt: skip
    if len(ft_complete):  # the ratio of the 5-fold means
        ngsim_mean["ratio_e4_ft"] = ft_complete["test_rmse"].mean() / ft_complete["e4_ft_rmse"].mean()
    ngsim = pd.concat([ngsim, pd.DataFrame([ngsim_mean])], ignore_index=True)
    ngsim_columns = [
        Column("fold", "fold", "text"), Column("val_rmse", "val RMSE s (m)", digits=3), Column("test_rmse", "test RMSE s (m)", digits=3),
        Column("e4_ft_rmse", "E4 fine-tuned test RMSE (m)", digits=3), Column("ratio_e4_ft", "ratio to E4", digits=3),
        Column("epochs", "epochs", "int"), *frozen_columns(), Column("frozen_stable_all", "stable at every frozen z", "flag"),
        Column("online_unstable", "unstable eq. (online)", digits=3), Column("online_max_gain", "max gain (online)", digits=3),
        Column("growth_error", "platoon growth error", digits=3), Column("cert_z0", "certificate z = 0", "flag"),
        Column("cert_n_z", "certificate: z of 21", "int"), Column("cert_all_z", "certificate at all 21 z", "flag"),
        Column("cert_reachable", "certificate on the reachable range", "flag"),
        Column("cert_reachable_min", "min guaranteed margin, reachable (s⁻²)", digits=4), Column("cert_fails", "z failing", "text"),
    ]  # fmt: skip
    ngsim_notes = [
        "e4_tt_ft: e4_tt fine-tuned on ngsim_i80 (init_from, the checkpoint as it is: core, scalers, certified budget; "
        "configs/queue/e4_tt.yaml), five folds, seed 0. Reference: the certified hybrid fine-tuned the same way "
        "(runs/e4_stable_ft/ngsim_i80/residual_idm, seed 0). The certificate is that of scripts/certificate.py after the "
        "fine-tuning (scanned a priori certificate at the 21 frozen latents and on the reachable range); the exact one at "
        "the 21 latents: table certificate_tt.",
        "Stability columns as in tt_pilot.",
    ]
    write_table(Table("tt_ngsim", "Two-timescale certified hybrid fine-tuned on NGSIM I-80 (D129)", ngsim_notes, ngsim,
                      ngsim_columns), out)  # fmt: skip
    print(f"tt_pilot: choice tau_z = {chosen:g} s ({rule})")
    print(summary)
    print(f"tt_summary: {time.perf_counter() - t_start:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
