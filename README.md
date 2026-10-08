# cf-stability: string stability of learned car-following models

Code, configurations, run manifests and results of the study

> **String Stability of Learned Car-Following Models: Audit, Differentiable Penalties, a Certified Hybrid
> and Corridor-Level Validation.** Mikhail Gorodnichev and Marina Moseva, Faculty of Information
> Technology, Moscow Technical University of Communication and Informatics. Submitted to *Mathematics*
> (MDPI), 2026.

The manuscript itself is not part of this repository; every table and figure of it is generated here
from the run manifests by the commands of the section [Reproduce](#reproduce).

## What the study does

Learned car-following (CF) models reproduce single-vehicle trajectories better than the Intelligent
Driver Model (IDM), but their use in simulation, where every vehicle follows another vehicle governed by
the same law, is decided by their *string stability*. This repository implements the whole study:

* **Data.** Event extraction (one leader-follower pair, no lane change, at least 15 s) from the highD
  events of the FollowNet benchmark, from NGSIM I-80 and US-101 (with an own physics-informed
  reconstruction of the raw trajectories), from the Waymo car-following pairs and from OpenACC; driver-level
  five-fold splits; IDM calibration per event and per data set (differential evolution on the GPU).
* **Models.** IDM, OVM, Newell, persistence, k-NN, MLP, GRU, LSTM, PIDL, PERL-style Newell+LSTM residual and
  a residual hybrid IDM+MLP with spectral normalisation; closed-loop training (one-step acceleration loss
  plus a 5 s teacher-free rollout loss) and closed-loop evaluation with collision detection.
* **Stability.** The analytic string-stability criterion at equilibria anchored to the spacing band of
  the data, the numerical frequency response, a 50-vehicle platoon test, differentiable stability
  penalties (Jacobian, rollout gain, combined, monotone), and an a priori certificate for the residual
  hybrid (margin-constrained IDM core plus a Lipschitz-bounded residual) that is checked before and after
  fine-tuning.
* **Experiments.** E1 audit (175 runs), E2 penalties (sweep and chosen weights, 300 runs, plus three
  arms), E3 transfer to NGSIM and Waymo, E4 certificate and the residual-amplitude sweep (350 runs), E5 the
  OpenACC control; restartable experiment queues with config hashes.
* **Corridors.** SUMO 1.27.1 scenarios of the NGSIM I-80 (500 m, three periods) and US-101 (640 m,
  weaving section) sites driven by the learned laws through an in-process libsumo loop, with the
  demand, the entry speeds and the downstream boundary from the data; 1 265 runs; macroscopic metrics
  (throughput, speeds, queue discharge, travel times, fundamental diagram, wave speed, number and
  amplitude of waves, collisions), a macro-error vector against the ground truth, equivalence tests,
  correlations, power, sensitivity to the boundary and the lane-change model, a temporal hold-out and
  an asymmetry/spectrum analysis.
* **Reporting.** Every table (CSV and LaTeX) and figure (PNG and PDF) of the paper and of its
  supplementary materials, the generated report `runs/_report/report.md` with every number read from the
  tables, and a manifest with the run counts and config hashes.

Main findings: 84-95 % of the equilibria of the unconstrained networks are string unstable while their
spacing RMSE is 8-33 % below the IDM's; the Jacobian penalty stabilises the memoryless models at no
accuracy cost and fails for the recurrent ones through five identifiable loopholes; the certified hybrid
keeps its certificate after fine-tuning on another site and is the most faithful law in both corridors
(macro error 0.125 on I-80 and 0.113 on the out-of-sample US-101, against 0.182 and 0.133 for the IDM)
and the only learned law without collisions; across 37 law-corridor pairs the macro error correlates with
the share of unstable equilibria (Spearman 0.76).

## Layout

| Path | Content |
|---|---|
| `cf_stability/data/` | event schema, processing, extraction, splits, loaders (OpenACC, FollowNet, highD, NGSIM + reconstruction, Waymo, synthetic), data views |
| `cf_stability/models/` | model interface, IDM, OVM, Newell, persistence, kNN, MLP, GRU, LSTM, PIDL, PERL, ResidualIDM |
| `cf_stability/train/` | closed-loop rollout, training with stability penalties, batched differential evolution, IDM calibration |
| `cf_stability/stability/` | equilibria, analytic and numerical frequency response, audit, penalties, certificate, platoon test |
| `cf_stability/corridor/` | SUMO scenarios of I-80 and US-101, the control loop (libsumo), the laws, macroscopic metrics, waves, asymmetry and spectrum |
| `cf_stability/eval/` | experiment queue, run collection, statistics, tables, figures, the report |
| `configs/` | hydra configs (`queue/`: the experiment queues; `corridor/`: sites and laws; `data/`, `model/`) |
| `scripts/` | entry points (`scripts/analysis/`: one-off analyses) |
| `tests/` | 600+ pytest tests (event invariants, criterion against the IDM, penalty gradients by finite differences, conservation in the corridor loop, tables and report) |
| `data/calibration/`, `data/splits/` | IDM calibrations and the fold definitions (the raw data and the extracted events are not in git: see the data access table) |
| `runs/` | every training run (`metrics.json`, `stability.json`, `platoon.json`, `certificate.json`, weights), every corridor run (`run.json`, `macro.json`, `asymmetry.json`), the SUMO scenarios and law files, the tables `runs/_tables/` and the generated report `runs/_report/` (trajectories and test frames are not in git) |
| `docs/` | `data_contract.md` (conventions and source formats), `decisions.md` (every choice and deviation, D1-D124), the milestone contracts and reports `m1`-`m8`, `theory_notes.md` |

## Setup

```bash
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt     # exact versions; torch 2.11.0+cu128
./.venv/Scripts/python.exe -m pip install -e . --no-deps
./.venv/Scripts/python.exe -m pytest
```

`requirements.txt` pins every package that `cf_stability/`, `scripts/` and `tests/` import, with the
versions the results were computed with (Python 3.11.8, Windows 10, one RTX 6000 Ada, CUDA 12.8 build of
torch). SUMO 1.27.1 comes as the `eclipse-sumo` and `libsumo` wheels. The commands below are written for
Git Bash on Windows (`./.venv/Scripts/python.exe`); on Linux use `.venv/bin/python`.

## Reproduce

The commands of every milestone in order, as in the section "Commands run" of its report in `docs/`
(section 10 of `runs/_report/report.md` lists them as well). Every output carries its resolved config
and its hash; the queues (`scripts/run_experiment.py`) and the corridor runs
(`scripts/run_corridor.py ... workers=<n>`) are restartable: complete steps are recognised by their files
and config hashes and skipped, a second call retries the failed ones. Two platform facts: SUMO cannot open
a non-ASCII path and `libsumo` cannot share a process with `pyarrow`, so the simulation process uses
8.3 short paths, imports `libsumo` first and writes npz/json only.

### Data access

| Source | Licence | How |
|---|---|---|
| OpenACC (JRC), 6 campaigns | CC BY 4.0 | `python scripts/download_data.py openacc` |
| FollowNet event sets (HighD, NGSIM I-80, Waymo; figshare collection 6777810) | CC0 | `python scripts/download_data.py follownet` (sha256 recorded; the pickled files are loaded only after the check, D21) |
| NGSIM I-80 and US-101 raw trajectories (US DOT data portal, Socrata API) | public domain | `python scripts/download_data.py ngsim --locations i-80 us-101` |
| NGSIM reconstruction (own, D42-D46; the Montanino-Punzo set is not public) | - | computed by `python scripts/extract_events.py data=ngsim_i80` and `data=ngsim_us101`, cached in `data/raw/ngsim/cache/ngsim_<location>_reconstructed.parquet` (the corridor reads it) |
| Waymo car-following pairs (Hu et al. 2022, Mendeley Data 10.17632/wfn2c3437n.2) | CC BY 4.0 | Mendeley answers scripts with a browser challenge (D24): download the five files in a browser or with PowerShell `Invoke-WebRequest` into `data/raw/waymo/` and check their sha256 against the Mendeley API |
| highD raw (levelXdata) | research licence | needs the levelXdata agreement, not included (D22): the FollowNet HighD events stand in |
| I-24 MOTION | - | not used (D101) |

Every downloaded file is recorded in `data/raw/<dataset>/SOURCES.json` (url, size, sha256, licence,
date). `data/` is not in git.

### M1: data, IDM calibration, persistence baseline

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install numpy pandas scipy pyarrow hydra-core matplotlib pytest osqp requests

.\.venv\Scripts\python.exe scripts\download_data.py follownet openacc
.\.venv\Scripts\python.exe scripts\download_data.py ngsim --locations i-80
.\.venv\Scripts\python.exe scripts\download_data.py ngsim --locations us-101
# Waymo pairs: PowerShell Invoke-WebRequest, sha256 checked against the Mendeley API (decision D24)

.\.venv\Scripts\python.exe scripts\extract_events.py data=synthetic
.\.venv\Scripts\python.exe scripts\extract_events.py data=openacc
.\.venv\Scripts\python.exe scripts\extract_events.py data=follownet_highd
.\.venv\Scripts\python.exe scripts\extract_events.py data=follownet_ngsim_i80
.\.venv\Scripts\python.exe scripts\extract_events.py data=follownet_waymo
.\.venv\Scripts\python.exe scripts\extract_events.py data=waymo
.\.venv\Scripts\python.exe scripts\describe_events.py openacc follownet_highd follownet_ngsim_i80 follownet_waymo waymo

# for each <name> of the five event sets
.\.venv\Scripts\python.exe scripts\calibrate_idm.py dataset=<name>
.\.venv\Scripts\python.exe scripts\persistence_baseline.py dataset=<name>

.\.venv\Scripts\python.exe -m pytest          # 136 passed
```

The NGSIM event sets (and the reconstruction cache) were written later with the same code
(`extract_events.py data=ngsim_i80`, `data=ngsim_us101`; M1 ran them in memory, the disk was full).

### M2: models, training, closed-loop evaluation

```powershell
.\.venv\Scripts\python.exe -m pytest                       # 231 passed (169 s on CPU)

# reference runs: every model, FollowNet HighD, driver split, fold 0, seed 0
powershell -ExecutionPolicy Bypass -File scripts\run_m2.ps1
#   = for each <model>:
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=<model> fold=0 seed=0 experiment=m2

# weight of the one-step acceleration term
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=residual_idm experiment=m2_acc0 train.acc_weight=0.0
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=residual_idm experiment=m2_acc01 train.acc_weight=0.1
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=mlp experiment=m2_acc0 train.acc_weight=0.0
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=gru experiment=m2_acc0 train.acc_weight=0.0
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=residual_idm experiment=m2_epoch0
```

### M3: stability audit, penalties, certificate

```bash
./.venv/Scripts/python.exe -m pytest                       # 306 tests: 300 passed, 6 skipped (146 s)

# trainings with a penalty (lambda = 1), reference fold and seed
python scripts/train.py data=follownet_highd fold=0 seed=0 model=mlp          train.penalty.kind=jacobian    train.penalty.weight=1.0 experiment=m3_jacobian
python scripts/train.py data=follownet_highd fold=0 seed=0 model=residual_idm train.penalty.kind=jacobian    train.penalty.weight=1.0 experiment=m3_jacobian
python scripts/train.py data=follownet_highd fold=0 seed=0 model=mlp          train.penalty.kind=linear_gain train.penalty.weight=1.0 experiment=m3_linear_gain
python scripts/train.py data=follownet_highd fold=0 seed=0 model=gru          train.penalty.kind=linear_gain train.penalty.weight=1.0 experiment=m3_linear_gain
python scripts/train.py data=follownet_highd fold=0 seed=0 model=lstm         train.penalty.kind=linear_gain train.penalty.weight=1.0 experiment=m3_linear_gain

# audit of the penalised models and of the reference models of M2 (stability.json next to every model.pt)
python scripts/audit_stability.py experiment=m3_jacobian    data=follownet_highd force=true
python scripts/audit_stability.py experiment=m3_linear_gain data=follownet_highd force=true
python scripts/audit_stability.py experiment=m2_acc0        data=follownet_highd force=true
python scripts/audit_stability.py experiment=m2             data=follownet_highd force=true
```

### M4: experiments E1-E5

```bash
./.venv/Scripts/python.exe -m pytest                     # 412 tests: 412 passed, 24 skipped (GPU variants)

# previews on the reference fold (fold 0, seed 0), before the experiments
python scripts/run_experiment.py configs/queue/preview.yaml            # anchored penalties, 5 runs
python scripts/run_experiment.py configs/queue/e4_pilot.yaml           # residual amplitude of the certified hybrid, 4 runs
python scripts/run_experiment.py configs/queue/preview_gain.yaml       # rollout penalty, 3 runs
python scripts/run_experiment.py configs/queue/preview_existence.yaml  # stand-in for the existence term, 2 runs

# experiments (restartable queues: complete steps are skipped, a second call retries failed jobs)
python scripts/run_experiment.py configs/queue/e1.yaml                 # 175 jobs
python scripts/run_experiment.py configs/queue/e2_sweep.yaml           # 120 jobs
python scripts/make_tables.py tables=[e2_sweep]                        # chosen weights (D85) -> e2_chosen.yaml, e5_penalised.yaml
python scripts/run_experiment.py configs/queue/e2_chosen.yaml          # 150 jobs
python scripts/run_experiment.py configs/queue/e2_existence.yaml       # 15 jobs
python scripts/run_experiment.py configs/queue/e4.yaml                 # 100 jobs
python scripts/run_experiment.py configs/queue/ngsim_idm.yaml          # 5 jobs
python scripts/run_experiment.py configs/queue/e5.yaml                 # 40 jobs
python scripts/run_experiment.py configs/queue/e5_penalised.yaml       # 30 jobs
python scripts/make_tables.py                                          # all tables and verdicts
```

### M5: the I-80 corridor

```bash
./.venv/Scripts/python.exe -m pip install libsumo==1.27.1 eclipse-sumo==1.27.1   # E9, new packages only
./.venv/Scripts/python.exe -m pytest                                              # 475 tests: 475 passed, 20 skipped

python scripts/run_experiment.py configs/queue/m5_laws.yaml     # 40 jobs: laws fine-tuned on NGSIM I-80 (D97)
python scripts/build_corridor.py                                # scenarios i80_p0, i80_p1, i80_p2 and the 13 law files
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2] laws=[...13 laws...] seeds=[0,...,9] workers=4   # 390 runs
python scripts/corridor_metrics.py                              # macro.json of every run and of the ground truth
python scripts/corridor_metrics.py compute=false tables=true    # runs/_tables/m5/
```

The thirteen laws of D97: `idm_global, idm_heterogeneous, knn, mlp, pidl, gru, lstm, perl, residual_idm,
residual_idm_certified, mlp_penalty, gru_penalty, lstm_penalty`.

### M6: the report

```bash
python scripts/make_report.py refresh=true    # reruns make_tables.py and corridor_metrics.py tables=true, then runs/_report/
python scripts/make_report.py                 # from the existing tables of runs/_tables/m4 and m5 (about 40 s)
./.venv/Scripts/python.exe -m pytest          # 511 tests: 491 passed, 20 skipped
```

### M7: revision arms

```bash
python scripts/run_experiment.py configs/queue/e2_lowfreq_pilot.yaml            # 6 runs (D110)
python scripts/make_tables.py tables=[e2_lowfreq]                               # weights by D85 -> e2_lowfreq.yaml
python scripts/run_experiment.py configs/queue/e2_lowfreq.yaml                  # 12 further runs
python scripts/run_experiment.py configs/queue/e4_rmax.yaml                     # 200 runs (D111)
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2] laws=[idm_heterogeneous_all] seeds=[0..9] workers=4   # 30 (D114)
python scripts/run_corridor.py scenarios=[i80_p1_gain0.01,i80_p1_gain0.04,i80_p1_nofeedback,i80_p1_lc_low,i80_p1_lc_high] \
    laws=[idm_global,residual_idm_certified,gru] seeds=[0..4] workers=4         # 75 (D113)
python scripts/run_corridor.py scenarios=[us101_p0,us101_p1,us101_p2] laws=[...13 laws...] seeds=[0..9] workers=4   # 390 (D112)
python scripts/build_corridor.py scenarios=false                                # r_max law files
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2] laws=[residual_idm_certified_r0.1,residual_idm_certified_r0.2,residual_idm_certified_r0.5,residual_idm_free_r0.3] seeds=[0..9] workers=4   # 120
python scripts/run_corridor.py scenarios=[us101_p0,us101_p1,us101_p2] laws=[...the same 4...] seeds=[0..9] workers=4   # 120
python scripts/corridor_metrics.py                                              # macro.json of every new run
python scripts/make_tables.py; python scripts/corridor_metrics.py compute=false tables=true; python scripts/make_report.py
./.venv/Scripts/python.exe -m pytest                                            # 552 tests: 530 passed, 22 skipped (after the M7 code, before the runs)
```

The scenarios of US-101 and the five variants of `i80_p1` are built before their runs (not listed in
the report; D112, D113):

```bash
python scripts/build_corridor.py laws=false sites=[us101]                       # us101_p0..p2
python scripts/build_corridor.py laws=false periods=[1] variant=gain0.01 corridor.i80.boundary.gain=0.01
python scripts/build_corridor.py laws=false periods=[1] variant=gain0.04 corridor.i80.boundary.gain=0.04
python scripts/build_corridor.py laws=false periods=[1] variant=nofeedback corridor.i80.boundary.feedback=false
python scripts/build_corridor.py laws=false periods=[1] variant=lc_low +corridor.i80.vtype.lcSpeedGain=0.5 \
    +corridor.i80.vtype.lcCooperative=0.5 +corridor.i80.vtype.lcAssertive=0.5
python scripts/build_corridor.py laws=false periods=[1] variant=lc_high +corridor.i80.vtype.lcSpeedGain=2.0 \
    +corridor.i80.vtype.lcCooperative=1.0 +corridor.i80.vtype.lcAssertive=1.5
```

### M8: strengthening arms

```bash
python scripts/make_figures_supplement.py                                   # gain expansion check, certificate tightness, band width, stability maps, corridor grids
python scripts/run_experiment.py configs/queue/e2_monotone.yaml              # 3 runs: the monotonicity control
python scripts/calibrate_idm.py dataset=ngsim_i80_p0 per_event=false         # global IDM of period 0
python scripts/run_experiment.py configs/queue/m8_temporal.yaml              # 15 runs: the temporal hold-out
python scripts/calibrate_idm.py dataset=ngsim_i80 per_event_margin=0.2 global_fit=false   # 7 651 per-event cores with margin 0.2
python scripts/certify_cores.py folds=[0,1,2,3,4]                            # certificate of every core with the shared residual
python scripts/run_experiment.py configs/queue/audit_bands.yaml              # 300 audits with the bands 1/99 and 10/90
python scripts/analysis/band_sensitivity.py                                  # runs/_tables/m8/band_sensitivity
python scripts/build_corridor.py scenarios=false                             # laws residual_idm_certified_het and *_p0
python scripts/run_corridor.py scenarios=[i80_p1,i80_p2] laws=[idm_global_p0,residual_idm_certified_p0,mlp_p0,gru_p0] seeds=[0..9] workers=4
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2,us101_p0,us101_p1,us101_p2] laws=[residual_idm_certified_het] seeds=[0..9] workers=4
python scripts/corridor_metrics.py; python scripts/corridor_asymmetry.py workers=4
python scripts/make_tables.py; python scripts/corridor_metrics.py compute=false tables=true; python scripts/make_report.py
```

### M9: revision after the internal review

```bash
python scripts/analysis/full_history_audit.py       # runs/_tables/m9/full_history: the recurrent laws linearised over their whole window (poles, windowed gain, M_w)
python scripts/analysis/threshold_sensitivity.py    # runs/_tables/m9/threshold_sensitivity: the audit at the thresholds 1.00, 1.01, 1.02 and 1.05
python scripts/analysis/equilibrium_roots.py        # runs/_tables/m9/equilibrium_roots: every upward zero crossing of the laws on the gap range
python scripts/analysis/certificate_exact.py        # runs/_tables/m9/certificate_exact: the a priori certificate verified exactly in the gap (piecewise polynomial in 1/s), the existence conditions, a sweep of 2 501 speeds
python scripts/run_experiment.py configs/queue/e4_variant_d.yaml       # 50 runs: the margin core with the bounded residual and no derivative budget (variant D)
python scripts/run_experiment.py configs/queue/e2_horizon_pilot.yaml   # 2 runs: the rollout gain penalty with a 380 s rollout, gain over the last 252 s (GRU, LSTM, fold 0)
python scripts/train.py data=ngsim_i80 model=idm fold=0 seed=0 calibration.stability_margin=0.2 experiment=m9_idm_margin_i80   # folds 0-4: the I-80 IDM with the margin (law B')
python scripts/build_corridor.py scenarios=false     # laws idm_core_margin (B), idm_margin_i80 (B') and residual_idm_margin_free_r0.3 (D)
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2,us101_p0,us101_p1,us101_p2] laws=[idm_core_margin,idm_margin_i80,residual_idm_margin_free_r0.3] seeds=[0..9] workers=3
python scripts/corridor_metrics.py; python scripts/corridor_asymmetry.py workers=2
python scripts/make_tables.py; python scripts/corridor_metrics.py compute=false tables=true; python scripts/make_report.py   # adds the tables e2_horizon (m4), correlation_clustered, contacts_absolute, h12_2_error and ablation_pairs (m8)
```

## Outputs

* `runs/_report/report.md`: the generated report (eleven sections, every number read from the tables),
  `runs/_report/tables/` (33 tables, CSV and LaTeX), `runs/_report/figures/` and
  `runs/_report/supplement/figures/` (28 figures, PNG and PDF), `runs/_report/manifest.json`.
* `runs/_tables/m4/`, `m5/`, `m8/`, `m9/`: the tables with their Markdown twins and `missing.txt` notes.
* `docs/decisions.md`: the decision log (D1-D124, E1-E10) that records every deviation from the study
  plan; `docs/study_plan.md`: the pre-specified hypotheses and thresholds and the list of the exploratory
  branches; `docs/m1_report.md` ... `docs/m8_report.md`: the milestone reports with the commands run, the
  key numbers and the deviations.

## Citation

```bibtex
@unpublished{gorodnichev2026string,
  author = {Gorodnichev, Mikhail and Moseva, Marina},
  title  = {String Stability of Learned Car-Following Models: Audit, Differentiable Penalties,
            a Certified Hybrid and Corridor-Level Validation},
  note   = {Submitted to Mathematics (MDPI)},
  year   = {2026}
}
```

## Data licences

OpenACC: CC BY 4.0 (JRC). FollowNet event sets: CC0. NGSIM: public domain (US DOT). Waymo
car-following pairs: CC BY 4.0. highD: non-commercial research licence of levelXdata, no
redistribution: extracted highD events are never written outside the working copy and `data/raw`,
`data/events` are excluded from git.

## Licence

The code, the configurations and the documentation are released under the MIT License (`LICENSE`);
the data sets keep their own licences as listed above.
