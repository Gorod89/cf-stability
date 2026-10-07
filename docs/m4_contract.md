# M4 contract: experiments E1-E5

Basis: section 5 of the specification, the decisions of the M3 review (all six proposals of
`docs/m3_report.md`, section 6, accepted on 2026-09-29) and `docs/decisions.md`, D78-D88.
Conventions as before: `dv = v - v_lead`, net gap, 10 Hz, float64 in every stability computation.

The work has three parts with separate owners of the files (sections 1-3). Interfaces between
the parts are fixed here; nobody edits a file of another part. `configs/train.yaml` is already
complete (all keys of this contract exist) and is not edited by the parts.

## 0. Layout of the results

```
runs/<experiment>/<data.name>/<model>/<split>_fold<k>_seed<s>/
    metrics.json  model.pt  test_events.parquet        scripts/train.py
    stability.json                                     scripts/audit_stability.py
    platoon.json                                       scripts/platoon_test.py
    transfer.json  transfer_<target>.parquet           scripts/evaluate_transfer.py
    certificate.json                                   scripts/certificate.py
runs/<queue name>/_logs/<job>.log, runs/<queue name>/_status.json      scripts/run_experiment.py
runs/_tables/<name>.parquet|csv                                        scripts/collect_results.py
```

Experiment names: `e1` (no penalty), `e2_<kind>_w<weight>` (`e2_jacobian_w0.1`,
`e2_linear_gain_w10`; weights printed with `%g`), `e2_gain_w<weight>` (rollout penalty of the
specification, one run), `e4_stable` (certified hybrid on HighD), `e4_stable_ft`, `e4_free_ft`
(fine-tuned on NGSIM I-80), `e5` and `e5_<kind>_w<weight>` (OpenACC), `m4_preview_<kind>`.

Every step script takes `run=<run directory>` (one run) or `experiment=<name> data=<set>` (all
runs of an experiment) like `scripts/audit_stability.py`, skips a run whose output is newer than
`model.pt` unless `force=true`, writes its config and `config_hash`, prints one line per run and
never stops at a failing run (`FAILED` line, error in the output file).

## 1. Part A: equilibria anchored to the data

Owner of: `cf_stability/train/tensors.py`, `cf_stability/train/trainer.py`,
`cf_stability/stability/{equilibrium,penalties,audit}.py`, `scripts/audit_stability.py`,
`configs/stability/default.yaml`, and their tests.

### 1.1 Spacing band of the data (D78)

`training_context(events, seed, band: BandConfig | None = None)` returns the old keys and `band`:

```
band = {"v": [...], "s_low": [...], "s_median": [...], "s_high": [...], "n": [...],
        "settings": {"dv_max": 0.5, "a_max": 0.3, "half_width": 0.5,
                     "quantiles": [0.05, 0.5, 0.95], "min_samples": 200}}
```

For every grid speed `v_g` of `V_GRID`: the training samples with `|dv| < dv_max`,
`|a| < a_max`, `|v - v_g| < half_width`; the three quantiles of their spacing; the speed is listed
only with at least `min_samples` samples. `band = None` with fewer than two listed speeds.
`BandConfig` (dataclass with `from_mapping`) lives in `tensors.py`; `TrainConfig.band` holds it
(`train.band` of the config). Between listed speeds the band is interpolated linearly in the
speed; it is defined on `[v[0], v[-1]]` only.

`cf_stability/stability/equilibrium.py` gets `Band` (tensors `v, s_low, s_median, s_high` on the
device and dtype of the model), `Band.from_context(context, model) -> Band | None` and
`Band.at(speeds) -> (s_low, s_high, has_band)`.

### 1.2 Equilibria (D79)

`find_equilibria(model, v=V_GRID, *, s_min=1.0, s_max=200.0, n_scan=400, n_bisect=50, band=None)`.

* Without band: as before.
* With band, at a speed that has a band: `f(s, 0, v)` is scanned on `n_scan` spacings of
  `[s_low(v), s_high(v)]`; the equilibrium is the first upward crossing there, refined by
  bisection. Without a crossing inside the band the old scan of `[s_min, s_max]` is used: its
  first upward crossing is an equilibrium **outside** the band.
* With band, at a speed without band: the old scan.

`Equilibria` gets `has_band` and `in_band` (bool tensors). `status`: `indifferent`, `none`,
`outside` (equilibrium outside the band of a speed that has one), `ok`, `multiple` (more than
one upward crossing in the interval that gave the equilibrium). `usable` keeps its meaning (the
law can be linearised there: found anywhere, or indifferent). New property `anchored`:
`usable & (in_band | ~has_band)`. No shape may depend on the values (CUDA graph capture).

### 1.3 Penalties (D80, D81)

* Existence term with a band (`PenaltyConfig.band`, set by the trainer from the training
  context): at every sampled speed that has a band,
  `relu(f(s_low(v), 0, v) + m_e) + relu(m_e - f(s_high(v), 0, v))`, mean over these speeds,
  times `existence_weight`. Without band (`None`): the term of D74, unchanged.
  `train.penalty.existence: band | fixed` chooses; `band` falls back to `fixed` when the
  context has no band.
* The stability terms of `jacobian`, `linear_gain` and `gain` are taken at the `anchored`
  equilibria (found with the band); `n_no_equilibrium` counts the sampled speeds that are not
  anchored.
* `linear_gain`: `train.penalty.aggregate: max | mean`. `max` (default): mean over the
  equilibria of `relu(max_w (|G_d(w)| - 1 + m_g(w)))`; `mean`: the mean over equilibria and
  frequencies of M3.
* Gradient tests against finite differences stay and cover the new term and `aggregate: max`.
  The CUDA-graph step must still give the weights of the eager step bit by bit.

### 1.5 Choice of the best epoch under a penalty (D89)

Without penalty nothing changes (early stopping on the validation closed-loop RMSE of the
spacing, bit by bit the weights of M2). With a penalty:

* after every epoch the penalty of the model is evaluated at the grid speeds `V_GRID` inside
  `[penalty.v_min, penalty.v_max]`, with the band and the settings of the training
  (`val_penalty` and its parts in the history);
* an epoch is **feasible** when `val_penalty <= penalty.tolerance` (0.01);
* the best epoch is the feasible epoch with the smallest validation RMSE; while no epoch is
  feasible, the epoch with the smallest validation RMSE (the rule of the specification);
* the patience restarts when the best epoch changes and, before the first feasible epoch, when
  `val_penalty` falls below `(1 - penalty.progress)` times its smallest earlier value;
* the initial weights stay ineligible (D76); `train_model` returns `best_feasible` and
  `n_feasible`.

`existence: fixed` reproduces M3 fully: the trainer passes no band then.

### 1.6 Penalty on a part of the steps, rollout penalty for the recurrent models (D90)

The linearised gain penalty is met by the recurrent models without string stability (preview
of 2026-09-29: derivatives of -35 to -237 1/s at the constant history of the equilibrium for
the GRU, locally unstable equilibria with a gain below 1 on the unit circle for the LSTM), so
E2 uses the penalties of the specification: `jacobian` for MLP, PIDL and ResidualIDM, `gain`
(rollouts) for GRU, LSTM and PERL.

* `train.penalty.every = n`: the penalty (stability terms and existence term) is evaluated on
  the training steps whose running number (from 0, over the whole training) is a multiple of
  `n`, and enters their loss multiplied by `n`; the other steps have no penalty term. The
  history reports the mean of the unscaled penalty and of its parts over the penalty steps of
  the epoch. `every = 1` is the behaviour so far, bit by bit.
* On CUDA with `cuda_graph: true` the steps without penalty are replayed from the CUDA graph of
  the plain step; the penalty steps of `gain` run eagerly, those of the capturable kinds are
  replayed from their own graph. Every mode gives the weights of the fully eager training with
  the same `every` bit by bit.
* `aggregate: max | mean` holds for `gain` as for `linear_gain`.

### 1.4 Audit (D79)

`audit_model` takes the band from `context["band"]` (absent or `None`: no band). Every record
gets `in_band` (`None` at a speed without band), `band_low`, `band_high`. The summary gets
`n_band` (`{"all", "support"}`) and `band_numerical`, `band_sign`: shares of the speeds that
have a band, keys `stable, unstable, outside, none, indifferent, undefined`, sum 1, for `all`
and `support` (`None` without such speeds). `stable` and `unstable` count the equilibria inside
the band only. `grid_numerical` and `grid_sign` stay (any equilibrium). The printed line of
`scripts/audit_stability.py` shows `band stable/unstable/outside/none` in percent of the speeds
in support, the largest gain and the time.

**Primary number of H1.1 and H1.2:** `band_numerical["support"]`; H1.2 is judged on
`1 - stable`.

## 2. Part B: queue, platoon script, collection, statistics

Owner of: `scripts/run_experiment.py`, `scripts/platoon_test.py`, `scripts/collect_results.py`,
`configs/queue/*.yaml`, `configs/platoon_test.yaml`, `cf_stability/eval/*`,
`cf_stability/stability/platoon.py`, and their tests.

### 2.1 Queue (D82)

`python scripts/run_experiment.py configs/queue/<name>.yaml [--workers N] [--dry-run]
[--only <substring>] [--max-jobs N]`. Plain YAML, read by the script. The directory is not called `experiment`: a directory
`configs/experiment/` would be a hydra config group and capture the override `experiment=<name>`
of every script.

```yaml
name: e2_sweep                 # directory of logs and status under runs/
workers: 2
timeout_h: 4.0                 # per step; the step's process tree is ended after it
min_free_ram_gb: 6.0           # no new job is started below it
order: [seed, fold, model]     # sort keys of the jobs, first key slowest
steps: [train, audit, platoon] # default steps of a job
groups:
  - experiment: "e2_jacobian_w{train.penalty.weight:g}"
    steps: [train, audit, platoon, transfer]      # optional, replaces the default
    overrides:                                     # Cartesian product of the lists
      data: [follownet_highd]
      model: [mlp, pidl, residual_idm]
      fold: [0, 1, 2, 3, 4]
      seed: [0]
      train.penalty.kind: [jacobian]
      train.penalty.weight: [0.01, 0.1, 1.0, 10.0]
    fixed: {init_from: "runs/e1/follownet_highd/{model}/driver_fold{fold}_seed{seed}"}  # optional, templates
```

* A job is the chain of its steps for one combination. Steps: `train`, `audit`, `platoon`,
  `transfer`, `certificate` = the five scripts of section 0, called in separate processes with
  the interpreter that runs the queue, working directory = repository root,
  `PYTHONIOENCODING=utf-8`, output appended to the log of the job.
* A step is complete when its output file exists, is not older than `model.pt` (steps after
  `train`) and, for `train`, `metrics.json["config_hash"]` equals the hash of the config composed
  with the overrides of the job (`hydra.compose`, `config_hash(to_plain(cfg))`, as
  `scripts/train.py` computes it). Complete steps are skipped: the queue can be restarted at any
  time and continues.
* Up to `workers` jobs run at once. A failing or timed-out step ends its job; the queue goes
  on. `_status.json` (rewritten after every job): per job its overrides, run directory, state
  (`done | failed | timeout | skipped`), exit code, step that failed, wall time; totals.
* One printed line per finished job: `JOB <i>/<n> <state> <experiment>/<data>/<model>/fold<k>_seed<s>
  (<minutes> min)`; at the end `QUEUE <name>: done <a>, failed <b>, skipped <c>`.
* Free memory through `ctypes` (`GlobalMemoryStatusEx`) on Windows, `os.sysconf` elsewhere;
  no new dependency.

### 2.2 Platoon script and hysteresis (D83)

`scripts/platoon_test.py` (hydra, `configs/platoon_test.yaml` with the group
`stability/platoon`): `platoon.json` = `{model, config, config_hash, git_revision,
wall_time_s, profiles: {<name>: {...}}, summary: {...}}`. Per profile: what `platoon_test`
returns today plus `hysteresis_area`. Summary: `growth_error` per OpenACC profile, their mean
over all, over the human and over the ACC profiles (`growth_error_mean`, `_human`, `_acc`),
`n_collided`, `std_ratio` (speed std of vehicle 50 over vehicle 1) per profile,
`hysteresis_area_pulse`.

Hysteresis loop area of a follower: area enclosed by its trajectory in the plane (gap, speed)
from the start of the run to its end, closed by the straight line back to the first point,
`0.5 * |sum(s_k v_{k+1} - s_{k+1} v_k)|` (m^2/s). Reported for follower 1 of every profile;
`hysteresis_area_pulse` is that of the braking pulse. `None` after a collision of the follower. Since 2026-10-01 (D107) the growth error and the standard-deviation ratio of a collided
platoon are `None` as well, and the summary means run over the profiles without collision.

### 2.3 Collection and statistics (D88)

`cf_stability/eval/collect.py`: `collect_runs(runs_root, experiments) -> DataFrame`, one row
per run: experiment, data, model, split, fold, seed, penalty kind and weight, init_from,
epochs, best epoch, wall time, validation and test summaries (`val_*`, `test_*`), audit
(`band_*`, `grid_*` shares of the numerical rule and of the sign criterion for `support`,
largest gain, agreement, counts), platoon summary, transfer per target, certificate. Missing
files give missing values, never an error. `scripts/collect_results.py experiments=[...]
name=<table>` writes `runs/_tables/<name>.parquet` and `.csv`.

`cf_stability/eval/stats.py`:
* `bootstrap_ci(values, groups, statistic=mean, n_resamples=1000, level=0.95, seed=0)`:
  percentile bootstrap over the groups (drivers, or runs);
* `driver_table(frames) -> DataFrame`: per-event frames of several runs (the
  `test_events.parquet` of all folds and seeds of one configuration) to one row per driver
  (mean over its events and over the seeds);
* `paired_comparison(a, b)`: paired differences over common drivers or runs, relative change
  of the means with bootstrap interval, Wilcoxon signed-rank test (`scipy.stats.wilcoxon`);
* `holm(p_values)`; `tost_relative(a, b, margin=0.10)` (two one-sided paired tests on the
  relative difference; used in M5).
RMSE-type numbers: the driver is the unit. Numbers of a model (shares, growth error): the run
(fold and seed) is the unit.

## 3. Part C: transfer, certificate, fine-tuning, OpenACC views

Owner of: `scripts/train.py`, `scripts/evaluate_transfer.py`, `scripts/certificate.py`,
`scripts/extract_events.py` (guard only), `cf_stability/stability/certificate.py`,
`configs/data/openacc_acc.yaml`, `configs/data/openacc_human.yaml`,
`configs/evaluate_transfer.yaml`, `configs/certificate.yaml`, `configs/e5_expected.yaml`,
`tests/test_e5_expected.py`, and their tests.

### 3.1 `scripts/train.py`

* `training_context(parts["train"], cfg.seed, train_cfg.band)`; the band is part of
  `metrics.json["context"]`.
* **Views of an event set (D87).** A data config may carry `events: <directory under
  events_root>` (default: `name`), `splits: <prefix of the split files>` (default: the events
  directory) and `filter: {mode: ACC | Human | null, exclude_files: [...]}`; the filter keeps
  the events whose `follower_id` ends with `:<mode>` and whose `meta["file"]` is not excluded,
  applied to the three parts of the fold. `openacc_acc` and `openacc_human` are such views of
  `openacc`; both exclude the five files of `configs/stability/platoon.yaml`.
  `scripts/extract_events.py` refuses a data config with `events`.
* **Fine-tuning (D86).** `init_from=<run directory>`: the model is `load_model(<run>/model.pt)`
  with its scaler, its fitted parts and its core; `fit` is not called; training as usual on the
  data of the new run; `metrics.json` gets `init_from` and the `config_hash` of the source run.
* **Stable core (D86).** `calibration.stability_margin=<m>` goes into the `CalibrationConfig`
  of the fold calibration of the IDM (the cache key holds the settings, so the free and the
  stable core have their own files).
* **Certified budget (D86).** `certificate.enforce=true` with `model=residual_idm`: after the
  calibration of the core, `lipschitz := certificate.safety * b / r_max`, `b` = the largest
  product `r_max * lipschitz` for which the a priori certificate holds at every speed of
  `V_GRID` **with the `r_max` of the model** (`max_residual_budget(..., a_priori=True,
  keep_r_max=True)`, minimum over the speeds): the spacings that can be an equilibrium of the
  hybrid grow with `r_max`. `b = 0` ends the
  run with an error. `metrics.json["certificate_budget"] = {"admissible": b, "safety": ..,
  "r_max": .., "lipschitz": ..}`.

### 3.2 Transfer (D84)

`scripts/evaluate_transfer.py` (`targets: [ngsim_i80, ngsim_us101, waymo]`, `horizon_s: 15.0`):
closed-loop evaluation of the model of a run on **all** events of every target, warm-up as in
training, twice: whole events, and the first `horizon_s` of every event (the length of the
HighD events). `transfer_<target>.parquet`: per event `event_id, follower_id, rmse_s, rmse_v,
collided` and the same with the suffix `_h` for the horizon. `transfer.json`: `{run, model,
source: {data, test: <summary of metrics.json>}, targets: {<target>: {full: <summary>, horizon:
<summary>, relative_degradation: horizon rmse_s_mean / source test rmse_s_mean - 1}},
config, config_hash, wall_time_s}`.

### 3.3 Certificate (D86)

`scripts/certificate.py`: `certificate.json` of a ResidualIDM run = `{model, core: {params,
margin_min, margin_max}, residual: {r_max, lipschitz, product, bounds, layer_norms},
a_priori: {holds, n_hold, n_grid, guaranteed_margin_min}, at_equilibria: {holds, n_hold,
n_equilibria, guaranteed_margin_min, empirical_margin_min}, per_speed: [...], config,
config_hash, wall_time_s}`; for any other model `{model, applicable: false}`.
`certificate_at_equilibria` takes an optional band and uses the anchored equilibria.

### 3.4 E5 (D87)

`configs/e5_expected.yaml` records the expected outcome of the control experiment;
`tests/test_e5_expected.py` compares it with the `stability.json` files of `runs/e5/openacc_acc`
and is skipped while they do not exist. Its content is filled in after the runs.

## 4. Experiments

| Id | Runs | Steps |
|---|---|---|
| preview | fold 0, seed 0: MLP and ResidualIDM with `jacobian`, MLP, GRU and LSTM with `linear_gain`, weight 1 | train, audit, platoon |
| E1 | HighD; MLP, PIDL, GRU, LSTM, PERL, ResidualIDM: 5 folds x 5 seeds; persistence, Newell, OVM, IDM, kNN: 5 folds | train, audit, platoon; transfer for seed 0 |
| E2 sweep | HighD; `jacobian` for MLP, PIDL, ResidualIDM, `linear_gain` for GRU, LSTM, PERL; weights 0.01, 0.1, 1, 10; 5 folds, seed 0 | train, audit, platoon |
| E2 chosen | the chosen weight of every architecture, seeds 1-4 | train, audit, platoon; transfer for seed 0 |
| E2 rollout | GRU, `gain`, chosen weight, fold 0, seed 0 | train, audit, platoon |
| E4 | `e4_stable`: ResidualIDM, core with margin 0.2, certified budget, 5 x 5; fine-tuning on `ngsim_i80` of `e4_stable`, of the ResidualIDM of E1 and of the MLP of E1, 5 x 5 each | train, audit, certificate |
| E5 | `openacc_acc` and `openacc_human`: IDM, MLP, GRU, LSTM without penalty and MLP, GRU, LSTM with the chosen weight, 5 folds, seed 0 | train, audit, platoon |

Choice of the weight (D85): per architecture, from the sweep, the weight with the smallest
mean validation RMSE of the spacing among the weights whose mean share `stable` of
`band_numerical["support"]` is at least 0.9; without such a weight, the one with the largest
share. Test data are not used.

## 5. Tests

Every part adds tests to `tests/` in the style of the existing ones (synthetic data, CPU,
scripts run as subprocesses with `CUDA_VISIBLE_DEVICES=""`, hydra paths quoted as in
`tests/test_audit.py`). The full suite stays green and below 5 minutes.
