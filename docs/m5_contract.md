# M5 contract: I-80 corridor

Basis: section 6 and the correlation and TOST items of section 7 of the specification, the
decisions of the M1, M2 and M4 reviews (D48, D49, corridor laws trained on NGSIM I-80, recurrent
laws stay in the list), and `docs/decisions.md`, E9 and D94-D101. Conventions as before:
`dv = v - v_lead`, net gap, 10 Hz, semi-implicit Euler, acceleration clip [-8, 4] m/s^2,
collision = net gap <= 0.

Two parts with separate owners of the files (sections 2-4 and 5-6); the file formats of
section 1 are the interface between them.

## 0. Three kinds of processes (E9)

`libsumo` 1.27.1 brings its own Arrow library and cannot share a process with `pyarrow` 25 of
the environment: whichever is imported second fails to load. SUMO cannot open paths with
non-ASCII characters either (the repository lies under `C:\Users\Город`), and the `sumo`
package sets `SUMO_HOME` to such a path.

* **Build** (`scripts/build_corridor.py`): pandas and pyarrow, no `libsumo`; calls `netconvert`
  as a child process.
* **Simulation** (`scripts/run_corridor.py`): `SUMO_HOME` is set to the 8.3 short path and
  `libsumo` is imported **before** pandas and torch; no parquet is read or written; every path
  handed to SUMO is a short path (`GetShortPathNameW`). `cf_stability.corridor.sumo_env` holds
  these helpers. Modules that the simulation imports must not import `pyarrow` at module level
  (`cf_stability/data/ngsim.py` does today and is reached through `cf_stability.data`).
* **Metrics** (`scripts/corridor_metrics.py`): pandas, no `libsumo`.

Tests never import `libsumo` in the pytest process; they run the scripts as child processes.

## 1. Files

```
runs/corridor/scenarios/<scenario>/          scripts/build_corridor.py
    net.net.xml  routes.rou.xml  (plain XML sources next to them)
    boundary.npz       t [T] (s), speed [6, T] (m/s): rows = NGSIM lanes 1..6
    ground_truth.npz   as trajectories.npz, from the data
    vehicles_truth.npz as vehicles.npz, from the data
    scenario.json      geometry, period, counts, config, config_hash
runs/corridor/laws/<law>.json (+ <law>.npz)   scripts/build_corridor.py laws=true
runs/corridor/<scenario>/<law>/seed<s>/      scripts/run_corridor.py
    trajectories.npz  vehicles.npz  run.json
    macro.json                               scripts/corridor_metrics.py
runs/corridor/scenarios/<scenario>/macro.json   (ground truth)
runs/_tables/m5/*.md|csv                     scripts/corridor_metrics.py tables=true
```

`trajectories.npz`, arrays of equal length, one row per vehicle and whole second, only rows
with `x_in <= x <= x_out`: `t` float32 (s from the start of the period), `vehicle` int32 (row of
`vehicles.npz`), `x` float32 (m, corridor coordinate of the front bumper), `lane` int8 (NGSIM
number: 1 = left ... 6 = right, 7 = auxiliary lane of the on-ramp), `v` float32 (m/s).

`vehicles.npz`, one row per vehicle of the demand: `vehicle_id` int32, `depart_planned`,
`depart` (NaN: never inserted), `entry_x`, `entry_lane`, `exit_t` (time of passing `x_out`, NaN
otherwise), `exit_lane`, `length`, `v_class`, `member` int8 (which member of the law drives it),
`collided` bool (removed after a collision as follower), `teleported` bool.

`run.json`: scenario, law, seed, `n_planned`, `n_inserted`, `n_exited`, `n_in_network`,
`n_collisions`, `n_sumo_collisions`, `n_teleports`, `n_not_inserted`, `mean_depart_delay_s`,
`sim_time_s`, `steps`, `vehicle_steps`, `vehicle_steps_per_second`, `wall_time_s`,
`sumo_version`, `config`, `config_hash`, `git_revision`.
Conservation: `n_inserted = n_exited + n_in_network + n_collisions + n_teleports_removed`.

## 2. Scenario of I-80 (D94, D95)

Corridor coordinate `x` = longitudinal position of the reconstruction
(`data/raw/ngsim/cache/ngsim_i-80_reconstructed.parquet`: `track_id, period, vehicle_id,
frame_id, lane_id, x, v, length, v_class`). Facts of the data: 5 678 tracks (periods 0, 1, 2 =
16:00-16:15, 17:00-17:15, 17:15-17:30); main-line tracks begin at `x` about 20 m, the on-ramp
(lane 7) at about 101.6 m; lane 7 ends at 204 m (95 % of the merges 7 -> 6 by 203.6 m); tracks
end at 500 m; vehicles already inside the section at the start of a period are not in the data.

* `x_in = 20`, `x_out = 500`. Edges: `main_a` [20, 100] 6 lanes; `main_b` [100, 204] 7 lanes,
  its right-most lane (SUMO index 0) is the auxiliary lane; `main_c` [204, 500] 6 lanes; `out`
  [500, 650] 6 lanes (downstream buffer). SUMO lane index = 6 - NGSIM lane on the 6-lane edges
  and 7 - NGSIM lane on `main_b` (NGSIM 7 -> index 0). Connections: `main_a` i -> `main_b`
  i + 1 -> `main_c` i -> `out` i; the auxiliary lane has no successor. Lane width 3.66 m, speed
  29 m/s. Built from plain node, edge and connection files with `netconvert`.
* Demand: every track of the period is one vehicle, inserted at the time, lane, position and
  speed of its first sample (tracks that begin inside the section are inserted where they
  begin), with its length and class (1 motorcycle, 2 passenger, 3 truck). Vehicle types:
  `minGap = 0` (the leader gap of SUMO is then the net gap), `sigma = 0`, LC2013 with SUMO's
  defaults except `lcKeepRight = 0`, the same for every run.
* Downstream boundary (D95): on `out` the speed of every vehicle is prescribed. Speed of the
  data per lane and second: mean speed of the samples with `470 <= x <= 500` in that lane
  within 2.5 s of the second; gaps are filled with the last value. It is multiplied by
  `g = clip(1 - 0.02 * dN, 0.2, 1.5)`, `dN` = simulated vehicles that have passed `x_out` minus
  data vehicles that have passed it by that time (`boundary.npz`: `exits [T]`); the speed of a
  buffer vehicle changes by at most -3 / +2 m/s^2.
* One scenario per period: `i80_p0`, `i80_p1`, `i80_p2`. Analysis window of the metrics:
  180 s to 840 s after the start of the period.
* I-24: `corridor.i24.enabled` exists and is false; with true the builder stops with the
  message that the MOTION data are not on this machine (D101).

## 3. Control loop (D96, D97)

`libsumo.start` with `--step-length 0.1`, `--seed <seed>`, `--collision.action warn`,
`--collision.mingap-factor 0`, `--time-to-teleport 300`, no step log. Per step, after
`simulationStep`:

* new vehicles: `setSpeedMode(id, 0)`, subscription of speed, lane, lane position, road, and of
  the leader; a member of the law is drawn for the vehicle (generator seeded by the
  seed of the run, draws in the order of the planned departures, so that a vehicle has the same
  member whatever happens in the simulation);
* vehicles on `main_a`, `main_b`, `main_c`: state `(s, dv, v)` from the leader subscription;
  without leader within 50 m: `s = 50 m`, `dv = 0` (D96); on the auxiliary lane a standing
  virtual leader at the end of the lane replaces a leader that is further away (`s = 204 - x`,
  `dv = v`, seen within 50 m); the history of a new vehicle is filled with its first state; acceleration of the
  member for the history, clipped to [-8, 4]; `v_next = max(v + a dt, 0)`; `setSpeed(id, v_next)`.
  SUMO's Euler update then moves the vehicle by `v_next dt`: the integration scheme of the
  project. Inference is batched per member;
* `s <= 0`: collision of the follower, counted once; the follower is removed;
  SUMO's own collision and teleport reports are counted as well;
* vehicles on `out`: `setSpeed` towards the boundary speed of their lane (section 2).

Laws (`configs/corridor/laws.yaml`, exported to `runs/corridor/laws/` by the build):

| Law | Members |
|---|---|
| `idm_global` | IDM parameters of `runs/e3_reference/ngsim_i80/idm/driver_fold{0..4}_seed0` |
| `idm_heterogeneous` | one parameter set per vehicle, drawn from the interior per-event estimates of `data/calibration/ngsim_i80/idm_per_event_fixed_v0_b.parquet` (`v0`, `b` fixed, D48) |
| `knn` | `runs/m5_ft/ngsim_i80/knn/driver_fold{k}_seed0` |
| `mlp`, `residual_idm` | `runs/e4_free_ft/ngsim_i80/<model>/driver_fold{k}_seed0` |
| `residual_idm_certified` | `runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold{k}_seed0` |
| `pidl`, `gru`, `lstm`, `perl` | `runs/m5_ft/ngsim_i80/<model>/driver_fold{k}_seed0` |
| `mlp_penalty` | `runs/m5_ft_jacobian_w1/ngsim_i80/mlp/driver_fold{k}_seed0` |
| `gru_penalty`, `lstm_penalty` | `runs/m5_ft_gain_w0.1/ngsim_i80/<model>/driver_fold{k}_seed0` |

Runs: 13 laws x 3 scenarios x seeds 0-9. `scripts/run_corridor.py scenario=<s> law=<l> seed=<n>`
runs one; `scripts/run_corridor.py scenarios=[...] laws=[...] seeds=[...] workers=<n>` runs the
missing ones as child processes (a run is complete when `run.json` carries the hash of its
config), restartable, one printed line per run, a failing run does not stop the others.

The throughput of the loop (vehicle steps per second with and without inference) is measured on
`i80_p1` and reported.

## 4. Tests of sections 2-3

Scenario: counts of vehicles per period and lane against the data, lane mapping, XML read back,
`netconvert` succeeds (child process); boundary speeds against a hand computation. Loop (child
processes, a synthetic scenario of 60 s with a few dozen vehicles and a toy law): conservation
of the vehicle count as in section 1, no teleport, a stable law without collisions, the state
handed to the law equals the gap and speeds of SUMO, the member of a vehicle does not depend on
the simulation, the run is reproducible for a fixed seed and restart-safe.

## 5. Macroscopic metrics (D98, D99)

`scripts/corridor_metrics.py` computes, for every run and for the ground truth of every
scenario, from `trajectories.npz` and `vehicles.npz` inside the analysis window:

* **Detectors** every 100 m (`x = 100, 200, 300, 400, 450`): vehicles passing per 30 s (linear
  interpolation of the 1 Hz positions), mean speed of the passing vehicles.
  `throughput_vph` = flow at `x = 450 m` over the window (total and per lane).
* **Edie grid** (cells 100 m x 30 s, lanes together and per lane): flow, density, speed
  (`cf_stability.corridor.macro.edie_grid`). `mean_speed` = total distance over total time.
* **Fundamental diagram**: the cells of the grid (lanes together) as points; mean flow per
  density bin of 10 veh/km, with counts; `fd_scatter` = standard deviation of the flow within
  the bins.
* **Capacity drop**: `queue_discharge_flow` = mean flow at `x = 450 m` over the 30 s intervals
  in which the speed at `x = 100 m` is below 11 m/s; `capacity_drop` = 1 -
  `queue_discharge_flow` / (largest 2-minute flow at `x = 450 m`); `None` without congested
  or without any interval.
* **Waves**: speed field on cells of 20 m x 2 s (lanes 1-6 together, time-mean speed); a cell
  is in a stop wave when its speed is below 40 % of the median speed of the window at that
  `x`; connected regions of at least 60 m and 6 s are waves; per wave a line through its
  leading edges (time of entry into the wave against `x`, least squares) gives the wave speed
  (negative upstream); `n_waves`, `wave_speed` (median over the waves, m/s), `wave_amplitude`
  (median of: median speed at that `x` minus the lowest speed inside the wave).
* **Travel time** from `x_in + 10 m` to `x_out - 10 m` of the vehicles that do both inside the
  window: mean, median, 10 % and 90 % quantiles, the sorted values (for distances).
* **Safety and demand**: collisions per 1000 vehicle kilometres, share of the planned vehicles
  inserted, mean insertion delay.

Macro-error vector of a run against the ground truth of its scenario (relative errors):
throughput, mean speed, queue discharge flow, fundamental diagram (RMSE of the binned flow
over the bins both have, over the mean flow of the truth), wave speed, number of waves (over
max(truth, 1)), wave amplitude, travel time (Wasserstein-1 distance over the mean travel time
of the truth). `macro_error` = mean of the absolute values of the components that exist; the
number of components is reported. A run with collisions keeps its metrics; the collision rate
is reported next to the macro error.

## 6. Tables and statistics (D100)

`scripts/corridor_metrics.py tables=true` writes to `runs/_tables/m5/`:

* `laws.md`: per law the mean over scenarios and seeds of every component, of `macro_error`
  and of the raw metrics, with 95 % bootstrap intervals (unit: run = scenario and seed),
  collisions, insertion share; the ground-truth values per scenario.
* `instability.md`: per law the unstable-equilibrium fraction of its members (mean of
  `share_unstable_numerical` of the speeds in support, and of the band share "not stable", from
  the `stability.json` of the member runs; for `idm_global` the members' audits; for
  `idm_heterogeneous` the share of grid speeds in the support of the NGSIM data at which the
  exact discrete-time gain of the vehicle's IDM exceeds 1.02 on the audit frequencies, mean over
  the parameter sets) and the macro error; Spearman and Pearson correlation over the laws with
  bootstrap intervals (resampling the laws, 1000 resamples); the same with and without the laws
  that collide.
* `tost.md`: `residual_idm_certified` against `idm_global`: for every raw metric the paired
  relative difference over the runs (scenario and seed) and the two one-sided tests for the
  margin of 10 % (`cf_stability.eval.stats.tost_relative`).

## 7. Tests of sections 5-6

Hand-made trajectories: uniform flow (flow, density, speed, throughput, travel time exact); a
synthetic stop wave travelling upstream at -5 m/s (wave speed within 0.5 m/s, one wave,
amplitude); detectors against crossings counted by hand; macro-error vector of identical
inputs is zero; tables from a hand-made tree of runs (means, correlation of a constructed
example, TOST accepts equal samples); missing files give empty cells, not errors.
