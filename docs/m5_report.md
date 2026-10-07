# Milestone M5 report: I-80 corridor

Date: 2026-09-30. Contract: `docs/m5_contract.md`. Decisions: `docs/decisions.md`, E9 and
D94-D104. The GPU and the CPU were shared with other jobs; timings are as measured. Nothing is
committed to git. Tables with intervals: `runs/_tables/m5/{laws,instability,tost}.md|csv`
(`python scripts/corridor_metrics.py tables=true`).

## 1. Commands run

```bash
./.venv/Scripts/python.exe -m pip install libsumo==1.27.1 eclipse-sumo==1.27.1   # E9, new packages only
./.venv/Scripts/python.exe -m pytest                                              # 475 tests: 475 passed, 20 skipped

python scripts/run_experiment.py configs/queue/m5_laws.yaml     # 40 jobs: laws fine-tuned on NGSIM I-80 (D97)
python scripts/build_corridor.py                                # scenarios i80_p0, i80_p1, i80_p2 and the 13 law files
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2] laws=[...13 laws...] seeds=[0,...,9] workers=4   # 390 runs
python scripts/corridor_metrics.py                              # macro.json of every run and of the ground truth
python scripts/corridor_metrics.py compute=false tables=true    # runs/_tables/m5/
```

The 390 runs took 70 minutes with four child processes; none failed. A run of the IDM laws
takes 16-18 s, of the MLP laws 23-32 s, of the recurrent laws and kNN 53-85 s (1.3-1.7 million
vehicle steps each). Throughput of the loop: about 300 000 vehicle steps per second without a
law, 120 000 with the IDM, 66 000-80 000 with the memoryless networks, 21 000-38 000 with the
recurrent networks and kNN (inference on the GPU).

## 2. What was built

| Part | Files | Content |
|---|---|---|
| SUMO environment | `corridor/sumo_env.py` | `SUMO_HOME` and every path as 8.3 short paths; `libsumo` imported before pandas; no parquet in the simulation process (E9) |
| Scenario | `corridor/scenario.py`, `scripts/build_corridor.py`, `configs/corridor/i80.yaml` | network of I-80 from plain node, edge and connection files through `netconvert`; one vehicle per track of the data; boundary speeds and cumulative exits of the data; ground truth at 1 Hz (D94, D95, D103) |
| Laws | `corridor/laws.py`, `configs/corridor/laws.yaml` | 13 laws, five members each (fold models or calibrations), per-vehicle parameters of the heterogeneous IDM, device per law (D97) |
| Control loop | `corridor/loop.py`, `scripts/run_corridor.py` | subscriptions, batched inference per member, speed mode 0, clip, Euler update of SUMO, boundary feedback, contact episodes, HOV lane, multi-run mode with child processes (D96, D102, D104) |
| Metrics | `corridor/macro.py`, `corridor/waves.py`, `scripts/corridor_metrics.py`, `configs/corridor_metrics.yaml` | detectors, Edie grid, fundamental diagram, queue discharge and capacity drop, stop waves (edges and cross-correlation), travel times, collisions, macro-error vector (D98, D99) |
| Tables | `eval/corridor_tables.py` | laws, instability against macro error with correlations, TOST of the certified hybrid against the IDM (D100) |

Two limits of the machine shaped the code (E9): `libsumo` 1.27.1 cannot share a process with
`pyarrow` 25 (the simulation imports `libsumo` first and reads and writes `npz` and JSON only),
and SUMO cannot open the non-ASCII path of the repository (short paths everywhere). Tests never
import `libsumo` in the pytest process.

## 3. Scenario and ground truth

Section [20, 500] m of the I-80 data (5 678 tracks in three periods), 6 lanes plus the auxiliary
lane of the on-ramp between 100 and 204 m, a buffer of 150 m downstream; every track of a
period is one vehicle inserted at the time, lane, position and speed of its first sample; lane
1 is an HOV lane (378, 415 and 408 eligible vehicles); LC2013 with SUMO's defaults except
`lcKeepRight = 0`. Downstream boundary: the speed of the data per lane and second, multiplied by
a feedback gain that keeps the number of simulated exits at the number of the data (D95).
Analysis window 180-840 s.

| Ground truth | Period 0 (16:00) | Period 1 (17:00) | Period 2 (17:15) |
|---|---:|---:|---:|
| Throughput at 450 m (veh/h) | 7 991 | 7 118 | 7 265 |
| Mean speed (m/s) | 7.71 | 5.25 | 5.21 |
| Congested share of the window | 0.68 | 1.00 | 1.00 |
| Peak 2-minute flow (veh/h) | 8 670 | 8 370 | 9 180 |
| Stop waves / speed of the waves by cross-correlation (m/s) | 3 / -5.43 | 3 / -5.56 | 5 / -5.41 |
| Wave amplitude (m/s) | 5.92 | 3.46 | 4.14 |
| Travel time median (s) | 57.4 | 78.6 | 82.4 |

The section is a queue from end to end in periods 1 and 2: no uncongested flow exists there,
so the "capacity drop" (1 - queue discharge over the peak 2-minute flow, D98) is a measure of
the variability of the flow, not of a breakdown. The wave speed by cross-correlation is the
same in the three periods and equals the textbook value; the speed of the leading edges of the
stop regions (-6.1 to -7.7 m/s) is faster because the waves grow as they travel upstream.

The first design of the boundary (the speed of the data alone) failed: the outflow is then the
law's own flow at that speed, a few percent above the observed flow for most laws, and the
section ran in free flow at 8-12 m/s with every law (D95). With the feedback the IDM laws
reproduce the queue (172 vehicles in the section against 179 of the data, 5.5 against 5.3 m/s
in period 1) and the gain settles at 0.73-0.84: the IDM laws would pass 16-27 % more flow than
the data at the observed downstream speed.

## 4. Laws of the corridor

All laws are fitted to NGSIM I-80 (D97): the calibrated IDM of the five driver folds, the
heterogeneous IDM of the interior per-event estimates (2 840 parameter sets, `v0` and `b` fixed,
D48), kNN fitted on the folds, and the HighD models of seed 0 fine-tuned on the folds (the
penalised ones with their penalty). Closed loop on the NGSIM test parts (mean over the five
folds) and audit of the members:

| Law | Spacing RMSE (m) | Collisions on the test events | Unstable among the equilibria | Stable inside the band |
|---|---:|---:|---:|---:|
| `idm_global` | 4.66 | 0.0 % | 0.00 | 1.00 |
| `residual_idm` (free core) | 4.28 | 0.0 % | 0.03 | 0.97 |
| `residual_idm_certified` | 4.89 | 0.0 % | 0.00 | 1.00 |
| `mlp` | 5.58 | 7.9 % | 0.96 | 0.04 |
| `mlp_penalty` (Jacobian, 1) | 5.44 | 14.9 % | 0.00 | 1.00 |
| `pidl` | 5.54 | 15.0 % | 0.91 | 0.08 |
| `gru` | 4.31 | 1.3 % | 0.72 | 0.27 |
| `gru_penalty` (rollout, 0.1) | 4.44 | 0.9 % | 0.36 | 0.64 |
| `lstm` | 4.51 | 6.4 % | 0.73 | 0.25 |
| `lstm_penalty` (rollout, 0.1) | 4.43 | 1.9 % | 0.50 | 0.50 |
| `perl` | 5.96 | 7.6 % | 0.94 | 0.06 |
| `knn` | 31.1 | 24.8 % | 0.92 | 0.08 |
| `idm_heterogeneous` | - | - | 0.09 | - |

On the congested NGSIM data the calibrated IDM (`T` 1.56 s) is string stable at every speed;
the memoryless networks collide on 8-15 % of the 48 s test events, kNN on 25 %.

## 5. Corridor results

Mean over the 30 runs of a law (3 periods x 10 seeds), 95 % bootstrap intervals over the runs.
Macro error: mean of the absolute relative errors of throughput, mean speed, queue discharge
flow, fundamental diagram, wave speed (cross-correlation), number of waves, wave amplitude and
travel time (Wasserstein-1) against the ground truth of the period (D99).

| Law | Macro error | Throughput | Mean speed | Wave speed | Waves | Travel time (W1) | Collisions per 1 000 veh-km | Vehicles in contact | Inserted | Gain g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `residual_idm_certified` | 0.125 [0.109, 0.144] | -2.9 % | +5.0 % | -2 % | -29 % | 0.080 | 0 | 0 % | 100 % | 0.73 |
| `residual_idm` | 0.158 [0.126, 0.207] | -1.8 % | +5.8 % | +47 % | -20 % | 0.082 | 0 | 0 % | 100 % | 0.73 |
| `idm_global` | 0.182 [0.176, 0.189] | -1.9 % | +3.3 % | +22 % | -74 % | 0.072 | 0 | 0 % | 100 % | 0.80 |
| `gru` | 0.192 [0.164, 0.223] | -4.3 % | -4.5 % | +26 % | -46 % | 0.150 | 206 | 8 % | 99 % | 1.37 |
| `gru_penalty` | 0.235 [0.185, 0.296] | -6.4 % | +4.0 % | +49 % | -36 % | 0.180 | 731 | | 99 % | 1.30 |
| `mlp_penalty` | 0.273 [0.254, 0.289] | -10.8 % | -1.2 % | +53 % | -43 % | 0.165 | 1 588 | | 98 % | 1.41 |
| `pidl` | 0.274 [0.245, 0.311] | -15.2 % | -4.7 % | +74 % | -21 % | 0.200 | 1 801 | | 97 % | 1.44 |
| `idm_heterogeneous` | 0.302 [0.255, 0.357] | -4.1 % | +21.0 % | +35 % | -51 % | 0.173 | 0 | 0 % | 100 % | 1.22 |
| `mlp` | 0.308 [0.274, 0.349] | -16.6 % | -4.5 % | +79 % | -49 % | 0.155 | 913 | 35 % | 96 % | 1.47 |
| `perl` | 0.431 [0.396, 0.464] | -4.1 % | -29.2 % | +93 % | -90 % | 0.609 | 11 683 | | 98 % | 1.48 |
| `knn` | 0.490 [0.460, 0.522] | -43.2 % | -34.7 % | +84 % | -12 % | 0.686 | 2 141 | | 69 % | 1.49 |
| `lstm` | 0.716 [0.662, 0.774] | -77.3 % | -73.7 % | +66 % | -7 % | 0.782 | 4 664 | | 43 % | 1.50 |
| `lstm_penalty` | 0.734 [0.673, 0.795] | -80.5 % | -83.2 % | +63 % | +7 % | 0.645 | 8 042 | | 39 % | 1.50 |

Wave speed: relative error of the cross-correlation estimate; a positive value means a slower
wave than the -5.4 to -5.6 m/s of the data (the IDM: -4.3 m/s; the certified hybrid: -5.6 m/s;
the memoryless networks: -1.2 to -2.6 m/s). Raw values and the other components are in
`laws.md`.

Reading:

1. **The three IDM-type laws with a stable core reproduce the corridor best** and without a
   single collision in 90 runs: throughput within 3 %, mean speed within 6 %, travel time
   within 1-3 % (the boundaries anchor these), and the certified hybrid also the wave speed
   (-5.6 m/s against -5.4 to -5.6) and, among the stable laws, the closest number of waves.
   The calibrated IDM damps the waves of the data (1.1 waves in the window against 3-5,
   -4.3 m/s).
2. **The heterogeneous IDM has too little capacity**: its interior parameter sets (large `T`,
   `s0`) hold back the entries (delay 24 s, gain 1.22), so the section is emptier and faster
   than the data (+21 % speed). The population of interior estimates is not the population that
   produced the traffic (D48 draws from the interior only).
3. **Every learned law collides in the corridor**, from 206 (GRU) to 11 700 (PERL) episodes per
   1 000 vehicle kilometres; the feedback gain sits at its ceiling (1.3-1.5) because the queues
   of these laws reach the entrance and hold the demand back (insertion delays 27-166 s, 4-61 %
   of the vehicles never enter for kNN and the LSTM laws). The MLP-type laws brake by at most
   about 3 m/s^2 into a queue and run into it. The collisions are not a matter of string
   stability: `mlp_penalty` is string stable at every speed and still has 1 588 episodes per
   1 000 vehicle-km.
4. **The penalties do not help in the corridor.** The penalised GRU has a larger macro error
   and more collisions than the free GRU (0.235 against 0.192; 731 against 206 episodes per
   1 000 vehicle-km); the penalised MLP has a smaller macro error (0.273 against 0.308) but more
   collisions (1 588 against 913). Neither penalty makes a learned law usable in a congested
   corridor.
5. **The LSTM laws collapse** (43 % and 39 % of the vehicles inserted, 30 teleports per run):
   the standstill regime of the queue is outside their training data, and once a lane stops
   they do not restart.

## 6. Correlation of macro error and instability (D100)

Over the 13 laws, with 1 000 bootstrap resamples of the laws:

| Instability measure | Laws | Spearman | Pearson |
|---|---:|---:|---:|
| Unstable among the equilibria of the members | 13 | 0.66 [0.24, 0.85] | 0.47 [0.15, 0.79] |
| Band share "not stable" | 12 | 0.65 [0.18, 0.88] | 0.47 [0.07, 0.82] |
| Unstable among the equilibria, laws without collisions | 4 | 0.63 [-1, 1] | 0.91 [-1, 1] |

**The correlation is positive and its interval excludes zero**: the laws whose members are
string unstable at more speeds reproduce the corridor worse. The correlation is carried by the
learned laws, which collide; among the four laws without collisions the sample is too small for
an interval.

## 7. Equivalence of the certified hybrid and the IDM (TOST, margin 10 %)

30 paired runs (period and seed), `residual_idm_certified` against `idm_global`:

| Metric | IDM | Certified hybrid | Relative difference | Equivalent within 10 % |
|---|---:|---:|---:|---|
| Throughput (veh/h) | 7 323 | 7 246 | -1.1 % [-1.4, -0.7] | yes |
| Mean speed (m/s) | 6.27 | 6.38 | +1.8 % [+1.4, +2.3] | yes |
| Queue discharge flow (veh/h) | 7 257 | 6 896 | -5.0 % [-6.7, -3.0] | yes |
| Peak 2-minute flow (veh/h) | 8 582 | 8 691 | +1.3 % | yes |
| Travel time mean / median (s) | 68.9 / 65.3 | 68.0 / 65.0 | -1.2 % / -0.6 % | yes / yes |
| Inserted share | 1.00 | 1.00 | 0 | yes |
| Waves in the window | 1.1 | 2.6 | +136 % | no |
| Wave speed by cross-correlation (m/s) | -4.28 | -5.59 | +31 % | no |
| Wave amplitude (m/s) | 3.76 | 4.44 | +18 % | no |
| FD scatter (veh/h) | 582 | 893 | +54 % | no |
| Congested share | 0.72 | 0.30 | -59 % | no |
| Capacity drop | 0.154 | 0.206 | +34 % | no |

The two laws are equivalent in every first-order quantity (flows, speed, travel times) and
differ in the oscillation pattern: the certified hybrid lets the waves of the data through
(2.6 waves at -5.6 m/s, amplitude 4.4 m/s: closer to the data's 3-5 waves at -5.4 to -5.6 m/s and
3.5-5.9 m/s) while the IDM damps them.

## 8. Deviations from the specification

| Id | Deviation | Reason |
|---|---|---|
| E9 | Three processes (build, simulation, metrics); short paths; no parquet in the simulation. | `libsumo` and `pyarrow` cannot share a process; SUMO cannot open the path of the repository. |
| D94, D103 | Insertion of every track at its first sample; `lcKeepRight = 0`; HOV lane 1. | Facts of the data. |
| D95 | Downstream boundary = speed of the data times a feedback gain on the cumulative exits. | With the speed alone the section runs in free flow with every law. |
| D96 | SUMO's Euler update instead of the ballistic update with `setPreviousSpeed`; leader range 50 m; virtual leader at the end of the auxiliary lane. | The scheme of the training; the training data of the laws end at 49 m. |
| D97 | 13 laws instead of 11, all fitted to NGSIM I-80; the corridor uses the fine-tuned HighD models. | Accepted proposal 1 of the M4 review; the corridor drives at 0-15 m/s. |
| D98, D99, D100 | Definitions of the metrics, of the macro error and of the statistics. | The specification names the quantities only. |
| D101 | No I-24 builder. | The data are not on the machine. |
| D102 | Collisions keep the vehicle (contact episodes) instead of removing it. | Removal drained the queues and hid the failure. |
| D104 | Insertion checks of SUMO with `tau = 1.0`, `decel = 8`. | The overlap check alone caused collisions at insertion. |

## 9. Open points for the review

1. **The macro-error vector mixes the anchored and the free quantities.** With the boundary
   feedback, throughput, mean speed and travel time are matched by construction for every law
   that keeps up, so the differences between the stable laws lie in the waves, the scatter of
   the fundamental diagram and the collisions. Proposal for M6: report the components
   separately, name the anchored ones as such, and give the correlation of section 6 also for
   a "dynamic" macro error made of the wave components and the FD scatter alone.
2. **Heterogeneous IDM.** The interior estimates of D48 give a population with too little
   capacity. Proposal: keep the result as it is (it is a finding about the identifiability of
   the per-event estimates, the topic of the sibling project) and add, in M6, a variant with
   `T`, `s0`, `a` drawn from all estimates clipped to the bounds, if the review wants it.
3. **Learned laws in congestion.** Every learned law collides at standstill and queue tails
   that are rare in the training events (48 s of following at 0-15 m/s). A safety layer
   (a speed cap as in SUMO's Krauss model) would remove the collisions and change the laws;
   proposal: no safety layer in the paper, the collision rate is the result, and the discussion
   says that the audit of string stability does not cover the queue-tail regime.
4. **Wave metrics.** The number of waves depends on the thresholds of D98 (median-relative
   speed, minimum size); the cross-correlation wave speed does not. Proposal: the wave speed by
   cross-correlation and the wave amplitude are the headline wave numbers of the paper, the
   count is reported with its settings.

## 10. Tests

475 tests: 475 passed, 20 skipped (GPU variants). New in M5: scenario builder (counts per period
and lane, lane mapping, XML read back, `netconvert`, boundary speeds and exits), control loop in
child processes (conservation, no teleport with a stable law, state handed to the law, member
assignment, reproducibility, restart, feedback, rate limit, leader range, insertion, contact
episodes, HOV permissions, pyarrow never loaded), metrics on hand-made trajectories (uniform
flow, a synthetic wave at -5 m/s by both estimators, detectors, macro error of identical inputs,
missing files), tables on a hand-made tree of runs (means, correlation, TOST).
