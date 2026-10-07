# Milestone M1 report: data pipeline, IDM calibration, persistence baseline

Date: 2026-09-27. Machine: Windows 10, i7-12700F, 32 GB RAM, RTX 6000 Ada (48 GB).
Python 3.11.8, torch 2.11.0+cu128, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, pyarrow 25.0.1,
hydra-core 1.3.7, osqp 1.1.3. Nothing is committed to git yet.

## 1. Commands run

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

Not run as commands (disk full, section 6): `extract_events.py data=ngsim_i80` and
`data=ngsim_us101`. Their numbers below come from the same code executed in memory without
writing the reconstruction cache and the event files.

## 2. Data

| Source | Files | Size | Licence | Status |
|---|---|---|---|---|
| OpenACC, 6 platoon campaigns | 139 runs + notes | 501 MB | CC BY 4.0 | downloaded |
| FollowNet event sets (HighD, NGSIM I-80, Waymo) | 3 | 176 MB | CC0 | downloaded, sha256 recorded |
| Waymo car-following pairs (Hu et al. 2022) | 5 | 52 MB | CC BY 4.0 | downloaded, sha256 verified |
| NGSIM raw I-80 | 4 566 387 rows | 150 MB parquet | public domain | downloaded |
| NGSIM raw US-101 | 4 802 933 rows | 159 MB parquet | public domain | downloaded |
| NGSIM reconstructed (Montanino-Punzo) | - | - | - | not public; own reconstruction |
| highD raw | - | - | levelXdata | **not available** (D22) |

## 3. Extracted events

| Event set | Events | Samples | Followers | Median duration | Remarks |
|---|---:|---:|---:|---:|---|
| openacc | 4 703 | 2 553 092 | 49 | 56.2 s | ACC 4 010, Human 662, unknown 31; chunks of at most 60 s; 4 205 samples with IVS <= 0 removed |
| follownet_highd | 12 512 | 1 876 800 | 12 512 | 15.0 s | 29 of 12 541 published events reach 120 m and are dropped |
| follownet_ngsim_i80 | 1 931 | 733 465 | 1 916 | 35.4 s | 1 916 of 1 930 kept, 15 split |
| follownet_waymo | 881 | 171 890 | 881 | 19.8 s | 881 of 1 440 kept: 547 stand still more than 20 % of the time, 212 contain speeds <= -0.1 m/s |
| waymo | 971 | 187 606 | 853 | 19.8 s | HV-HV 666, AV follows HV 119, HV follows AV 186 |
| ngsim_i80 (in memory) | 7 647 | 3 701 692 | 5 520 | 38.2 s | periods 2 634 / 2 492 / 2 521, lanes 1-6 |
| ngsim_us101 (in memory) | see section 5 | | | | |

OpenACC events per campaign: AstaZero 906, Casale 77, Cherasco 392, JRC 88, Vicolungo 505,
ZalaZone 2 735.

| Event set | Spacing p05 / p50 / p95 (m) | Speed p05 / p50 / p95 (m/s) | Time gap p50 (s) | std(a) (m/s^2) | slope d(s)/dt vs dv |
|---|---|---|---:|---:|---:|
| openacc | 10.7 / 22.0 / 52.5 | 7.0 / 13.2 / 33.1 | 1.62 | 0.45 | -0.87 |
| follownet_highd | 10.7 / 26.2 / 55.7 | 8.7 / 22.5 / 25.5 | 1.29 | 0.36 | -1.01 |
| follownet_ngsim_i80 | 4.2 / 12.5 / 36.1 | 2.0 / 7.5 / 16.0 | 1.72 | 0.84 | -0.99 |
| follownet_waymo | 3.5 / 14.8 / 46.3 | 0.4 / 7.6 / 22.3 | 1.83 | 0.65 | -1.00 |
| waymo | 3.2 / 14.1 / 46.0 | 0.4 / 7.5 / 22.5 | 1.78 | 0.66 | -1.00 |

The last column is the check of the sign convention on real data: with `dv = v - v_lead` the
spacing must shrink at the rate `dv` (expected slope -1). The OpenACC value reflects the noise
of two independent measurements (GNSS spacing, Doppler speed), not a sign problem.

Splits: `data/splits/<name>_driver.json` and `<name>_site.json` for every event set (5 folds,
seed 0). Site-level folds: OpenACC 5 folds over 6 campaigns; the FollowNet sets and Waymo have
one site (no site split); driver-level folds of the FollowNet sets are event-level folds (D19).

## 4. IDM calibration and persistence baseline

Per event (closed loop over the whole event, `J = NRMSE(s) + NRMSE(v)`):

| Event set | J median | RMSE s median / mean (m) | RMSE v median / mean (m/s) | Collisions | Events with a parameter on a bound | Time (GPU) |
|---|---:|---:|---:|---:|---:|---:|
| openacc | 0.073 | 1.08 / 1.64 | 0.30 / 0.42 | 0.02 % | 90.8 % | 158 s |
| follownet_highd | 0.020 | 0.28 / 0.39 | 0.17 / 0.21 | 0 | 91.3 % | 130 s |
| follownet_ngsim_i80 | 0.159 | 1.11 / 1.30 | 0.61 / 0.63 | 0 | 84.0 % | 99 s |
| follownet_waymo | 0.074 | 0.44 / 0.65 | 0.31 / 0.37 | 0 | 90.4 % | 30 s |
| waymo | 0.074 | 0.43 / 0.63 | 0.31 / 0.37 | 0 | 89.3 % | 31 s |

Every event converged (at most 272 generations, median 53-80); time is for three runs of the
optimiser. Check on 25 random OpenACC and 25 HighD events against an independent numpy
implementation of the objective minimised with `scipy.optimize.differential_evolution`: the
objective values of the two implementations agree to 3e-17 at the stored parameters; the stored
optimum is at most 2e-4 above SciPy's (with or without polishing) and up to 1.5e-3 below;
SciPy puts 88-96 % of the same events on a bound.

Spread of the per-event estimates (median; share on the lower / upper bound):

| Event set | v0 (m/s) | T (s) | s0 (m) | a (m/s^2) | b (m/s^2) |
|---|---|---|---|---|---|
| openacc | 42.9; 4 % / 36 % | 0.74; 11 % / 1 % | 9.94; 8 % / 53 % | 1.08; 9 % / 4 % | 5.97; 15 % / 53 % |
| follownet_highd | 37.1; 1 % / 31 % | 0.50; 37 % / 2 % | 6.47; 24 % / 27 % | 0.75; 23 % / 11 % | 5.96; 15 % / 52 % |
| follownet_ngsim_i80 | 18.3; 17 % / 10 % | 0.92; 17 % / 3 % | 3.82; 21 % / 15 % | 1.12; 7 % / 8 % | 2.29; 26 % / 31 % |
| waymo | 20.3; 22 % / 12 % | 0.93; 18 % / 5 % | 3.60; 17 % / 17 % | 1.49; 9 % / 13 % | 2.30; 28 % / 35 % |

OpenACC by driving mode (medians per campaign): ACC followers `T` 0.55-1.08 s and `s0`
8.5-10.0 m (upper bound 10 m); human drivers `T` 0.32-1.24 s and `s0` 4.8-9.3 m.

Global parameter set per dataset (errors pooled over the events):

| Event set | v0 | T | s0 | a | b | J | RMSE s median / mean (m) | RMSE v median / mean (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openacc | 45.0* | 0.86 | 10.0* | 1.89 | 0.50* | 0.414 | 5.48 / 7.73 | 0.61 / 0.77 |
| follownet_highd | 29.3 | 0.67 | 1.99 | 0.35 | 0.50* | 0.191 | 2.92 / 3.80 | 0.53 / 0.64 |
| follownet_ngsim_i80 | 27.4 | 1.69 | 0.65 | 0.77 | 0.50* | 0.441 | 4.32 / 5.41 | 0.82 / 0.90 |
| follownet_waymo | 45.0* | 1.66 | 1.81 | 1.38 | 0.50* | 0.389 | 3.52 / 5.11 | 0.71 / 0.85 |
| waymo | 45.0* | 1.61 | 1.68 | 1.38 | 0.50* | 0.382 | 3.27 / 4.89 | 0.70 / 0.84 |

`*` on a bound. Full tables: `data/calibration/<name>/idm_per_event.parquet`, `idm_global.json`,
`idm_spread.json`; metrics: `runs/baselines/<name>/idm/metrics.json`.

Persistence baseline (`runs/baselines/<name>/persistence/metrics.json`):

| Event set | Open loop RMSE a, `a_hat[k+1] = a[k]` (m/s^2) | Open loop RMSE a, model on the state history | Closed loop RMSE s median / mean (m) | RMSE v median / mean (m/s) | Collisions |
|---|---:|---:|---:|---:|---:|
| openacc | 0.041 | 0.144 | 101.7 / 164.2 | 5.38 / 7.88 | 52.9 % |
| follownet_highd | 0.018 | 0.016 | 5.59 / 8.07 | 1.11 / 1.58 | 18.9 % |
| follownet_ngsim_i80 | 0.139 | 0.076 | 81.6 / 130.2 | 5.76 / 7.82 | 62.8 % |
| follownet_waymo | 0.051 | 0.033 | 17.7 / 23.1 | 2.67 / 3.37 | 37.3 % |
| waymo | 0.056 | 0.031 | 17.9 / 23.4 | 2.72 / 3.44 | 40.2 % |

Reading: one step ahead the persistence error is 5-17 % of the standard deviation of the
acceleration, so an open-loop acceleration error cannot rank models; in closed loop the same
baseline is useless. Only the closed-loop metrics are meaningful in the later tables.

## 5. NGSIM reconstruction (real data, in memory)

Constrained smoothing of the longitudinal positions (D42-D46), all tracks of both sites,
single process, no cache written.

| | I-80 | US-101 |
|---|---:|---:|
| Raw rows / exact duplicate rows removed | 4 566 387 / 0 | 4 802 933 / 704 000 |
| Tracks (three periods) | 5 678 (2 052 + 1 836 + 1 790) | 6 101 |
| Wall time, load to events | 269 s | 106 s |
| Tracks re-solved with the leader constraint, per pass | 1039, 369, 189, 114, 94 | 386, 227, 107, 61, 43 |
| Solver status other than "solved" | 4 | 2 |
| RMS correction of the positions | 0.32 m | 0.22 m |
| Frames with net gap <= 0, raw -> reconstructed | 0.59 % -> 0.11 % | 0.15 % -> 0.06 % |
| Frames with net gap < 0.5 m | 0.80 % -> 0.13 % | 0.18 % -> 0.06 % |
| Leader frames left unconstrained (overlap > 5 m) | 0.12 % | 0.06 % |
| abs(acceleration), 99 % quantile / maximum | 17.0 / 898 -> 4.0 / 8.0 m/s^2 | 6.5 / 523 -> 2.9 / 8.0 m/s^2 |
| abs(jerk), 99 % quantile / maximum | 280 / 9 266 -> 7.8 / 15.2 m/s^3 | 66 / 9 192 -> 5.1 / 15.0 m/s^3 |

Macroscopic agreement of raw and reconstructed trajectories (Edie, cells of 50 m x 10 s):

| | I-80, all lanes | I-80, per lane | US-101, all lanes | US-101, per lane |
|---|---:|---:|---:|---:|
| Cells | 2 841 | 16 218 | 3 717 | 17 818 |
| Total distance, relative difference | -6e-5 | -6e-5 | -6e-6 | -6e-6 |
| Total time, relative difference | 0 | 0 | 0 | 0 |
| Cell speed: RMSE (m/s) / correlation | 0.024 / 0.99995 | 0.050 / 0.99993 | 0.020 / 0.99999 | 0.035 / 0.99996 |
| Flow, relative error | 0.12 % | 0.20 % | 0.05 % | 0.08 % |
| Density, relative error | 0.13 % | 0.18 % | 0.09 % | 0.12 % |

Events from the reconstructed trajectories (mainline lanes, not stored yet):

| | I-80 | US-101 |
|---|---:|---:|
| Events / samples / followers | 7 647 / 3 701 692 / 5 520 | 7 351 / 3 587 957 / 5 866 |
| Events per period | 2 634 / 2 492 / 2 521 | 2 371 / 2 503 / 2 477 |
| Candidate runs / too short / follower standing | 13 853 / 5 940 / 266 | 12 407 / 5 009 / 47 |
| Median duration | 38.2 s | 45.7 s |
| Spacing p05 / p50 / p95 | 2.7 / 9.9 / 30.2 m | 4.4 / 13.8 / 35.6 m |
| Speed p05 / p50 / p95 | 0.5 / 5.5 / 13.3 m/s | 1.5 / 9.1 / 15.3 m/s |
| std(a), 99 % quantile of abs(a) | 0.98, 3.65 m/s^2 | 0.90, 2.70 m/s^2 |

Reading: the reconstruction removes the impossible accelerations and four fifths of the
overlaps while the macroscopic picture is unchanged (flow and density within 0.2 %). The
remaining overlaps sit on frames where the raw data put the follower more than 5 m into its
leader; these samples are cut by the extraction rule `s > 0`. On I-80 at least 1 % of the
reconstructed accelerations sit on the upper bound of 4 m/s^2, which indicates that the bound,
not the smoother, limits the noise there; the cut-off frequency (0.5 Hz) and the bounds are
configuration values to revisit when the corridor ground truth is built (M5).

## 6. Deviations from the specification and open points

| Id | Deviation | Reason |
|---|---|---|
| D22 | No raw highD. The FollowNet HighD events stand in for it. | No levelXdata agreement; events of 15 s without driver or site ids. E1-E4 must be re-run when raw highD arrives. |
| D14 | OpenACC `IVS_i` is the gap between vehicles `i` and `i + 1`. | The specification text is contradicted by the campaign notes and by the files. |
| D5 | Minimum duration is 150 samples (`n * dt >= 15 s`). | FollowNet's own rule; otherwise every FollowNet HighD event is rejected. |
| D10, D34, D47 | Differential evolution is a batched torch implementation with a tighter stopping rule than SciPy's default and three restarts. | Per-event calibration of 12 512 events in 130 s; parameters are not settled at SciPy's default tolerance; single runs end in a local optimum in a few per cent of the events. |
| D17, D18 | FollowNet relative speed is negated; FollowNet NGSIM and Waymo spacing is reduced by 4.5 m. | Their relative speed is `v_lead - v`; their spacing is gross. |
| D7 | OpenACC runs are cut into events of at most 60 s. | Runs last tens of minutes. |
| D4 | "Speed > 0" is `v > 0.1 m/s`. | Measured speeds of standing vehicles are not exactly zero. |
| D24 | Waymo pairs were downloaded outside `download_data.py`. | Mendeley Data answers the Python client with a browser challenge. |
| - | NGSIM events and the reconstruction cache are not on disk; manifests of the five stored event sets were written before the option `min_spacing` was added (same events, older config hash). | Drive C: had 0.3 GB free. One re-run of the extraction commands fixes both. |

Open points for the review:

1. **Parameters on the bounds.** 84-91 % of the per-event estimates have at least one parameter
   on a bound (b for 56-68 % of the events, s0 for 61 % of the OpenACC events, T = 0.3 s for
   37 % of the HighD events); SciPy's optimiser gives the same picture, so this is a property
   of the problem and not of the optimiser. Events of 15-60 s do not excite all five parameters. The spread
   of the estimates is therefore not a distribution of driver characteristics, and sampling it
   for the heterogeneous IDM of the corridor (M5) would create vehicles with a desired speed of
   45 m/s and extreme reactions. Options: calibrate on longer events, fix `v0` and `b` and
   calibrate `T, s0, a`, or regularise towards the global set. Plan P2 addresses this.
2. **Global IDM of HighD has a = 0.35 m/s^2.** The HighD events contain almost no acceleration
   from low speed (5 % quantile of the speed 8.7 m/s). For the I-80 corridor the IDM should be
   calibrated on the NGSIM events of the same site.
3. **Minimum spacing.** 0.9-1.4 % of the events of OpenACC, FollowNet NGSIM and Waymo come
   closer than 0.5 m; proposal `extraction.min_spacing = 0.5` (D41).
4. **Noise in the inputs.** Stored speeds are not smoothed (D3). The effect of the noise of `dv`
   on the learned partial derivatives should be measured in M3 (`extraction.smooth_speeds`).
5. **Disk space.** About 1 GB is needed for the NGSIM cache and events, more for M2-M4.

## 6a. Addendum of 2026-09-28: numbers after the review

Decisions of the review (D41, D48-D50): `min_spacing = 0.5` m, heterogeneous IDM with fixed
`v0` and `b`, corridor IDM calibrated on `ngsim_i80`, minimum gap of the reconstruction 0.75 m.
All event sets were extracted again, the NGSIM sets are on disk, all calibrations were repeated
(`scripts\run_m1.ps1`). These tables replace those of sections 3-5 where they differ.

| Event set | Events | Samples | Followers | Config hash |
|---|---:|---:|---:|---|
| openacc | 4 715 | 2 556 685 | 49 | 3cb07fa3876f |
| follownet_highd | 12 512 | 1 876 800 | 12 512 | 22fd3e32d190 |
| follownet_ngsim_i80 | 1 937 | 732 546 | 1 916 | 73f98291456e |
| follownet_waymo | 880 | 171 609 | 880 | c4905e432fb3 |
| waymo | 967 | 186 711 | 849 | 6ca7ac18bee8 |
| ngsim_i80 | 7 651 | 3 705 762 | 5 521 | 7e35cf19e55e |
| ngsim_us101 | 7 353 | 3 588 946 | 5 866 | 98ea95fca13c |

IDM, five parameters per event (left) and one global set (right; `b` is on its lower bound
everywhere, `v0` on its upper bound where it reads 45.0, `s0` for OpenACC):

| Event set | J median | RMSE s median / mean (m) | On a bound | v0 | T | s0 | a | b | J global | RMSE s median / mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openacc | 0.073 | 1.08 / 1.75 | 90.7 % | 45.0 | 0.86 | 10.00 | 1.96 | 0.50 | 0.420 | 5.50 / 7.76 |
| follownet_highd | 0.020 | 0.28 / 0.39 | 91.3 % | 29.3 | 0.67 | 1.99 | 0.35 | 0.50 | 0.191 | 2.92 / 3.80 |
| follownet_ngsim_i80 | 0.159 | 1.11 / 1.30 | 83.7 % | 26.9 | 1.68 | 0.69 | 0.77 | 0.50 | 0.441 | 4.29 / 5.40 |
| follownet_waymo | 0.074 | 0.44 / 0.65 | 90.5 % | 45.0 | 1.65 | 1.87 | 1.39 | 0.50 | 0.389 | 3.51 / 5.12 |
| waymo | 0.074 | 0.43 / 0.63 | 89.3 % | 45.0 | 1.60 | 1.64 | 1.37 | 0.50 | 0.382 | 3.29 / 4.91 |
| ngsim_i80 | 0.217 | 1.30 / 1.48 | 83.6 % | 26.6 | 1.57 | 1.68 | 0.88 | 0.50 | 0.483 | 3.78 / 4.86 |
| ngsim_us101 | 0.146 | 1.21 / 1.42 | 79.9 % | 21.8 | 1.47 | 1.58 | 1.04 | 0.50 | 0.460 | 4.71 / 5.90 |

Heterogeneous IDM (`v0`, `b` fixed at the global values, `T, s0, a` per event):

| Event set | v0 | b | J median | RMSE s median / mean (m) | Events with no parameter on a bound | T median (5 % - 95 %) | s0 median | a median |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| ngsim_i80 | 26.6 | 0.50 | 0.254 | 1.58 / 1.82 | 37.1 % | 1.09 (0.30 - 2.95) | 2.49 | 0.80 |
| ngsim_us101 | 21.8 | 0.50 | 0.179 | 1.59 / 1.85 | 41.5 % | 0.95 (0.30 - 2.63) | 3.31 | 0.92 |

Fixing `v0` and `b` costs 0.3-0.4 m of spacing RMSE per event and doubles the share of interior
estimates (from 16-20 % to 37-42 %), but `s0` still sits on its lower bound in 27-31 % of the
events and `T` in 18-19 %. For the corridor (M5) the vehicles should be drawn from the interior
estimates only (about 2 800 events on I-80).

Persistence in closed loop: spacing RMSE median 5.6 m (HighD) to 104 m (NGSIM), collisions
19-63 %; one-step acceleration RMSE 0.018-0.19 m/s^2.

## 7. Tests

136 tests: event schema and parquet round trip, resampling, Savitzky-Golay acceleration,
extraction invariants (no non-positive spacing, spacing below 120 m, at least 150 samples,
single leader, no lane change, moving fraction, sign of `dv`), splits, loaders of OpenACC,
FollowNet, highD (synthetic recording) and Waymo, NGSIM loader, reconstruction bounds and
macroscopic agreement, Edie's definitions, IDM equilibrium and signs, closed-loop integration,
batched differential evolution against SciPy, calibration, persistence, hydra entry points.
