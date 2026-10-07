# Data contract (M1)

Binding for every module. Deviations must be recorded in `docs/decisions.md`.

## 1. Conventions

| Item | Convention |
|---|---|
| Units | SI everywhere (m, s, m/s, m/s²). NGSIM feet are converted at load time (×0.3048). |
| Sampling | 10 Hz, `DT = 0.1` s (`cf_stability.data.schema.DT`). |
| Relative speed | `dv = v - v_lead` (approach rate, **positive when closing in**). Never the opposite sign. |
| Spacing | `s` is the **net gap** (leader rear bumper to follower front bumper), m. |
| Positions | `x_lead` = leader **rear** bumper, `x_follower` = follower **front** bumper, same longitudinal axis, increasing in the driving direction. Hence `s = x_lead - x_follower` exactly. |
| Positions without a common frame | If a source has no usable longitudinal coordinate (OpenACC, FollowNet): `x_lead = cumulative trapezoid integral of v_lead`, starting at 0, and `x_follower = x_lead - s`. |
| Event time | `t[0] = 0`; the source time of the first sample goes to `meta["t0"]`. |
| Speeds stored | `v`, `v_lead` are the measured (resampled) speeds, **not** smoothed. Switch `extraction.smooth_speeds` (default `false`) stores the Savitzky-Golay filtered speeds instead. |
| Acceleration | Dataset acceleration if the source provides one; otherwise central differences of the Savitzky-Golay filtered follower speed (window 1.1 s = 11 samples, order 2). `meta["a_source"]` is `"dataset"` or `"savgol"`. |
| Integration | semi-implicit Euler at 0.1 s: `v[k+1] = max(v[k] + a[k]*dt, 0)`, `x[k+1] = x[k] + v[k+1]*dt`, `s[k+1] = x_lead[k+1] - x[k+1]`. |
| Acceleration clip | `[-8, 4]` m/s² in every closed-loop rollout. |
| Collision | `s <= 0` at any step of a rollout. |
| Identifiers | `follower_id = "<dataset>/<site>/<raw follower id>"`, `leader_id` likewise, `event_id = "<follower_id>|<raw leader id>|<k>"` with `k` the running index of the segment for that pair; when the series carries a recording tag (`PairSeries.run`), `event_id = "<follower_id>|<raw leader id>|<run>|<k>"`. All ids are strings and globally unique. |

## 2. Event schema

`cf_stability.data.schema.Event`: arrays `t, s, dv, v, a, v_lead, x_lead, x_follower`
(float64, equal length), ids `dataset, site, follower_id, leader_id`, `event_id`, and a
JSON-serialisable `meta` dict. `Event.validate()` checks: uniform time step, finite values,
`s > 0`, non-negative speeds, `dv == v - v_lead`, `s == x_lead - x_follower`.

`EventSet` stores a dataset as a directory `data/events/<dataset>/`:

* `samples.parquet` - long format, one row per sample: `event_id` + the eight arrays;
* `events.parquet` - one row per event: `event_id, dataset, site, follower_id, leader_id, n_samples, duration, meta_json`;
* `manifest.json` - config, config hash, counts, source files.

## 3. Intermediate representation for loaders

Loaders of track-based sources produce `PairSeries` objects (`cf_stability.data.extraction`):
the follower's series on a uniform 10 Hz grid together with its instantaneous leader.

| Field | Meaning |
|---|---|
| `dataset, site, follower_id` | ids (follower_id already globally unique) |
| `t` | uniform 10 Hz grid, s (source time) |
| `x_follower, v` | follower front-bumper position and speed; NaN where unobserved |
| `a` | dataset acceleration or `None` |
| `leader_id` | object array of globally unique leader ids, `None` where there is no leader |
| `x_lead, v_lead` | leader rear-bumper position and speed at the same instants, NaN without leader |
| `lane_follower, lane_leader` | integer lane ids or `None` when the source has no lanes |
| `segment_key` | optional array; a segment boundary is placed wherever consecutive values differ (copied to `meta["segment_key"]`) |
| `run` | optional tag of the recording (file, segment); becomes part of the event id and `meta["run"]` |
| `meta` | dict copied into every event (`leader_length` etc.) |

Sources that are already segmented into events (FollowNet) build `Event` objects directly,
and still pass through the duration / spacing / moving-fraction filters.

## 4. Extraction criteria (FollowNet-compatible)

A sample is *valid* when the leader exists, all values are finite, `0 < s < 120` m and both
speeds are non-negative. A segment boundary is placed wherever a sample is invalid, the
leader id changes, the lane of the follower or of the leader changes, or the time grid has a
gap. Events are the maximal runs of valid samples with

1. a single leader and no lane change of either vehicle (by construction of the runs);
2. duration `n * dt >= 15` s, i.e. at least 150 samples (the rule FollowNet applied: its shortest events have exactly 150 samples at 10 Hz and 375 frames at 25 Hz);
3. spacing `< 120` m at every sample (by construction);
4. follower moving (`v > moving_speed_eps`, default 0.1 m/s) for at least 80 % of the samples; otherwise the event is dropped.

Optional `max_duration` (must be `>= 2 * min_duration`): a run longer than that is cut into
`ceil(n / n_max)` consecutive chunks of nearly equal length (used for OpenACC, whose platoon
runs last tens of minutes; `null` for the other datasets).

## 4a. Source formats (verified on the files, 2026-09-27)

**OpenACC** (JRC, CC BY 4.0). Five metadata lines (`Date`, `Vehicle_order`, `Number_of_vehicles`,
`ACC`, `Distance_setting`; trailing commas possible), then a table whose columns are identified
by name and vehicle index: `Time`, `Speed<i>`, `Lat<i>`, `Lon<i>`, `Alt<i>`, `E<i>`, `N<i>`, `U<i>`,
`VE<i>`, `VN<i>`, `VU<i>`, `Driver<i>`, `IVS<i>`. The set and the order of the columns differ
between campaigns and files.

* `IVS<i>` is the bumper-to-bumper gap between vehicle `i` and vehicle `i + 1`, i.e. the spacing
  of **follower `i + 1`** to its leader `i` (the campaign notes are explicit; the project
  specification describes it the other way round).
* `Driver<i>` is `"Human"` or `"ACC"` per sample; the column is missing for some vehicles and
  for the whole ZalaZone campaign. Header flag `ACC`: 0 manual, 1 ACC, 2 mixed, empty = unknown.
* ZalaZone: vehicles without measurements are listed in `Vehicle_order` but their columns are
  absent, so the indices present in the table have holes; a pair is usable only when `IVS<i>`,
  `Speed<i>` and `Speed<i+1>` all exist.
* Cells can be empty, IVS can be negative for short periods, time can have gaps (tunnels, tolls).
* `Time` is a common time frame in seconds with an arbitrary origin; speeds are raw Doppler speeds,
  unfiltered in most campaigns.

**FollowNet event sets** (figshare, CC0). One array per dataset, each event has four rows:
`spacing`, `follower speed`, `relative speed`, `leader speed`.

* `relative speed = v_lead - v_follower` (verified: equals row 4 - row 2 exactly and matches
  d(spacing)/dt). It is the opposite of the project convention: `dv = -relative speed`.
* HighD: dense float array `[12541, 4, 375]`, 25 Hz, every event exactly 15 s; NGSIM I-80 and
  Waymo: object arrays (pickled), 10 Hz, 150-914 and 150-199 samples.
* Spacing of NGSIM I-80 and Waymo is a gross distance that contains a vehicle length (median
  at standstill 7.6 m and 8.0 m, 1 % quantile 4.7 m and 4.9 m). HighD spacing is `dhw` of
  the source, taken as the net gap until it can be checked against the raw highD tracks.
* No vehicle, recording or site identifiers; a few events violate FollowNet's own criteria
  (HighD: 29 events reach 120 m; Waymo: 2 events with non-positive spacing, 547 of 1440 events
  with the follower standing for more than 20 % of the time).

**Waymo car-following pairs** (Hu et al. 2022, Mendeley Data 10.17632/wfn2c3437n.2, CC BY 4.0).
`all_seg_paired_cf_trj_final_with_large_vehicle.csv` with columns `segment_id, local_veh_id,
length, local_time, follower_id, leader_id, filter_pos, filter_speed, filter_accer`.

* A pair is (segment_id, follower_id, leader_id); the rows of both vehicles are in the file and
  `local_veh_id` says which one a row describes. Vehicle 0 is the Waymo AV (length 5.18 m).
* 10 Hz, 20 s segments. `filter_pos` is the position of the **centre** of the bounding box along
  the path, so the net gap is `pos_lead - pos_follower - (L_lead + L_follower) / 2` (as in the
  authors' `waymo_data_process.py`). Filtered accelerations are provided.
* The FollowNet Waymo spacing equals the centre-to-centre distance of these pairs.

**NGSIM raw** (data.transportation.gov `8ect-6jqj`, public domain). 25 columns, feet and ft/s,
10 Hz; `local_y` is the longitudinal position of the front centre, `preceding = 0` means no leader.
Verified on I-80 (4 566 387 rows):

* The combined table has no period column. `global_time - 100 * frame_id` takes exactly three
  values (the epoch of frame 0 of each original file) and identifies the period; vehicle ids
  restart in every period, so the track key is (period, vehicle_id). 2052 + 1836 + 1790 tracks,
  every one contiguous in time with `total_frames` rows.
* The data of the three files span 15:58:55-16:15:36, 16:59:27-17:15:46 and 17:12:45-17:32:14.
  Files 2 and 3 overlap by 180 s but hold different vehicles (a vehicle belongs to the file
  of its entry time): nothing is duplicated.
* `space_headway` is the front-to-front distance; the net gap is
  `local_y_lead - local_y - v_length_lead`. It is non-positive on 0.59 % of the rows with a leader.
* `v_acc` is clipped at +-3.41 m/s^2 (9 % of the rows sit on the limit): raw accelerations are
  not used. Lanes 1-6 mainline, 7 on-ramp.

## 5. Splits

`data/splits/<dataset>_driver.json` and `data/splits/<dataset>_site.json`:

```json
{"dataset": "...", "kind": "driver", "n_folds": 5, "seed": 0,
 "groups": {"<group>": 3}, "folds": {"<event_id>": 3}}
```

Groups are `follower_id` (driver level) or `"<dataset>/<site>"` (site level). Groups are
shuffled with `numpy.random.default_rng(seed)` and dealt to folds greedily so that the folds
are balanced in number of samples. For fold `k`: test = fold `k`, validation = fold
`(k + 1) % n_folds`, train = the rest. If there are fewer groups than folds, `n_folds` is
reduced to the number of groups (leave-one-group-out) and the fact is written to the file.
With two folds the non-test fold is divided by follower into training and validation
(at least 20 % of its events for validation); with one fold there is nothing to train on.

## 6. IDM calibration

* IDM: `a = a_max * (1 - (v / v0)^4 - (s_star / s)^2)`, `s_star = s0 + max(0, v*T + v*dv / (2*sqrt(a_max*b)))`, with `dv = v - v_lead`.
* Closed-loop rollout of the follower over the whole event from the observed `s[0], v[0]`, leader given by `x_lead, v_lead` (section 1 integration and clipping).
* Objective (Punzo, Zheng, Montanino 2021): `J = NRMSE(s) + NRMSE(v)`, `NRMSE(y) = sqrt(mean((y_sim - y_obs)^2)) / sqrt(mean(y_obs^2))`. A rollout with a collision gets `J = 10 + fraction of samples with s <= 0`.
* Bounds: `v0 [10, 45]`, `T [0.3, 3]`, `s0 [0.5, 10]`, `a [0.2, 4]`, `b [0.5, 6]`.
* Optimiser: differential evolution (best1bin, population 15 x 5 = 75, dithered mutation U(0.5, 1), recombination 0.7, Latin hypercube initialisation, relative tolerance 0.01), run batched over events in torch; validated against `scipy.optimize.differential_evolution` in the tests.
* Global parameter set per dataset: same optimiser, `J` computed from the errors pooled over all events (each event rolled from its own initial state), plus `10 * collision rate`.
* Outputs in `data/calibration/<dataset>/`: `idm_per_event.parquet`, `idm_global.json`, `idm_spread.json` (mean, std, median, quantiles, coefficient of variation, share of estimates at a bound, correlation matrix).
