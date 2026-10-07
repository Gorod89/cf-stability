# Milestone M7 report: revision arms

Date: 2026-10-06. Contract: `docs/m7_contract.md`. Decisions: `docs/decisions.md`, D108-D114 and
E10. Git: `0de1a75` (M1-M6), `47ceac0` (contract), `6861ea2` (M7 code), and the commit of this
report with every run output of M7. Tables: `runs/_tables/m4`, `runs/_tables/m5`; the generated
report `runs/_report/report.md` (22 tables, 7 figures, 0 notes on missing inputs). The machine
was shared with the sibling sessions' GPU jobs during the runs (E10): timings are not
representative, no result depends on them.

## 1. Commands run

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

Three failed US-101 runs (SUMO `bad allocation`, CUDA out of memory under the shared load, E10)
were rerun by the same command; all 1 050 corridor runs and 218 training runs are complete.

## 2. Verdicts of H12.1-H12.3 (D108)

| hypothesis | verdict | basis |
|---|---|---|
| H12.1 | open (I-80: LSTM confirmed, MLP and GRU open; US-101: MLP, GRU, LSTM confirmed) | degradation against `idm_global` on the triple throughput / travel time / wave speed: on US-101 every network loses >= 15 points on >= 2 of 3 (MLP +0.23/+0.26/+0.56, GRU +0.21/+0.47/-0.13, LSTM +0.74/+1.39/+0.16); on I-80 the MLP misses the threshold by 0.3 points on throughput (+0.147 [+0.135, +0.157]) and the GRU degrades all three by 0.03-0.13 |
| H12.2 | open (both parts fail) | macro: equivalent within 10 % in throughput and travel time on both corridors, not in wave speed (I-80 +30.5 % [+23.1, +38.2] faster, US-101 -7.1 % [-25.0, +13.9]); micro: certified hybrid +4.8 % [+3.7, +5.9] RMSE against the NGSIM-calibrated IDM over 5 521 drivers (free hybrid -8.8 %, fine-tuned MLP +21.2 %) |
| H12.3 | confirmed on both corridors | Spearman of the macro error with the share of unstable equilibria: I-80 0.634 [0.197, 0.851], permutation p 0.017, 14 laws; US-101 0.735 [0.356, 0.906], p 0.006, 13 laws; on US-101 Pearson (0.661, p 0.015) and the dynamic macro error (0.724, p 0.008) pass as well |

The in-sample caveat of the M6 review is now measurable: the laws were fine-tuned on I-80 and
simulated on I-80 (every fold member saw 80 % of its drivers); on US-101 with the same laws
unchanged, H12.1 holds for every network and the learned laws lose 0.12-0.23 of macro error
against their I-80 values (GRU 0.192 -> 0.421, MLP 0.308 -> 0.428) while the IDM laws and the
hybrids keep theirs (IDM 0.182 -> 0.133, certified hybrid 0.125 -> 0.113).

## 3. Second corridor: US-101 (D112)

Scenarios `us101_p0..p2` from the own reconstruction (640 m, lanes 1-5, auxiliary lane 6 at
184-412 m between the on-ramp at 148-169 m and the off-ramp at 430-469 m; 2 169 / 2 017 /
1 915 vehicles per period, 127-141 from the on-ramp, 72-79 to the off-ramp; ground truth
8 308 / 7 592 / 7 274 veh/h, 11.3 / 8.8 / 7.7 m/s, wave speed -6.0 / -5.5 / -5.4 m/s). "Everything
else as I-80" did not hold: with the I-80 settings even the IDM deadlocked in the weaving
section (through vehicles drift into the faster auxiliary lane, LC2013 refuses the gaps the
data show, an on-ramp and an off-ramp vehicle block each other at the lane end). Three rules
from the data fix it (D112): the auxiliary lane admits only its users in the data (643 vehicles,
18 of 5 476 through vehicles), those get `lcAssertive` 2.5 (LC2013 then admits 89 % of the 434
observed merges instead of 31 %), and an off-ramp vehicle sees its lane end 30 m early.

| law | macro error I-80 | macro error US-101 | dynamic US-101 | collisions / 1000 veh-km US-101 | inserted US-101 |
|---|---:|---:|---:|---:|---:|
| residual_idm_certified | 0.125 | **0.113** [0.107, 0.119] | 0.122 | 0.23 | 1.000 |
| idm_heterogeneous | 0.302 | 0.125 | 0.196 | 0.15 | 0.999 |
| idm_global | 0.182 | 0.133 | 0.229 | 0.03 | 1.000 |
| residual_idm | 0.158 | 0.156 | 0.256 | 0.03 | 1.000 |
| gru_penalty | 0.235 | 0.347 | 0.458 | 753 | 0.950 |
| mlp_penalty | 0.273 | 0.380 | 0.518 | 1 161 | 0.938 |
| pidl | 0.274 | 0.412 | 0.526 | 1 294 | 0.903 |
| gru | 0.192 | 0.421 | 0.517 | 211 | 0.909 |
| mlp | 0.308 | 0.428 | 0.651 | 726 | 0.896 |
| knn | 0.490 | 0.628 | 0.583 | 1 956 | 0.567 |
| perl | 0.431 | 0.661 | 0.685 | 13 481 | 0.891 |
| lstm_penalty | 0.734 | 0.886 | 0.786 | 5 540 | 0.308 |
| lstm | 0.716 | 0.888 | 0.837 | 2 858 | 0.382 |

The certified hybrid is the best law on both corridors; the heterogeneous IDM, worst of the
collision-free laws on I-80, is second on US-101 (its capacity shortfall of D63 matters less
in the weaving section). The learned laws collide on every run (211-13 481 episodes per 1 000
veh-km) and insert only 31-95 % of the demand; the three IDM-based laws show isolated contact
episodes on US-101 (1-7 of 30 runs, 0.03-0.23 per 1 000 veh-km) in the weaving section. TOST
on US-101: equivalent within 10 % in throughput (+0.0 %) and travel time (-6.5 %), not in wave
speed.

## 4. Sensitivity to the boundary and to the lane-change model (D113)

Period 1 of I-80, seeds 0-4, three laws, five variants. The anchored components (throughput,
travel time) move little for every law; the wave components of the certified hybrid do.

| variant | idm_global | residual_idm_certified | gru | first |
|---|---:|---:|---:|---|
| baseline (gain 0.02, LC2013 default) | 0.177 [0.169, 0.184] | **0.131** [0.098, 0.164] | 0.170 | certified |
| gain 0.01 | 0.165 | 0.230 [0.198, 0.276] | 0.183 | IDM |
| gain 0.04 | 0.161 | 0.197 [0.164, 0.224] | 0.222 | IDM |
| no feedback (speed-only boundary) | 0.273 | 0.369 | 0.202 | GRU (the queue drains, D95) |
| lc_low (0.5 / 0.5 / 0.5) | 0.184 | 0.227 [0.178, 0.273] | 0.252 | IDM |
| lc_high (2.0 / 1.0 / 1.5) | 0.156 | 0.201 [0.164, 0.247] | 0.156 | GRU / IDM |

The ranking among the three collision-free laws is not robust: the certified hybrid leads on
period 1 only under the baseline boundary and lane-change settings; under the four plausible
alternatives the IDM is first and the hybrid's error rises to 0.20-0.23 (intervals disjoint
from the baseline). The components show why: the hybrid's number of waves (-0.27 at the
baseline, -0.47 to -0.60 under the variants) and wave amplitude (+0.36, then +0.55 to +0.70)
are decided by the boundary feedback and by how LC2013 feeds the waves; the IDM damps the
waves under every setting (-0.6 to -0.67 waves, amplitude +0.09 to +0.30) and is insensitive.
Consequence for the paper: the claim "the certified hybrid is the best law" holds over the
full design (3 periods x 10 seeds on two corridors) but its margin over the IDM is within the
boundary-design sensitivity; the robust claims are the separation between the collision-free
laws (0.11-0.34) and the collapsing learned laws (0.35-0.89 with collisions), and H12.3.

## 5. Residual-amplitude sweep of the certified hybrid (D111)

| r_max | core | certificate HighD / after fine-tuning | unstable equilibria NGSIM | RMSE HighD (m) | RMSE NGSIM (m) | macro error I-80 | macro error US-101 | collisions / 1000 veh-km I-80, US-101 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0.1 | certified | 25/25, 25/25 | 0.00 | 4.54 | 5.38 | 0.197 | 0.177 | 0, 0.23 |
| 0.2 | certified | 25/25, 25/25 | 0.00 | 4.51 | 5.26 | 0.181 | 0.147 | 0, 0.06 |
| 0.3 | certified | 25/25, 25/25 | 0.00 | 4.51 | 5.16 | 0.125 | 0.113 | 0, 0.23 |
| 0.5 | certified | 25/25, 25/25 | 0.00 | 4.54 | 5.07 | **0.108** | **0.104** | 0.11, 0.16 |
| 0.3 | free | none | 0.07 | 2.81 | 5.18 | 0.209 | 0.194 | 0, 0.03 |
| 1.0 | free (E4) | none | 0.04 | 2.70 | 4.49 | 0.158 | 0.156 | 0, 0.03 |

Three facts. The certificate holds a priori and after fine-tuning at every amplitude. The
accuracy cost on HighD (4.5 m against 2.7-2.8 m) comes from the margin-0.2 core and the
Lipschitz budget, not from the amplitude: the free core at r_max 0.3 keeps 2.81 m. And the
corridor fidelity improves monotonically with the certified amplitude on both corridors
(0.197 -> 0.108 on I-80, 0.177 -> 0.104 on US-101), while the free core at the same amplitude
0.3 is worse (0.209 / 0.194) than the certified one (0.125 / 0.113): the structure, not the
small residual, buys the corridor fidelity. r_max 0.5 is a better operating point than the
0.3 of D86 at the price of isolated contact episodes (0.11-0.16 per 1 000 veh-km).

## 6. Low-frequency penalty arm (D110)

Penalty kind `combined` = rollout gain penalty of E2 + Jacobian penalty of the memoryless view
(the omega -> 0 limit of |G| <= 1) + needle guard (3 1/s). Pilot on fold 0 (weights 0.1 and 1),
chosen by D85: GRU 0.1, LSTM 0.1, PERL 1; then five folds, seed 0, against E1 seed 0.

| architecture | weight | not stable in band (E2 gain only) | not stable (combined) | RMSE change (E2) | RMSE change (combined) | H1.2 (combined) |
|---|---:|---:|---:|---:|---:|---|
| GRU | 0.1 | 0.66 [0.55, 0.76] | 0.45 [0.25, 0.65] | +14.7 % | +17.3 % [+15.1, +19.3] | open |
| LSTM | 0.1 | 0.39 [0.31, 0.48] | 0.50 [0.24, 0.76] | +25.0 % | +29.0 % [+27.1, +30.6] | refuted |
| PERL | 1 | 1.00 | 0.97 [0.92, 1.00] (no equilibrium at 65 % of the speeds) | +26.9 % | +28.3 % | refuted |

The fold-0 pilot had promised more (LSTM stable at 91 % of the band speeds); over five folds the
combined penalty improves the GRU by 0.2 of band share for +2.6 points of RMSE and does nothing
for LSTM and PERL. The verdict of M4 stands with the obvious fix tried: stability penalties do
not stabilise the recurrent models at an acceptable accuracy cost. The PERL runs with the
combined penalty have no equilibrium at most speeds, so their numerical share is undefined
(4 notes in `runs/_tables/m4/missing.txt`).

## 7. H1.3 on the collision-free prefix (D109)

| view | model | collided profiles without / with penalty | first collided position without / with | growth error without / with | reduction (pairs) | H1.3 |
|---|---|---|---|---|---|---|
| ACC | MLP | 7/10, 0/10 | 36.5, 51 | 0.092, 0.094 | -2.6 % [-29.1, +14.9] (5) | refuted |
| ACC | GRU | 8/10, 10/10 | 14.2, 24.4 | 0.097, 0.096 | -2.2 % (1) | open |
| ACC | LSTM | 8/10, 9/10 | 17.4, 23.7 | 0.152, 0.100 | +52.5 % (1) | open |
| human | MLP | 11/15, 13/15 | 18.0, 13.5 | 0.147, 0.218 | -48.2 % [-123.9, +3.2] (5) | refuted |
| human | GRU | 13/15, 12/15 | 13.1, 13.9 | 0.176, 0.091 | +48.5 % [+11.5, +76.5] (5) | confirmed |
| human | LSTM | 14/15, 13/15 | 9.9, 13.2 | 0.092, 0.057 | +38.4 % [-61.6, +68.1] (5) | open |

The prefix rule gives H1.3 a verdict where D107 could not: refuted for the MLP on both views
(the penalty removes its platoon collisions on the ACC profiles, 7/10 -> 0/10, but does not
improve the growth before them), confirmed for the GRU on the human profiles, open elsewhere
(one pair, or an interval that includes 0). The IDM never collides (position 51 throughout).
The E2 growth columns use the same rule (reductions MLP -22 %, PIDL -40 %, GRU -20 %, LSTM
-68 %, PERL +95 %, hybrid -11 % over 15-25 pairs; the sign convention there is "change", a
negative value is an improvement of the penalised model).

## 8. Heterogeneous IDM from all estimates (D114)

`idm_heterogeneous_all` (7 651 parameter sets, 4 811 with a parameter on a bound) on I-80:
macro error 0.336 [0.276, 0.406] against 0.302 of the interior-only law and 0.182 of the global
IDM; it inserts only 98 % of the demand (lower capacity, the bound-hitting sets include long
time gaps). The choice of D63 stands.

## 9. Deviations and environment

D108-D114 as in the contract, with these departures: D109 compares the followers only (the
leader is shared) and uses `travel_time_mean` in the H12.2 TOST; D112 needed the three weaving
rules of section 3; D113's expected ranking is reported as found, not enforced. E10: the shared
load (sibling sessions, a Codex job) slowed the training runs 4-7 times and caused three
corridor failures that the restartable runner reran. The macro files carry two config hashes
(one per corridor, the metric settings differ by design); the note in `missing.txt` about it is
expected. The strengthening proposals of `docs/m7_strengthening.md` predate these results; its
section 2 (in-sample I-80) is now addressed by US-101.

## 10. Tests

552 tests after the M7 code: 530 passed, 22 skipped (GPU variants), 0 failed.
