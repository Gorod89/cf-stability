# Milestone M6 report: report generation

Date: 2026-10-01. Contract: `docs/m6_contract.md`. Decisions: `docs/decisions.md`, D105-D107.
Nothing is committed to git. The generated report with every table and figure is
`runs/_report/report.md` (`python scripts/make_report.py`); this document records what was built,
what the figures add to the tables of M4 and M5, the corrected verdict of H1.3 and the
deviations.

## 1. Commands run

```bash
python scripts/make_report.py refresh=true    # reruns make_tables.py and corridor_metrics.py tables=true, then runs/_report/
python scripts/make_report.py                 # from the existing tables of runs/_tables/m4 and m5 (about 40 s)
./.venv/Scripts/python.exe -m pytest          # 511 tests: 491 passed, 20 skipped
```

The second command is the one that reproduces every table and figure; the first one also
rebuilds the inputs. Nothing was retrained or resimulated in M6: the only change of numbers
comes from D107 (platoon files reread, see section 4).

## 2. What was built

- `scripts/make_report.py` (hydra, `configs/make_report.yaml`), `cf_stability/eval/report.py`
  (tables to CSV and LaTeX, the generated `report.md`, the manifest), `cf_stability/eval/figures.py`
  (the seven figures of the contract), `tests/test_make_report.py` (12 tests).
- `runs/_report/`: `report.md` (10 sections, 767 lines), `tables/` (16 tables, CSV with full
  precision and LaTeX with booktabs and rounded numbers), `figures/` (7 figures, PNG 200 dpi and
  PDF), `manifest.json` (682 run directories in 35 experiments, all with `metrics.json`; 390
  corridor runs with `run.json` and `macro.json`; config hashes of the inputs; versions; date).
- New tables (D105): `m5_components` (the eight signed components per law, anchored and dynamic
  ones marked), `macro_error_dynamic` (mean of the absolute dynamic components) and
  `macro_error_dynamic_correlation`; `e2_existence` (the existence arm of D93) is also exported.
- D107 in `cf_stability/eval/collect.py`, `stats.py`, `tables.py`: the growth error and the
  standard-deviation ratio of a platoon that collides are undefined; the existing
  `platoon.json` files are reread with this rule, no rerun.

The script calls the table code of M4 and M5 and recomputes nothing; a missing input gives an
empty cell or a note in the last section of the report (none at present).

## 3. What the figures add to the tables

1. `rmse_vs_instability` (E1). Two groups: the learned laws at RMSE 2.1-2.9 m with 0.84-0.95 of
   their equilibria unstable, and the mechanistic laws at 3.1-3.4 m with 0.00-0.15. The hybrid
   lies between them (2.70 m, 0.31). The band panel shows where the recurrent laws hide their
   instability: GRU, LSTM and PERL have no equilibrium inside the spacing band of the data at
   30-39 % of the speeds in support ("outside") and no equilibrium at all at 2-12 %.
2. `e2_tradeoff` (E2 sweep). The MLP reaches the stable share 1.00 inside the band at weights
   0.1-10 and the PIDL at 1-10 without leaving their RMSE (2.78-2.94 m); the hybrid reaches 0.99
   at weight 0.1 (2.74 m) and falls back to 0.87 at weights 1-10, where the penalty costs it
   0.65-0.77 m of RMSE. The recurrent laws do not: the rollout gain penalty at weights 1-10
   costs GRU and LSTM +24-39 % RMSE and leaves them at 0.00-0.10 stable, the chosen weight 0.1 is
   the least bad point (0.24 and 0.55 stable), and PERL never leaves 0.00.
3. `gain_curves` (fold 0). The penalised MLP and PIDL sit on the margin |G| = 1.0 up to about
   1 rad/s and only then fall below it: the Jacobian penalty makes the response marginal, not
   damped, while the IDM and the hybrid decay below 0.5 at 2 rad/s. The unpenalised GRU peaks at
   |G| = 1.3-2.3 between 0.05 and 0.5 rad/s; the penalised one is damped to 0.7 between 0.1 and
   0.3 rad/s but returns to 1.0 below 0.05 rad/s (the blind band of the rollout penalty). The
   unpenalised LSTM has three speeds with |G| = 10-100 (needle equilibria); the penalised one is
   close to 1 everywhere. The penalised PERL is worse than the free one: |G| = 3-30 below
   0.1 rad/s at most speeds.
4. `growth_curves` (E5). The OpenACC platoons have 3-10 vehicles; the IDM is the only law whose
   platoon reaches position 50 on every profile. On the ZalaZone ACC profiles the ACC-calibrated
   IDM amplifies the speed standard deviation from 2.2 to 7 m/s over 50 positions where the data
   reach 2.3-3.4 m/s at position 8-10; the penalised MLP grows least (3.6 m/s at position 50)
   and does not collide; MLP, GRU and LSTM collide at positions 5-25 on every ACC profile and
   at the first position on the Vicolungo profiles. Behind the braking pulse the IDM damps
   (1.0 to 0.2 m/s), the learned laws amplify to 1.4-1.9 m/s before they collide, the penalised
   MLP amplifies to 1.9 m/s at positions 30-40 and then decays without a collision.
5. `corridor_speed_contours` (i80_p1, seed 0). The data carry backward waves over the whole
   500 m at 2-8 m/s. `idm_global` keeps the first 100 m in free flow (12 m/s) for the whole
   period and is congested downstream with few, weak waves; the certified hybrid has the same
   free entry but carries sharp backward waves downstream like the data. The MLP has the
   inverted profile: 0-3 m/s in the first 200 m and free flow above 250 m between 300 and 700 s,
   with two strong waves near 150 and 800 s. The GRU is congested like the data at somewhat lower
   speeds with backward waves; the LSTM is in gridlock (0-2 m/s everywhere) from 150 s on.
6. `fundamental_diagrams` (i80_p1). The data occupy densities 250-600 veh/km (all lanes) at
   4000-9000 veh/h. `idm_global` reproduces this band and adds a free branch at 100-250 veh/km;
   the certified hybrid scatters below the data at mid densities (FD scatter +53 % in the TOST
   table); the MLP is shifted to 150-250 veh/km and 6000-7000 veh/h (its free downstream half);
   the GRU follows the band; the LSTM lies at 1000-4000 veh/h over 100-900 veh/km.
7. `macro_error_vs_instability`. The laws without collisions (IDM, heterogeneous IDM, free and
   certified hybrid) sit at instability 0.00-0.09 and macro error 0.13-0.30; the learned laws at
   0.72-0.96 and 0.19-0.73; the penalised MLP at 0.00 instability keeps a macro error of 0.27
   because it still collides (1588 episodes per 1000 veh-km), which is the point that keeps the
   Pearson correlation at 0.47.

## 4. Corrected verdict of H1.3 (D107)

The platoon test keeps simulating after a collision, and the speed standard deviation behind
it reaches 100-500 m/s. The M4 report judged H1.3 "confirmed on the human profiles" on such
numbers (growth errors of 6.5 and 15.6 in two folds of the unpenalised GRU and LSTM against 0.1
elsewhere). With D107 (growth error undefined after a collision, pairs with both values only,
at least three pairs for a verdict) every row of H1.3 is **open**: the OpenACC-trained laws
collide on 7-14 of their 10-15 profiles with and without penalty, so no comparison has three
pairs. What remains is descriptive: the penalty removes the platoon collisions of the MLP on
the ACC profiles (7 of 10 without, 0 of 10 with; growth error -31 % [-95 %, +1 %] over the two
valid pairs) and changes nothing for GRU (8 to 10 of 10) and LSTM (8 to 9 of 10). The addendum is
in `docs/m4_report.md`, section 8; the growth-error columns of the E2 table changed in the same
way (pairs 25 for MLP, PIDL and ResidualIDM, 4 for GRU, 15 for LSTM, 0 for PERL).

## 5. Final verdicts (`runs/_report/tables/verdicts.csv`)

| hypothesis | verdict | basis |
|---|---|---|
| H1.1 | holds (MLP, GRU, LSTM) | 0.94, 0.86, 0.92 of the equilibria unstable; RMSE -7.7, -32.6, -29.0 % against the IDM |
| H1.2 | confirmed for MLP, PIDL, ResidualIDM | not stable inside the band 0.00; RMSE -0.5, -0.3, +1.5 % |
| H1.2 | refuted for GRU, LSTM, PERL | not stable 0.66, 0.39, 1.00; RMSE +14.7, +25.0, +26.9 % |
| H1.3 | open (every view and model) | fewer than three valid pairs after D107 |
| H1.5 | holds | certified hybrid unstable 0.00 in 25/25 runs after fine-tuning; fine-tuned MLP 0.98 |

Corridor (M5, unchanged): macro error certified hybrid 0.125 < free hybrid 0.158 < IDM 0.182 <
GRU 0.192 < ... < LSTM 0.716; dynamic macro error certified hybrid 0.190 best; Spearman of the
macro error with the unstable share 0.66 [0.24, 0.85] (dynamic 0.51 [0.01, 0.80]); TOST
certified hybrid against IDM: equivalent within 10 % in throughput, speed, queue discharge,
peak flow and travel times, not in the wave pattern.

## 6. Deviations from the specification

New in M6: D105 (components and dynamic macro error, heterogeneous IDM unchanged, no safety
layer, headline wave numbers), D106 (the report script reruns the table scripts and recomputes
nothing), D107 (growth error undefined after a collision; H1.3 open below three pairs). Section
9 of `runs/_report/report.md` lists all 107 decisions by milestone, one line each.

## 7. Tests

511 tests: 491 passed, 20 skipped (GPU variants), 0 failed (`pytest -q` on top of the `-q` of
`pyproject.toml` hides the summary line; the counts are those of the progress characters). New
in M6: `tests/test_make_report.py` (12 tests: every table in
both formats, every figure in both formats, every section of the report, a missing input gives
a note and no exception, LaTeX rows balanced).

## 8. Open points for the review

1. The pipeline is complete: six milestones, every table and figure from one command. Nothing
   is committed.
2. Not done, offered at the M5 review as optional: the heterogeneous IDM drawn from all
   estimates (30 corridor runs).
3. The penalised PERL has larger low-frequency gains than the free one (figure 3); the chosen
   weight 0.1 is the least bad RMSE, not a stabilising one. The verdict "refuted" stands; the
   figure makes the mechanism visible.
4. Cosmetic: the "LSTM" label of figure 1 touches the left axis; the LaTeX tables need
   `\usepackage{booktabs}`.
