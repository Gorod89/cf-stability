# M8 contract: strengthening arms

Basis: the user accepted the M7 report on 2026-10-07 and asked to carry out the list of
`docs/m7_strengthening.md`. Items that need computation or code are below; the thesis and the
positioning (section 1 of that note) and the theory derivations (section 3) are written by the
lead in `docs/theory_notes.md`; the work packages provide the numerical checks. Decisions
D115-D124, numbers fixed per item; append to the `## Revision (M7)` table of
`docs/decisions.md` (re-read right before editing; three work packages append concurrently). Outputs
of M8 go to `runs/_tables/m8/<name>.{csv,md}` and `runs/_report/supplement/`; `make_report`
gets a section 11 "Supplement (M8)" that includes every table of `runs/_tables/m8/` and every
figure of `runs/_report/supplement/figures/` (a caption file `<name>.txt` next to each figure).
Nothing of M1-M7 is retrained; `configs/train.yaml` keeps every hash (hash-neutral defaults
for any new key, as in D110). The long queues are started by the lead.

| work package | items | files |
|---|---|---|
| T (theory checks, figures) | 1, 2, 3, 8 | `cf_stability/eval/figures_supplement.py` (new), `scripts/make_figures_supplement.py` (new), `scripts/analysis/*`, their tests |
| E (training, calibration, audit) | 4, 5, 6, 7 (training side) | `cf_stability/stability/*`, `cf_stability/train/*`, `cf_stability/data/*` (views only), `configs/train.yaml`, `configs/calibrate_idm.yaml`, `configs/data/*`, `configs/queue/*`, `scripts/{train,calibrate_idm,certificate,audit_stability,run_experiment}.py`, `scripts/analysis/band_sensitivity.py`, their tests |
| C2 (corridor, statistics, report, release) | 5 and 7 (corridor side), 9, 10, 11, 12 | `cf_stability/corridor/*`, `cf_stability/eval/{collect,stats,tables,corridor_tables,report}.py`, `scripts/{corridor_*,make_tables,make_report,build_corridor,run_corridor}.py`, `configs/corridor/*`, `configs/corridor_metrics.yaml`, `configs/make_report.yaml`, `README.md`, `requirements.txt`, their tests |

## 1. Low-frequency expansion check (work package T, D115)

Claim of the theory notes: for the memoryless law a = f(s, dv, v) linearised at an equilibrium,
|G(i omega)|^2 = 1 - omega^2 M / f_s^2 + O(omega^4) with M = f_v^2 + 2 f_v f_dv - 2 f_s. Check it
on the audits: for every fold-0 run of E1 and of the chosen E2 weight (mlp, pidl, residual_idm,
gru, lstm, perl, idm) and every speed with an equilibrium, compare the numerical gain of the
audit at the three lowest frequencies (0.02, 0.0245, 0.03 rad/s) with the expansion computed
from the stored partial derivatives (memoryless view for the recurrent models). Table
`lowfreq_expansion`: per architecture and arm, the number of speeds, the median and 90 %
quantile of |gain_numerical - gain_expansion| at each of the three frequencies, and the share
of speeds whose sign of (|G| - 1) agrees between the two. Figure `lowfreq_expansion`: one panel
per architecture, numerical gain against the expansion at 0.02 rad/s, the diagonal, points
coloured by arm. If the audits do not store the partials or the gain per frequency and speed,
say which field is missing (the stability.json `audit` block holds the per-speed records).

## 2. Certificate tightness (work package T, D116)

The certificate of E4 bounds the string-stability margin of the hybrid from below by the core
margin minus the residual budget. Table `certificate_tightness`: for every run of e4_stable,
e4_stable_ft and the r_max variants (e4_stable_r{0.1,0.2,0.5}, e4_stable_ft_r{...}; 200 runs),
per grid speed, the certified lower bound of the margin against the audited margin M of the
hybrid (analytic criterion of the audit): the median and the 10 % quantile of the slack
(audited minus certified), the share of speeds where the slack is negative (would be a
violation of the bound: must be 0 and is reported), per r_max and stage (before / after
fine-tuning). Figure `certificate_tightness`: audited margin against certified bound, one
panel per r_max, the diagonal. Read the certificate files (`certificate.json` of the runs) and
the audits; if a quantity is not stored, say which, and compute it from the saved model with
the existing certificate code rather than approximating.

## 3. The price of a common equilibrium (work package T, D123 part a)

Figure `band_width`: the spacing band of the data (D78: 5/50/95 % quantiles of the near-steady
samples per grid speed) for follownet_highd as the relative width (s_high - s_low) / s_median
against the speed, with the per-driver dispersion of the equilibrium spacing at each speed
(the standard deviation over drivers of the driver's median near-steady spacing, drivers with
at least 20 near-steady samples at that speed) as a second curve; the same for ngsim_i80 in a
second panel. Table `band_width` with the numbers per speed. The theory notes relate these to
the existence-term cost of `e2_existence`.

## 4. RACER-type control: monotonicity terms only (work package E, D117)

Penalty kind `monotone` = relu(-f_s) + relu(f_dv) + relu(f_v) at the anchored equilibria of E2
(no string term, no margin), the rational constraints of RACER (Li, Halatsis and Stern, arXiv
2312.07003) in our convention dv = v - v_lead. Queue `e2_monotone.yaml`: mlp, pidl,
residual_idm; fold 0; seed 0; weight 1; steps train, audit, platoon; experiment `e2_monotone_w1`.
Expected (to be measured): the local terms alone leave the string instability of E1. The table
(work package C2, `e2_monotone` in `runs/_tables/m8/`): RMSE and its change against E1 fold 0, band
shares, unstable among equilibria, max gain, next to the E1 and the E2 (Jacobian) rows of the
same fold.

## 5. Heterogeneous certified hybrid (work packages E and C2, D118)

Per-event IDM cores of `ngsim_i80` calibrated with the stability margin 0.2 (the per-event
calibration of M1 with the margin term of D72 added to its objective: `calibrate_idm.py` gets
`per_event_margin` or the work package's equivalent; batched differential evolution as before; file
`data/calibration/ngsim_i80/idm_event_margin0.2_<hash>.json` with the settings and, per event,
the parameters, the objective, the smallest margin on the speed grid and whether the margin
holds). Certificate per core: with the shared certified residual of a fold member of
`e4_stable_ft` (r_max 0.3; the same residual budget), the a priori certificate is checked for
every core (`scripts/certificate.py` extended or a new `scripts/certify_cores.py`) and the
file records the share of cores certified; cores that fail are excluded from the law and the
share is reported. Law `residual_idm_certified_het` (work package C2, `corridor/laws.py` and
`configs/corridor/laws.yaml`): one certified core per vehicle drawn from the certified cores
(as `idm_heterogeneous` draws its parameter sets) and the residual of a fold member drawn as
for `residual_idm_certified`; 30 runs on I-80 and 30 on US-101 (lead). Rows in the
laws, components, instability (exact gain of the cores, as for `idm_heterogeneous`) and H12
tables.

## 6. Band sensitivity of the audit (work package E, D119)

The band of D78 uses the 5/50/95 % quantiles. Recompute the band shares (stable / unstable /
outside / none, unstable among equilibria) of every E1 run of the learned models (mlp, pidl,
residual_idm, gru, lstm, perl; 150 runs) for the quantile pairs 1/99, 5/95 (the reference) and
10/90 offline from the stored audits when the audit stores the equilibrium spacing per speed
and the band only reclassifies (verify; the anchored search of D79 may make the "outside"
status depend on the band through the search region: if so, say so and rerun the audit for the
two new bands on the GPU, 300 audits, from the lead with a queue file
`audit_bands.yaml`). Output `runs/_tables/m8/band_sensitivity.{csv,md}`: per architecture and
band, the mean shares with bootstrap intervals over runs, and the H1.1 statistic (unstable
among equilibria) with its interval; `scripts/analysis/band_sensitivity.py`.

## 7. Temporal hold-out on I-80 (work packages E and C2, D120; optional, last)

Data view `ngsim_i80_p0`: the events of period 0 only (follower ids `ngsim_i80/i80/p0_*`; a
filter key for the id prefix in the view mechanism of D87). IDM global calibration on the view
(`calibrate_idm.py` on the view if it accepts views, else a note), fine-tuning queue
`m8_temporal.yaml`: residual_idm certified from e4_stable (margin core, certificate enforced,
r_max 0.3), mlp and gru from e1, on the view, five folds, seed 0 (15 runs; experiments
`m8_temporal_ft`, `m8_temporal_stable_ft`). Laws `idm_global_p0`, `residual_idm_certified_p0`,
`mlp_p0`, `gru_p0` (work package C2); 4 laws x periods 1 and 2 x 10 seeds = 80 runs on I-80 (main
session). Table `temporal` (work package C2): the macro error of the four laws on periods 1-2 against
the same laws of D97 (fine-tuned on all periods) on the same periods.

## 8. Stability maps and supplementary figures (work package T, D123 part b)

Figure `stability_map_<architecture>` for mlp, pidl, residual_idm, gru, lstm, perl (fold 0,
E1 and the chosen E2 weight): grid speed on the x axis, spacing on the y axis, the band of
the data as a shaded strip, the equilibrium of every speed as a point coloured by status
(stable / unstable / outside / none), E1 and E2 as two panels. Figures `contours_<corridor>_p<k>`
and `fd_<corridor>_p<k>` for both corridors and all three periods with every law of the
corridor (the five of the main report plus the others, 4 x 5 panels at most, seed 0), using
the figure code of M6 (`figures.py` functions) through the new module; captions in the `.txt`
files; PNG 200 dpi and PDF.

## 9. Asymmetry and spectrum (work package C2, D121)

From the existing corridor trajectories (`trajectories.npz` of every run and the ground
truth of the scenarios), per run: the acceleration asymmetry index = mean of a over the
samples with a > 0.1 m/s^2 divided by the mean of |a| over the samples with a < -0.1 m/s^2,
the shares of time spent accelerating and decelerating, and the oscillation spectrum = the
power spectral density of the speed series of the virtual detectors (Welch, 2 s samples,
window 128 s) averaged over the detectors inside the analysis window, summarised by the
frequency of its peak and the spectral centroid between 0.002 and 0.05 Hz. Module
`cf_stability/corridor/asymmetry.py`, script `scripts/corridor_asymmetry.py` writing
`asymmetry.json` next to `macro.json` (and for the ground truth), table `asymmetry` in
`runs/_tables/m8/`: per corridor and law the means with bootstrap intervals and the ground
truth, and the signed relative errors of the asymmetry index and the peak frequency. Report
whether the penalties and the certificate change the asymmetry against the free laws.

## 10. Pooled correlation and power (work package C2, D122)

Table `correlation_pooled`: H12.3 over all laws of both corridors with every law that has runs
(the 13 of D97, `idm_heterogeneous_all`, the four r_max laws, `residual_idm_certified_het`
when it exists): per corridor and pooled with the corridor as a stratum (rank within corridor,
then pooled Spearman; permutation p-value permuting within corridor), with the intervals.
Table `power`: for the macro error, the throughput and the travel time of `idm_global` and
`residual_idm_certified`, the standard deviation over seeds within scenario and the number of
seeds per scenario needed to detect a 10 % difference between two laws at alpha 0.05 and power
0.8 (paired by seed, from the observed spread); the sentence for the paper.

## 11. Report integration (work package C2)

`make_report` section 11 "Supplement (M8)": every table of `runs/_tables/m8/` (CSV and LaTeX
exported to `runs/_report/tables/`), every figure of `runs/_report/supplement/figures/` with its
caption; the loophole table of the theory notes is a document, not a generated table. The
verdict tables gain nothing; the H12 tables include the new laws when present.

## 12. Release files (work package C2, D124)

`requirements.txt` with the exact versions of the packages the pipeline imports (from the
shared environment, no pip call), a `README.md` section "Reproduce" listing the commands of
every milestone in order (from section 10 of `runs/_report/report.md`) and the data access
steps, the manifest already in `runs/_report/manifest.json`. No licence file (the user's
choice).

## 13. Order

1. Work packages T, E, C2 in parallel; smoke tests only (fold 0, few epochs, one short corridor run).
2. Lead: `e2_monotone` (3 runs), the per-event margin calibration, `certify_cores`,
   `m8_temporal` (15 runs); corridor: `residual_idm_certified_het` (60 runs), temporal (80);
   `corridor_asymmetry.py` for all runs; audits of item 6 if needed.
3. Tables, supplement figures, `make_report`, `docs/theory_notes.md` (lead),
   `docs/m8_report.md`, commit, stop for review.
