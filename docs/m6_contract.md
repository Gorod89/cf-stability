# M6 contract: report generation

Basis: section 7 of the specification ("make_report.py produces tables (CSV and LaTeX) and
figures: RMSE vs instability scatter, gain curves |G(omega)|, concave-growth curves, speed
contours for corridors, fundamental diagrams, correlation plot"), the quality requirement "one
command reproduces each table", and the four proposals of the M5 review (accepted 2026-10-01):
components of the macro error reported separately with a "dynamic" macro error; the
heterogeneous IDM kept as it is; no safety layer; wave speed by cross-correlation and wave
amplitude as the headline wave numbers.

## 1. Output

`python scripts/make_report.py` (hydra, `configs/make_report.yaml`) writes `runs/_report/`:

```
report.md                       the generated report (sections below), with the tables inline and the figures linked
tables/<name>.csv  tables/<name>.tex   every table of M4 and M5 (and the verdicts), CSV with full precision, LaTeX (booktabs) rounded
figures/<name>.png  figures/<name>.pdf
manifest.json                   run counts per experiment, config hashes of the inputs, software versions, date
```

The script calls the existing table code (`cf_stability/eval/tables.py` of M4,
`cf_stability/eval/corridor_tables.py` of M5) rather than recomputing numbers; it reads the
`runs/_tables/m4/*.csv` and `runs/_tables/m5/*.csv` that `scripts/make_tables.py` and
`scripts/corridor_metrics.py tables=true` write, and reruns those two when `refresh=true`.
Missing inputs give empty cells or a missing figure with a note in `report.md`, never an
exception.

## 2. Tables (CSV and LaTeX)

From M4: `e1`, `e2_sweep`, `e2`, `e3`, `e4`, `e5`, `verdicts`. From M5: `laws`, `laws_raw`
(the raw metrics per law and the ground truth per period), `instability` and
`instability_correlation`, `tost`. New: `m5_components` (the eight components of the macro
error per law, with intervals, the anchored ones (throughput, mean speed, queue discharge,
travel time) and the dynamic ones (FD, wave speed, waves, wave amplitude) marked) and the
dynamic macro error `macro_error_dynamic` (mean of the absolute dynamic components) with its
correlation to the instability (Spearman and Pearson, bootstrap over the laws, all laws and the
laws without collisions). The LaTeX tables carry the units in the column heads and a caption
that names the design (units, numbers of runs, interval type).

## 3. Figures (matplotlib, PNG 200 dpi and PDF)

1. `rmse_vs_instability`: E1, one point per model: test spacing RMSE (x) against the share of
   unstable equilibria among the equilibria found (y), with the 95 % intervals as error bars,
   the IDM marked; a second panel with the band shares stable / unstable / outside / none as
   stacked bars per model.
2. `e2_tradeoff`: per architecture the sweep: test RMSE (x) against the share of speeds that are
   stable inside the band (y) for the four weights (connected, labelled), the point without
   penalty of E1, and the chosen weight marked.
3. `gain_curves`: |G(omega)| of the numerical frequency response over the 25 frequencies, from
   the `stability.json` of the fold-0 runs of E1 and of the chosen E2 weight: one panel per
   architecture (IDM, MLP, PIDL, ResidualIDM, GRU, LSTM, PERL), the curve of every speed in
   support as a thin line, its median as a thick line, without and with penalty in two colours;
   the line at 1.02.
4. `growth_curves`: platoon test, speed standard deviation per platoon position: the empirical
   curves of the five OpenACC profiles and the braking pulse against the model curves of the E5
   laws of the matching driving mode (IDM, MLP, GRU, LSTM without and with penalty; mean over
   the five folds); one panel per profile.
5. `corridor_speed_contours`: x-t speed fields (20 m x 2 s, lanes 1-6 together) of period 1 for
   the ground truth, `idm_global`, `residual_idm_certified`, `mlp`, `gru` and `lstm` (seed 0),
   same colour scale, the analysis window marked.
6. `fundamental_diagrams`: density-flow scatter of the Edie cells (lanes together, 100 m x 30 s)
   of period 1 for the ground truth and the same five laws, with the binned mean flow curve of
   the truth on every panel.
7. `macro_error_vs_instability`: one point per law: share of unstable equilibria of the members
   (x) against the macro error (y) with its interval, laws with collisions marked, the Spearman
   and Pearson values with intervals in the legend; a second panel for the dynamic macro error.

Style: one consistent style for all figures (font size readable at column width, colour-blind
safe palette, no title inside the figure, labels with units); every figure also written as PDF.

## 4. The generated report (`runs/_report/report.md`)

Sections: 1 data and runs (counts, event sets, folds, seeds); 2 verdicts of the hypotheses
(the verdict table); 3 E1 with figure 1; 4 E2 with figures 2 and 3 and the existence arm; 5 E3;
6 E4; 7 E5 with figure 4; 8 corridor with the laws table, the components, the dynamic macro
error, figures 5-7, the correlation and the TOST tables; 9 deviations from the specification
(the list of D-entries by milestone, one line each, taken from `docs/decisions.md`); 10 how to
reproduce (the commands of every milestone in order). Numbers in the text come from the tables
(no hand-typed numbers).

## 5. Tests

`tests/test_make_report.py`: the script runs on a hand-made tree of tables and a few run
directories with the files of the figures (small synthetic data), writes every table in both
formats and every figure in both formats, and `report.md` with every section; a missing input
gives a note and no exception; LaTeX tables are valid (balanced `&` per row, `\\` line ends).
