# Study plan (fixed in advance) and its deviations

This file records what was fixed before any model was trained, so that the confirmatory part of the
study can be told apart from the analyses added later. It is an English summary of Part A of the
internal study plan of 26 September 2026 (the resources of that plan were checked on that date; the
first entry of the decision log, D1, is dated 27 September 2026, the day the data pipeline started).
The plan was not registered with a public registry; "pre-specified" in the paper means this file and
the dated decision log, not a registration.

## Aim

Show that (1) a large share of neural car-following models trained by the standard protocol are string
unstable at their linearised equilibria despite a low trajectory error; (2) a differentiable stability
penalty removes the instability at a small loss of accuracy; (3) in corridor simulation a stable model is
not inferior to the calibrated IDM on macroscopic metrics while a pure network is.

## Hypotheses and decision rules (verbatim thresholds)

| hypothesis | statement | confirmed when | refuted when |
|---|---|---|---|
| H1.1 | networks trained without constraint have >= 50 % string-unstable equilibria for v_e in 5-30 m/s at an RMSE below the IDM's | for at least two of MLP, GRU, LSTM, with the 95 % interval not crossing 30 % | share < 20 % for most architectures |
| H1.2 | the stability penalty lowers the share of unstable equilibria below 10 % at an increase of the closed-loop spacing RMSE of at most 10 % | both parts hold with their intervals | RMSE loss > 20 % or share > 25 % |
| H1.3 | the penalty improves the reproduction of the concave growth of the oscillation amplitude by >= 30 % | relative reduction of the growth-curve error with an interval excluding 0 | improvement < 10 % |
| H1.5 | the residual hybrid with a Jacobian bound on its residual keeps its certificate after fine-tuning | share of unstable equilibria after fine-tuning < 10 %, that of the free network > 30 % | no difference |
| H12.1 | a pure network degrades >= 2 of 3 corridor macro metrics by >= 15 % relative to the IDM | on both corridors | no metric degraded |
| H12.2 | the stable hybrid is not inferior to the IDM in macro metrics (TOST, +-10 %) and superior in micro metrics | equivalence in macro and superiority in micro | macro worse by > 10 % |
| H12.3 | the macro error correlates with the share of unstable equilibria, r > 0.6 over >= 10 models | r > 0.6, p < 0.05 | r < 0.3 |

Notes on the rules as applied: the statistic of H1.1 is the share of string-unstable equilibria among
the equilibria found (D92, 2026-09-29) by the numerical rule of the audit; "r > 0.6, p < 0.05" is a
point-estimate rule with a permutation p-value for the null of no association, so a confirmed H12.3 does
not establish that the lower confidence limit exceeds 0.6; the triple of H12.1 and the TOST metrics of
H12.2 were fixed in D108 (2026-10-05) before the US-101 runs. The plan numbers its single-vehicle hypotheses H1.1, H1.2, H1.3 and H1.5 (it has no H1.4); the
data-volume variants of the models (10 / 30 / 100 % of the training data) listed in its experimental
matrix were not run.

## Design fixed in advance

- Data: highD events (the FollowNet events stand in for the raw recordings, D22), NGSIM I-80 and US-101
  with a physics-informed reconstruction, Waymo car-following pairs, OpenACC as the control; FollowNet
  extraction criteria; five-fold splits by follower identifier and by site; IDM calibration following
  Punzo et al.; persistence baseline.
- Models: IDM, k-NN, MLP, GRU, LSTM, PIDL-CF, PERL-style Newell + LSTM residual, residual hybrid IDM +
  MLP with a bounded residual; 5 seeds x 5 folds per configuration.
- Stability: analytic criterion at the grid of equilibrium speeds, numerical frequency response, platoon
  growth test; Jacobian penalty for memoryless and residual models, rollout gain penalty for recurrent
  models; certificate for the residual hybrid.
- Experiments E1-E5 and the I-80 corridor (ten CF laws x ten seeds; I-24 as an extension if available).
- Statistics: bootstrap over drivers (1 000 resamples), paired Wilcoxon tests with Holm's correction,
  TOST with +-10 % for H12.2, Spearman and Pearson with bootstrap intervals for H12.3.

## Confirmatory and exploratory parts

Confirmatory (fixed in the plan): E1-E5 and the I-80 corridor with H1.1, H1.2, H1.3, H1.5 and
H12.1-H12.3. Added after the first results, and labelled as such in the paper: the anchoring of the
equilibria to the spacing band (D78-D80, after the M3 review), the collision-free prefix of the platoon
test (D109), the second corridor US-101 (D112), the residual-amplitude sweep (D111), the sensitivity
variants (D113), the temporal hold-out (D120), the combined and monotone penalty arms (D110, D117), the
asymmetry and spectrum analysis (D121), the pooled and clustered correlations (D122 and the revision of
7 October 2026), the factorial ablation of the certified hybrid and the long-window arm of the rollout
penalty (both of the revision of 7 October 2026). Every
deviation from the plan is an entry of `docs/decisions.md` with its date.
