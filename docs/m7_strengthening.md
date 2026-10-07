# How to strengthen the study beyond the M7 arms

Written 2026-10-05 at the M6 review, before the M7 runs. The M7 arms (`docs/m7_contract.md`)
close the gaps a reviewer would name first; this note lists what would raise the weight of the
paper further, ordered by value per unit of work. Numbers refer to `runs/_report/report.md` of
2026-10-01.

## 1. Thesis and positioning

The positive method of the plan, the Jacobian penalty, has a precedent: KIDL (Communications in
Transportation Research 2025, arXiv 2504.14241) trains a memoryless network with the same
inequality f_v^2 + 2 f_v f_dv - 2 f_s >= 0 at all equilibria, validated by trajectory RMSE and a
100-vehicle platoon; the audit framework is Zhang, Sun, Zheng, Sun (TR-C 2024). The paper should
therefore not be "a penalty" but "closed-loop fidelity is not trajectory accuracy":

- the certified hybrid has the worst microscopic RMSE of the hybrids (4.89 m against 4.66 of
  the IDM and 4.27 of the free hybrid on the NGSIM test parts) and the best corridor fidelity
  (macro error 0.125 against 0.182 of the IDM, 0.158 of the free hybrid, 0.716 of the LSTM) with
  no collisions; GRU and LSTM beat the IDM on RMSE and collide;
- penalties stabilise memoryless models at no cost and fail for recurrent ones for identifiable
  reasons (needles, out-of-band equilibria, the blind band below 0.05 rad/s, the accuracy price
  of a common equilibrium);
- the certificate survives fine-tuning (25/25), the penalty is a regulariser that does not.

Present the hypotheses of Part A as pre-registered (the plan predates every result) with the
thresholds as written; the "open" and "refuted" verdicts are then evidence of rigour, not
weakness.

## 2. Validity threat to remove: the I-80 corridor is in-sample

The corridor laws are fine-tuned on `ngsim_i80` (D97) with driver folds, and every vehicle draws
one of the five fold members, each of which saw 80 % of the I-80 drivers; the ground truth is
the same recording. The IDM is calibrated on the same events, so the comparison is fair but not
out-of-sample. US-101 (M7 item 5) with the I-80 laws unchanged is the out-of-sample corridor and
should be the headline validation; I-80 becomes the in-sample check. If US-101 behaves
differently, that is the result to report, not to tune away.

## 3. Theory worth one page each

1. The low-frequency expansion |G(i omega)|^2 = 1 - omega^2 M / f_s^2 + O(omega^4) with
   M = f_v^2 + 2 f_v f_dv - 2 f_s: it ties the Jacobian criterion to the rollout penalty's blind
   band and motivates the combined penalty of D110. One derivation, one figure (the gain curves
   of E1 and E2 already show it: the penalised MLP sits on |G| = 1 up to 1 rad/s).
2. The certificate as a theorem with an explicit margin: if the core satisfies M_core >= m at
   every equilibrium of the band and the residual's partial derivatives are bounded by epsilon
   (spectral norms x r_max x the input scale), then M >= m - K epsilon - O(epsilon^2) with K an
   explicit function of the core derivatives; report how tight the bound is against the audited
   margins of the 25 certified runs (the certificate code of M4 computes both).
3. The accuracy price of a common equilibrium: the existence term alone costs GRU/LSTM/PERL
   +2 to +8 % RMSE (e2_existence); relate it to the width of the spacing band of the data (D78:
   5/50/95 % quantiles) as the heterogeneity a memoryless law cannot represent.

## 4. Experiments with a high return

1. **Heterogeneous certified hybrid** (fleet heterogeneity, plan A7): the shared certified
   residual on top of per-event IDM cores calibrated with the stability margin 0.2 (batched
   differential evolution of M1 with the margin constraint; one GPU hour), certificate checked
   per core; 30 corridor runs. It answers "five members are not a fleet" and tests whether the
   certificate scales to a population.
2. **Asymmetry and spectrum metrics** (plan A6 and risk A11) from the existing corridor
   trajectories (no new runs): acceleration/deceleration asymmetry of every law against the
   data, and the oscillation spectrum (PSD of the speed at fixed x). The claim to test: the
   penalty and the certificate do not erase the asymmetry of human driving.
3. **Band sensitivity of the audit**: the shares of unstable/outside equilibria for band
   quantiles 1/99, 5/95, 10/90 (audit only, minutes). Shows that H1.1 does not hinge on D78.
4. **Hybrid with the KIDL/RACER constraints as a named baseline**: the E2 Jacobian arm already
   is the KIDL constraint on an MLP; add the monotonicity terms of RACER (relu(-f_s), relu(f_dv),
   relu(f_v) only, no string term) on fold 0 as a one-row control showing that local terms alone
   leave the string instability (cheap: 3 runs).
5. **Temporal hold-out on I-80**: fine-tune on period 0 drivers, simulate periods 1 and 2 (the
   laws of D97 see all three); a cheap second out-of-sample check if US-101 is contested.

## 5. Statistics

- H12.3 with n = 13 laws is descriptive; with the r_max laws, `idm_heterogeneous_all` and two
  corridors it reaches n = 18 laws x 2 corridors. Report the correlation per corridor and a
  pooled version with the corridor as a stratum; keep Spearman as the decider (D108).
- Report effect sizes with the intervals everywhere (already done) and add the number of runs
  needed to detect a 10 % macro difference at the observed spread (a power statement reviewers
  of TR-C ask for).

## 6. Presentation

- One "stability map" figure per architecture: grid speed x spacing with the band and the
  stable / unstable / outside / none status (the audits store it).
- A table of the penalty loopholes with mechanism, symptom and the fix (needles -> guard;
  out-of-band equilibria -> existence term; |G| < 1 with unstable poles -> local terms; blind
  band -> Jacobian term): a methodological contribution others can reuse.
- Supplementary x-t contours and fundamental diagrams for all laws and periods (the report code
  draws five laws of one period; a loop over the rest is trivial).
- The code and the run manifest on Zenodo with the environment file; the one-command
  reproduction already exists.

## 7. Data

Raw highD when levelXdata grants it (rerun E1-E4, about three GPU days); until then cite
FollowNet as the source of the HighD events and say so in the limitations.
