# Theory notes for the paper

Written 2026-10-07 (M8, `docs/m8_contract.md`). Convention of the whole pipeline: a = f(s, dv, v)
with s the net gap, dv = v - v_lead (positive when closing), v the follower speed. Partial
derivatives at an equilibrium (s_e, 0, v_e) are f_s, f_dv, f_v; the string-stability margin is
M = f_v^2 + 2 f_v f_dv - 2 f_s. The numerical checks of these notes are the tables
`lowfreq_expansion`, `certificate_tightness` and `band_width` of `runs/_tables/m8/` (D115, D116,
D123).

## 1. Thesis and positioning

The paper's claim is not "a stability penalty" (KIDL, Communications in Transportation
Research 2025, trains a memoryless network with the same inequality; RACER adds the rational
sign constraints; Zhang, Sun, Zheng and Sun, TR-C 2024, give the audit framework). The claim is
that **trajectory accuracy does not predict closed-loop fidelity**, with four pieces of
evidence that no single prior work has:

1. an audit of seven learned architectures anchored to the spacing band of the data (84-95 %
   of their equilibria unstable; recurrent models hide instability by having no equilibrium
   in the band at 30-40 % of the speeds);
2. penalties that stabilise memoryless models at no accuracy cost and fail for recurrent ones
   for identifiable reasons, including the obvious fix (section 4);
3. a residual hybrid with an a priori certificate that survives fine-tuning (25/25 at every
   residual amplitude) and whose corridor fidelity improves with the certified amplitude
   while a free core at the same amplitude is worse (M7, section 5);
4. corridor validation on two NGSIM sites, one of them out of sample, in which the learned
   laws that beat the IDM on RMSE collide and collapse (macro error 0.35-0.89) while the
   certified hybrid with a worse RMSE is the best law (0.113-0.125), with the macro error
   correlating with the unstable share across laws (Spearman 0.63 and 0.74).

The hypotheses of Part A were fixed before any result; the paper reports them with their
pre-registered thresholds, including the open and refuted ones.

## 2. The low-frequency expansion of the gain (D115)

Linearising ds/dt = v_lead - v and dv/dt = f(s, dv, v) at an equilibrium and writing the
Laplace variable p:

    p dv = f_s ds + f_dv (dv - dv_lead) + f_v dv,    p ds = dv_lead - dv,

so the speed transfer function from the leader to the follower is

    G(p) = (f_s - p f_dv) / (p^2 - p (f_dv + f_v) + f_s),

which at p = i omega is the expression of the plan (A3). With N = f_s^2 + omega^2 f_dv^2 and
D = f_s^2 + omega^2 (f_dv^2 + M) + omega^4,

    |G(i omega)|^2 = N / D = 1 - omega^2 (M + omega^2) / D.                                   (1)

Three consequences.

- **String stability for all omega is M >= 0** (the usual criterion): (1) is below 1 at every
  omega if and only if M + omega^2 >= 0 for every omega > 0.
- **An unstable equilibrium is unstable only at low frequencies**: with M < 0, |G| > 1 exactly
  on 0 < omega < sqrt(-M) and |G| < 1 above. The audit's lowest frequency 0.02 rad/s therefore
  sees every equilibrium with M < -4e-4 s^-2; the rollout penalty of the specification, whose
  lowest frequency is 0.05 rad/s, sees only M < -2.5e-3 s^-2. The band -2.5e-3 < M < 0 is its
  blind band, and an equilibrium pushed there by the penalty is audited unstable.
- **The expansion**: for omega well below the slow pole f_s / |f_dv + f_v| of G, (1) gives
  |G|^2 = 1 - omega^2 M / f_s^2 + O(omega^4), so the Jacobian criterion is the omega -> 0 limit
  of the gain condition. This is the content of the combined penalty of D110: the Jacobian
  term of the memoryless view handles the limit the rollout cannot reach. The slow pole of the
  audited laws is 0.02-0.045 rad/s (table `lowfreq_expansion`), the same order as the audit's
  lowest frequencies, so the audits follow the exact form (1) (to 1e-4 for the memoryless
  laws) rather than the truncated expansion; the sign of |G| - 1 agrees with M at 96-100 % of
  the speeds, the exceptions lying in -omega^2 < M < 0.

For a recurrent law the transfer function is not the rational G(p) above: the hidden state adds dynamics. The
memoryless view (the window filled with the constant state; the derivatives summed over the
window positions) gives the static gains, i.e. the omega -> 0 limit of the true response
provided the hidden dynamics are asymptotically stable at the equilibrium; the needles of M3
(f_dv of -35 to -237 1/s) are static gains that the finite-amplitude rollout saturates (clip
to [-8, 4] m/s^2), which is why the rollout and the Jacobian terms disagree there and why the
guard of D110 bounds |f_dv| and |f_v| at 3 1/s. Table `lowfreq_expansion` reports how far the
numerical gains of the audits at 0.02-0.03 rad/s are from (1) per architecture.

## 3. The certificate as a theorem (D116)

Hybrid f = F + r with F the IDM core, |r| <= r_max, and per-input bounds |dr/ds| <= B_s,
|dr/d dv| <= B_dv, |dr/dv| <= B_v (products of the spectral norms of the layers, the output
scale r_max times the tanh slope, and the input scaling; `ResidualIDM.jacobian_bound`).

**Proposition (a priori certificate).** Fix a speed v_e. Let I(v_e) be the set of spacings at
which the hybrid can have an equilibrium, I = { s : |F(s, 0, v_e)| <= r_max } (an interval,
because F rises with s). If at every s in I the minimum of

    M(F_s + d_s, F_dv + d_dv, F_v + d_v)  over the box |d_j| <= B_j

is non-negative, and F_s + d_s > 0 and (F_v + d_v) + (F_dv + d_dv) < 0 hold throughout the
box, then every equilibrium of the hybrid at speed v_e is locally and string stable, whatever
the residual does, before and after any fine-tuning that keeps the bounds.

Proof. The hybrid's equilibrium at v_e lies in I since F + r = 0 and |r| <= r_max. Its partial
derivatives are F_j + d_j with d_j = dr/dx_j, |d_j| <= B_j, so its margin and its local-stability
signs are those of some point of the box. The implementation (`certificate.py`, `_a_priori`)
scans I and takes the exact minimum over the box; `enforce_budget` sets the Lipschitz constant
so that the condition holds at every speed of the grid with a safety factor (D86).

**Explicit first-order form.** Expanding M at the core's partials,

    M(F + d) = M(F) + 2 d_v (F_v + F_dv) + 2 F_v d_dv - 2 d_s + d_v^2 + 2 d_v d_dv
            >= M(F) - 2 B_v |F_v + F_dv| - 2 B_dv |F_v| - 2 B_s - 2 B_v B_dv,

so with a common bound B the certified margin is at least M(F) - K B - 2 B^2 with
K = 2 (|F_v + F_dv| + |F_v| + 1): the core margin m of the calibration (D72, 0.2 in E4) buys a
residual budget of order m / K. This is why the certified cores need the margin and why the
accuracy cost of the certified hybrid sits in the core (RMSE 4.5 m against 2.8 m of a free core
with the same r_max, M7 section 5): the margin moves the core away from the calibrated optimum,
the residual cannot move it back by more than the budget.

**Tightness.** Table `certificate_tightness` compares the certified lower bound with the
audited margin of the trained hybrids at every grid speed (200 runs, four amplitudes, before
and after fine-tuning): the slack is the price of certifying against the worst residual rather
than the trained one; a negative slack would contradict the proposition and is reported as
such (expected 0).

## 4. The price of a common equilibrium (D123 a)

A memoryless law has one equilibrium spacing s_e(v) per speed. A driver d who holds spacing
s_d(v) at that speed produces, in closed loop, a steady spacing error s_d(v) - s_e(v) that no
memoryless law can remove; over the near-steady samples of the data the closed-loop spacing
RMSE of any memoryless law is therefore bounded below by the dispersion of s_d(v) over
drivers (the best s_e(v) is the conditional mean):

    RMSE_steady^2 >= Var_d [ s_d(v) ]   (per speed, over near-steady samples).

A recurrent law reads s_d from the window and can follow every driver: it needs no common
equilibrium, and the audit finds none in the band at 30-40 % of the speeds (E1, "outside" and
"none"). Forcing one (the existence term of D80 alone, `e2_existence`) costs GRU +1.8 %, LSTM
+6.7 % and PERL +7.6 % of RMSE: the price of the common equilibrium is real but small, and
the large cost of the penalties for the recurrent models (+15 to +29 %) is the price of
stability, not of the equilibrium. Figure and table `band_width` give the relative band width
(s_95 - s_5) / s_50 and the per-driver dispersion of the equilibrium spacing per speed for
HighD and I-80, which bound the attainable gain of memory on the near-steady part of the data.

## 5. How stability penalties get gamed (M3-M8)

| loophole | symptom | where seen | fix |
|---|---|---|---|
| needles | a stiff equilibrium with f_dv of -35 to -237 1/s satisfies the linearised criterion trivially (M > 0 through 2 f_v f_dv) while the finite-amplitude response saturates | GRU, linearised gain penalty (M3) | guard relu(abs(f_dv) - c) + relu(abs(f_v) - c), c = 3 1/s (D110); finite-amplitude rollout term |
| out-of-band equilibria | the only equilibrium at a speed lies outside the spacing band of the data, where the penalty never samples | GRU, LSTM, PERL (E1: 30-39 % of the speeds) | anchor the equilibria to the band and audit the status "outside" (D78, D79); existence term at the band edges (D80) |
| no equilibrium | the model has no root at a speed, the Jacobian term is undefined and the penalty is zero | PERL under the combined penalty (65 % of the speeds) | existence term with a weight that cannot be out-traded; report "none" as unstable in the band share (D92) |
| locally unstable with abs(G) < 1 | the discrete transfer function has gain below 1 on the unit circle while a pole lies outside it; the gain criterion alone is satisfied | LSTM, linearised gain penalty (M3) | local-stability terms relu(-f_s) + relu(f_v + f_dv) in every penalty; the audit's sign flags |
| blind band | the rollout gain penalty cannot see instabilities with -2.5e-3 < M < 0 s^-2 (unstable only below 0.05 rad/s, section 2) | GRU, LSTM with the rollout penalty (E2: audited unstable at 0.05-0.1 rad/s) | Jacobian term of the memoryless view (combined penalty, D110); it does not rescue the recurrent models (M7 section 6) |
| epoch choice | the best validation epoch of a penalised run is one where the penalty is violated | all penalised runs before D89 | penalty-aware choice of the best epoch (D89) |

The table is the methodological contribution: each row is a mechanism, not an observation,
and each fix is in the pipeline with a test.
