# M3 contract: stability module

Binding for the M3 modules in `cf_stability/stability/`. Conventions of `docs/data_contract.md`
and `docs/m2_contract.md` apply: `a = f(s, dv, v)`, `dv = v - v_lead`, 10 Hz, semi-implicit
Euler. All stability computations run in float64 (`copy.deepcopy(model).double().eval()`).

## 1. Theory

Platoon of identical vehicles, deviations from an equilibrium `(s_e, 0, v_e)`, `f(s_e, 0, v_e) = 0`:
position `y_n`, speed `u_n`, `ds = y_(n-1) - y_n`, `ddv = u_n - u_(n-1)`.

Continuous time, memoryless law with partial derivatives `f_s, f_dv, f_v`:

    G(i w) = (f_s - i w f_dv) / (f_s - w^2 - i w (f_dv + f_v))

* local stability: `f_s > 0` and `f_v + f_dv < 0`;
* string stability, `|G(i w)| <= 1` for all `w`: margin `M = f_v^2 + 2 f_v f_dv - 2 f_s >= 0`;
* when `M < 0`, `|G| > 1` exactly on the band `0 < w^2 < -M`.

Discrete time, the integration scheme of the project (`u[k+1] = u[k] + dt a[k]`,
`y[k+1] = y[k] + dt u[k+1]`, the same for the leader), `z = exp(i w dt)`:

    G_d(z) = (dt^2 F_s z - dt F_dv (z - 1)) / ((z - 1)^2 + dt^2 F_s z - dt (F_dv + F_v) (z - 1))

with `F_x = f_x` for a memoryless law. For a law with a history window of `W` states,
`a[k] = f(state[k - W + 1 .. k])`, the Jacobian `J [W, 3]` with respect to the history gives
`F_x(z) = sum_j J_x[W - 1 - j] z^(-j)`, `j = 0 .. W - 1` (`j = 0` is the latest state). This
"linearised window response" is exact for small perturbations and is used to check the rollouts
and as a fast analysis of recurrent models. The memoryless view of a windowed model uses
`f_x = sum_j J_x[j]`, the derivative with respect to a constant shift of the whole history,
i.e. the limit `w -> 0`.

## 2. Equilibria (`equilibrium.py`, fixed API)

`V_GRID` = 5, 6, ..., 30 m/s. `find_equilibria(model, v=V_GRID, s_min=1, s_max=200, n_scan=400)`
evaluates `f(s, 0, v)` on a constant history for `n_scan` spacings, takes the first upward zero
crossing (`f < 0` then `f >= 0`) and refines it by bisection (50 halvings). Result `Equilibria`
with `v, s, found, n_crossings, status`:

| status | meaning |
|---|---|
| `ok` | one upward crossing |
| `multiple` | several upward crossings; the first one is used |
| `none` | no upward crossing in [1, 200] m: "no equilibrium" |
| `indifferent` | `|f| < 1e-9` for every spacing (e.g. Newell's law): every spacing is an equilibrium; the nominal spacing `s = 2 m + 1.5 s * v` is used for the frequency response |

## 3. Analytic criterion (`analytic.py`, fixed API)

`partials(model, s, v, create_graph=False) -> (f_s, f_dv, f_v)` by autograd through the constant
history; central finite differences with steps `0.25 * scale` of the model's scaler for models
with `differentiable = False` (kNN). `history_jacobian(model, s, v) -> [n, W, 3]`.
`criterion(f_s, f_dv, f_v) -> dict` with `local_stable`, `rational` (`f_s >= 0, f_dv <= 0,
f_v <= 0`), `margin`, `string_stable`, `band_upper` (`sqrt(-M)` or 0).
`transfer_continuous`, `transfer_discrete`, `transfer_windowed` return complex gains `[n, n_w]`.

## 4. Numerical frequency response (`frequency.py`)

Two-vehicle closed-loop rollout with `rollout_model` at each equilibrium and each frequency:
leader speed `v_e + A sin(w t)`, `A = 0.2` m/s, leader position by the integration scheme,
follower history filled with the equilibrium state, 25 frequencies log-spaced from 0.02 to
2.0 rad/s.

* **Deviation from the specification** (120 s, first 60 s discarded, FFT): the period at
  0.02 rad/s is 314 s, so 60 s cannot hold one period and the slow pole of an IDM-like law has
  a time constant of about 30 s. Per frequency: discarded time = max(60 s, 1 period), measured
  time = the smallest whole number of periods >= max(60 s, 2 periods). The amplitude at `w` is
  the least-squares projection of the speed on `sin(w t), cos(w t), 1`, which equals the FFT
  line when `w` is a line of the window.
* gain = follower amplitude / leader amplitude, phase, relative residual of the fit;
  `unstable = max gain > 1.02`; flags for clipping, `v = 0` and collisions during the rollout.
* Frequencies are processed in groups of similar length to bound the cost.

## 5. Audit (`audit.py`, `scripts/audit_stability.py`)

`audit_model(model, context, cfg) -> dict`, written as `stability.json` next to the model:

* per equilibrium: `v, s, status, in_support` (speed inside the 1 % - 99 % range of the training
  states), partial derivatives, criterion flags and margin, numerical gains, maximum gain and
  its frequency, numerical flag, analytic maximum gain on the same frequencies (continuous,
  discrete, windowed);
* summary: numbers of grid points, equilibria, equilibria without solution, share of string
  unstable equilibria (numerical and analytic; all equilibria and those in support), agreement
  of the two flags, largest gain, shares of locally unstable and of non-rational equilibria;
* primary numbers (D75): `grid_numerical` and `grid_sign`, the shares of the grid speeds with a
  stable equilibrium, an unstable one, none, an indifferent law, an undefined flag; they sum to
  1; for all speeds and for the speeds in support.

`python scripts/audit_stability.py run=<run directory>` or `experiment=<name> data=<set>`
(all runs of an experiment); hydra config `configs/audit_stability.yaml`, options in
`configs/stability/default.yaml`.

## 6. Platoon growth test (`platoon.py`)

50 identical vehicles behind a leader with a prescribed speed profile, all starting at the time
gap `gap_s0 + gap_T * v` of the leader's first speed (`start_gap: time_gap`, D73) or at the
equilibrium of the law (`start_gap: equilibrium`); vehicle `n` follows the simulated vehicle `n - 1`.
Profiles: (a) leader speed of an OpenACC run (`configs/stability/platoon.yaml`: file and columns),
(b) synthetic braking pulse (from `v0` down by `dv_pulse` with deceleration `b_pulse`, hold,
back up). Output: standard deviation of the speed per vehicle index, collisions, and the
concave-growth error = RMSE between the model curve and the empirical curve over the vehicle
indices of the empirical curve, divided by the maximum of the empirical curve. Empirical curves:
standard deviation of the speed per platoon position in the OpenACC run (computed from the raw
file), and the values of `configs/empirical_growth.yaml` when present.

## 7. Penalties (`penalties.py`) and their use in the trainer

* `JacobianPenalty` (memoryless and residual models): equilibria at `n_equilibria` (16) speeds
  drawn uniformly from [5, 30] m/s, redrawn every epoch, `s_e` from `find_equilibria` without
  gradient; `mean over the equilibria of relu(m - M) + relu(-f_s) + relu(f_dv) + relu(f_v + f_dv)`,
  `m = 0.5`, derivatives with `create_graph=True`. Speeds without equilibrium are counted and
  enter the existence term.
* `GainPenalty` (recurrent models): differentiable two-vehicle rollouts of 40 s at 3 equilibria
  (speeds drawn every epoch) and the frequencies 0.05, 0.1, 0.2, 0.4, 0.8 rad/s, gain from the
  last 20 s by least-squares projection on `sin, cos, 1`; `mean of relu(gain - 1 + m_g)`,
  `m_g = 0.05`.
* `LinearGainPenalty` (extension D70, any differentiable model): mean over equilibria and
  frequencies of `relu(|G_d(w)| - 1 + m_g(w))` with the linearised window response of section 1;
  no rollout. Frequencies: the 25 of the audit (0.02-2.0 rad/s);
  `m_g(w) = 0.05 * min(1, (w / 0.1)^2)` (D76).
* Existence term of every penalty (D74): at the sampled speeds without equilibrium inside the
  speed range of the training data, `relu(f(1 m, 0, v) + m_e) + relu(m_e - f(200 m, 0, v))`,
  `m_e = 0.1 m/s^2`; `existence_weight = 0` switches it off.
* Trainer: `train.penalty = {kind: none | jacobian | gain | linear_gain, weight: lambda, ...}`,
  `loss = data loss + lambda * penalty`, `lambda` in {0.01, 0.1, 1, 10}; the penalty value and
  its parts are logged per epoch. With a penalty the initial weights are not eligible as the
  best epoch (D76).

## 8. Certificate for ResidualIDM (`certificate.py`)

`f = f_idm + r`, `|r| <= r_max`, `|dr/dx_j| <= B_j = r_max * lipschitz * prod(layer norms) / scale_j`.
For given IDM derivatives the guaranteed margin is the minimum of
`(f_v + d_v)^2 + 2 (f_v + d_v)(f_dv + d_dv) - 2 (f_s + d_s)` over `|d_j| <= B_j`: `d_s = B_s`,
and the minimum of `p^2 + 2 p q` over the rectangle of `p = f_v + d_v`, `q = f_dv + d_dv`, which
is attained at `q` on an edge and `p` at an end or at `p = -q`.

* `certificate_at_equilibria`: IDM derivatives at the equilibria of the hybrid (a posteriori).
* `certificate_a_priori`: minimum over all spacings that can be an equilibrium of any residual
  with `|r| <= r_max`, i.e. `|f_idm(s, 0, v_e)| <= r_max`; it does not depend on the weights and
  therefore survives fine-tuning.
* Both report, per speed, the guaranteed margin, the guaranteed signs (`f_s > 0`,
  `f_v + f_dv < 0`), whether the certificate holds, and the largest product `r_max * lipschitz`
  for which it would hold.
