# M11 contract: extensions after the submission (hypotheses H-N1 to H-N5, algorithms A1 to A3)

Started 19 October 2026 after the submission of the manuscript (paper/v2 is frozen: nothing of M11 enters it; the
results go to `docs/m11_report.md`, `runs/_tables/m11/` and, where a law enters the corridor, the usual corridor
tables as further rows). Every hypothesis has a decision rule fixed below before the runs. Numbers of decisions
D127-D133; deviations are recorded by the agents in their reports and by the main session in `docs/decisions.md`.

Constraints that hold for every agent: the shared `.venv` (no new packages, no upgrades); new keys of
`configs/train.yaml` and `configs/run_corridor.yaml` must be hash-neutral at their defaults (as D110 did for the
combined penalty: the config hash of existing runs must not change; check with `make_tables.py` before and after);
the GPU is shared with other sessions (one or two training processes at a time per agent, `min_free_ram_gb: 6`);
restartable queues under `configs/queue/`, logs in the scratchpad; tests for every new module; no AI mentions in
any committed file; nothing is committed by the agents (the main session commits).

## D127 - H-N1: predictive validity of single-vehicle metrics for the macro error (analysis, agent C)

Units: the law-corridor pairs of the pooled correlation (`correlation_pooled`, 37 pairs) plus the six pairs of the
ablation laws as a separate check. Response: the macro error of the pair (mean over seeds, `laws.csv`). Predictors
per law, mean over its member runs where they are per run: (a) the pre-specified share of unstable equilibria
(H1.1 statistic); (b) the share of grid speeds not stable in the band; (c) the maximal numerical gain over speeds
and audit frequencies; (d) the maximal gain at frequencies <= 0.1 rad/s; (e) the platoon transient growth of the
members (`platoon.json`; run the platoon step for the members that lack it); (f) the frozen gain along corridor
trajectories: for the seed-0 run of every pair, 20 000 follower states (s, dv, v) sampled uniformly from
`trajectories.npz` (states with a leader inside the 50 m leader range), the memoryless partials of the law at those
states (autograd; the memoryless view for recurrent laws, as the audit), the frozen gain max over the audit
frequencies, and two statistics: the share of states with gain > 1.02 and the mean log gain; (g) the contact rate
per 1 000 vehicle-km; (h) the asymmetry index. Statistic: Spearman per corridor and pooled with the cluster bootstrap
over laws and the permutation test of `cf_stability/eval/stats.py` (as `correlation_clustered`), and for every
predictor P the paired cluster-bootstrap interval of rho_P - rho_(a). Decision rule: P "predicts better" when that
interval excludes zero and P ranks E (residual_idm_certified) better than D (residual_idm_margin_free_r0.3) on both
sites; H-N1 is confirmed when the frozen-gain statistic (f) predicts better, partially confirmed when another
predictor does, refuted when none does. Table `m11/predictive_validity` (predictor, rho per corridor, pooled rho
with interval, difference to (a) with interval, D-vs-E rank).

## D128 - A1 and H-N3: the analytic windowed-gain penalty for recurrent laws (agent A)

Penalty kind `windowed_gain` in `cf_stability/stability/penalties.py`. At a penalty step, for `n_equilibria` = 16
speeds drawn as the gain penalty draws them, the model is driven for its window W (the `window` of the model
config, 30 steps at 0.1 s) on the equilibrium input (s*, 0, v*) of the speed; the acceleration output is
differentiated once with respect to the whole input history tensor [W, 3], which gives the lag Jacobians J_{x,k}
of Appendix A of the manuscript in one backward pass per speed. From them F_x(z) = sum_k J_{x,k} z^{-k} and the
windowed transfer function G_w(z) of equation (A2) at z = exp(i omega dt) for the 25 audit frequencies 0.02-2
rad/s, and the low-frequency coefficient M_w (equation windowed_margin) in closed form. Penalty = mean over speeds
and frequencies of relu(|G_w| - 1 + 0.02)^2 + relu(0.5 - M_w) (the margin of the Jacobian penalty) + relu(rho - 0.995)
with rho the spectral radius of the closed-loop (e, u) block of the windowed law (eigenvalues of its companion
matrix, `torch.linalg.eigvals`), every term weighted by `train.penalty.weight`. `every`: 1 in the pilot; raised to 2
or 4 only if the penalty step costs more than three times the plain step (recorded). The existence term of the
rollout penalty is kept as in `combined` (an equilibrium must exist in the band). Tests: the penalty of a linear
recurrent toy law equals the gain of its known transfer function; gradients by finite differences; the M_w of the
penalty equals `full_history_audit.py`'s value on a stored run.

Runs: `e2_windowed_pilot` (GRU, LSTM, PERL; highD fold 0, seed 0; weights 0.1, 1, 10; steps train, audit, platoon),
weights chosen by the rule of D85, written into `e2_windowed` (5 folds, seed 0, 15 runs, steps train, audit,
platoon); `full_history_audit.py` extended to the new experiment (poles, windowed gain, M_w of the trained runs);
the long-window check of the pilot runs of fold 0: the rollout gain with a 380 s rollout measured over the last
252 s (as `e2_horizon_pilot`) at omega in {0.02, 0.05, 0.1} must stay below 1.02. Then the fine-tuning on NGSIM
I-80 as `m5_laws` does (`init_from` the e2_windowed runs, 5 folds), law files `gru_windowed`, `lstm_windowed`,
`perl_windowed` (`configs/corridor/laws.yaml`, `build_corridor.py scenarios=false`), corridor runs on i80_p0-p2 and
us101_p0-p2 with seeds 0-9 (180 runs, `workers=3`), `corridor_metrics.py`, `corridor_asymmetry.py`, then the tables
(`make_tables.py`, `corridor_metrics.py compute=false tables=true`). Decision rule of H-N3 (5-fold means on highD,
against the E1 run of the same architecture): confirmed when GRU and LSTM reach a stable share in the band >= 0.9
at a test spacing RMSE increase <= 10 % and the long-window check holds; partial when the share is reached at a
larger increase; refuted otherwise. PERL is reported, not judged.

## D129 - A2 and H-N2: the two-timescale certified law (agent B)

Model `residual_idm_tt` (`cf_stability/models/residual_idm_tt.py`, registered as "residual_idm_tt"): the certified
hybrid of E4 (margin core, Lipschitz-bounded residual with `r_max` 0.3) whose core time gap is modulated by a slow
latent, T(z) = T_c exp(z ln 1.5), z in [-1, 1] (T from T_c/1.5 to 1.5 T_c; s_0 and the other parameters fixed);
the residual takes (s, dv, v, z). The latent follows z_{t+1} = z_t + (dt / tau_z)(tanh(g_phi(h_t)) - z_t) with g_phi
a one-layer GRU (hidden 16) over the model's window, z_0 = 0, gradients flowing through z; tau_z in {15, 30, 60} s
in the pilot. Audit: the model exposes `frozen_z` (None: the online latent); the standard audit and the certificate
run at frozen z in {-1, -0.5, 0, 0.5, 1}; the exact certificate (`certificate_exact.py` machinery) at 21 values of
z in [-1, 1] before and after fine-tuning; the derivative budget of the residual is the same for every z (z is an
input of its own scale). Runs: pilot highD fold 0 seed 0 (3 runs, tau_z), chosen tau_z = the smallest validation
RMSE among the runs with 100 % stable equilibria at every frozen z (else the largest stable share); `e4_tt` (5
folds, seed 0), fine-tuning on NGSIM I-80 (`e4_tt_ft`, 5 folds, with the certificate rechecked), law
`residual_idm_tt` (members = the five fine-tuned folds, drawn as `residual_idm_certified` draws them; the latent
runs online per vehicle), corridor runs on both sites, seeds 0-9 (60 runs), metrics, asymmetry, tables. Decision
rule of H-N2 (5-fold means on highD): confirmed when the test spacing RMSE is <= 1.05 x the GRU's E1 RMSE, every
frozen-z audit has 100 % stable equilibria and the certificate holds at all 21 z after fine-tuning; partial when
the RMSE is <= 1.15 x; refuted otherwise. The corridor macro error is compared with E (paired by seed) and reported.

## D130 - A3: the robust (non-equilibrium) certificate, the budget allocation and the safety filter (agent C)

Module `cf_stability/stability/robust.py`. For a law with derivative boxes f_s in [a_1, b_1], f_v in [a_2, b_2],
f_dv in [a_3, b_3] over a region R, the follower's perturbation dynamics e' = w - u, u' = f_s e + f_dv (w - u) + f_v u
(e the spacing error, u the speed error, w the leader's speed perturbation, output u) is a polytopic LPV system with
the eight box corners as vertices. The bounded-real LMI with a common quadratic storage function P (2 x 2, three
parameters) is solved without an SDP solver: P = L L^T and gamma from a bisection, the maximal eigenvalue of the
vertex LMI matrices minimised with scipy (Nelder-Mead and BFGS starts), and every certificate verified afterwards
in float64 (all vertex eigenvalues <= -1e-9). gamma < 1 certifies the L2 gain w -> u below one for every trajectory
whose derivatives stay in the boxes (incremental string stability over R). Regions: R = the spacing band of the
NGSIM I-80 data at the grid speeds (the band of the audit) and, separately, the whole feasible-gap set I(v) of the
certificate. Laws: the certified hybrid at r_max 0.1, 0.2, 0.3 (E), 0.5, the free hybrid C (residual_idm_free_r0.3),
the ablation variants D (residual_idm_margin_free_r0.3) and B (idm_core_margin), the IDM, and the memoryless
networks MLP and PIDL with their empirical boxes (partials sampled at 20 000 states of R; for them the result is a
necessary-condition check, said so). Boxes of the hybrids: the core's derivative ranges over R plus the residual
budget. Decision rule (the explanation of the ablation): "explains" when E has gamma < 1 on the band and D has gamma
>= 1 or an infeasible LMI on the same region; the table `m11/robust_certificate` reports gamma per law and region
next to the macro error of the law on both sites. Budget allocation: for the margin core of E, the budgets (B_s,
B_v, B_dv) that maximise B_s + B_v + B_dv subject to gamma <= 1 on the band (coordinate ascent with bisection),
compared with the uniform allocation of r_max 0.3; reported, not trained (training with per-input budgets is a
stretch goal).

Safety filter (`cf_stability/corridor/safety.py`, applied in the corridor loop when `sim.safety_filter` is set;
hash-neutral default off): barrier h = s - s_min - v tau_s - max(0, v^2 - v_lead^2) / (2 b) with s_min = 2 m,
tau_s = 0.5 s, b = 4 m/s^2; the acceleration is clipped to the largest value with dh/dt + alpha h >= 0 under a
constant leader speed, alpha = 1 1/s. The filter is inactive wherever that bound exceeds the law's acceleration, so
the audit of the equilibria of the band is unchanged (checked: h > 0 and the bound inactive at every band
equilibrium of the laws below). Runs: laws mlp, gru, lstm and residual_idm_certified with the filter on i80_p1 and
us101_p1, seeds 0-4 (40 runs, scenario variants `*_filter` or a run-config override that enters the hash),
metrics. Decision rule: the filter "works" when the contact rate of the three networks falls to zero on both
scenarios and the macro error of the certified hybrid changes by at most 0.02; the networks' macro error change is
reported.

## D131 - H-N4: the empirical gain of human drivers against the laws (agent C)

For the FollowNet highD and the NGSIM I-80 event sets: the leader-to-follower speed gain estimated by Welch's
method (segments of 40 s with a Hann window, 50 % overlap) on events of at least 40 s, detrended speeds, per speed
bin of 5 m/s (the mean speed of the event) and frequency, for the frequencies 0.3-2 rad/s (lower frequencies are not
estimable from events of this length; said so); the median and the 25-75 % range over events per cell. Laws: the
numerical gains stored in `stability.json` of the E1 runs (fold 0-4, seed 0) of the five unconstrained networks, the
IDM and k-NN, interpolated to the same cells. Statistic: the excess log gain of a law = mean over cells of
log |G_law| - log |G_emp, median|, with a cluster-bootstrap interval over events (the empirical side) and over folds
(the law side). Decision rule: confirmed when the excess is positive with an interval excluding zero for at least
four of the five networks and the IDM's excess is negative; refuted when fewer than three networks exceed the
drivers. Table `m11/empirical_gain`, figure `empirical_gain` (gain curves per speed bin: drivers against the laws).
Stretch goal: penalty kind `gain_match` (the rollout gain penalty with the empirical median gain as the target
curve instead of 1), one GRU pilot on fold 0, reported only.

## D132 - H-N5: member weights of the heterogeneous certified hybrid matched to the fundamental diagram (agent B)

The law `residual_idm_certified_het` draws a per-event certified core per vehicle uniformly. New law
`residual_idm_certified_het_fd`: the same cores, drawn with weights proportional to exp(beta_T T_i + beta_s s0_i)
(a two-parameter exponential tilt of the per-event (T, s0)); (beta_T, beta_s) minimise the squared distance between
the mixture's equilibrium flow-density curve (mean spacing at speed v: sum_i w_i s_i(v) plus the vehicle length;
k = 1 / mean spacing, q = k v, evaluated at the density bins of the ground truth) and the binned fundamental
diagram of the ground truth of i80_p1 (`macro.json` of the scenario), with the uniform weights as the start and
|beta| <= 5. Every member keeps its certificate (the cores are the certified ones; the weights change nothing in
the laws). Runs: the law on both sites, seeds 0-9 (60 runs), metrics, asymmetry, tables. Decision rule: confirmed
when the macro error of the tilted law is below the uniform het law on both sites with paired-by-seed intervals
excluding zero; the comparison with E is reported.

## D133 - reporting

Each agent writes its section of `docs/m11_report.md` (what was done, numbers, deviations, commands run) and
leaves the tables in `runs/_tables/m11/` (CSV + Markdown twins through `cf_stability/eval/tables.py` conventions)
or in the corridor tables. The main session assembles the report, the verdicts and the decision log. Agent
ownership: A - `cf_stability/stability/penalties.py`, `scripts/analysis/full_history_audit.py`, the queues
`e2_windowed*`, `m11_ft_windowed`, the law entries `*_windowed`; B - `cf_stability/models/residual_idm_tt.py`,
`cf_stability/models/__init__.py`, `cf_stability/stability/certificate.py` (extension only), `cf_stability/corridor/
laws.py` (tilted het law, the tt law), the queues `e4_tt*`, `m11_ft_tt`; C - `cf_stability/stability/robust.py`,
`cf_stability/corridor/safety.py`, `cf_stability/corridor/loop.py` (the filter hook), `scripts/analysis/
predictive_validity.py`, `scripts/analysis/robust_certificate.py`, `scripts/analysis/empirical_gain.py`,
`configs/run_corridor.yaml` (the filter key). Shared files (`configs/corridor/laws.yaml`, `configs/
corridor_metrics.yaml`, `cf_stability/eval/tables.py`, `corridor_tables.py`) are edited by appending only, each
agent its own block, and the agent says in its report what it appended.
