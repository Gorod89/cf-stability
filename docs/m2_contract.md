# M2 contract: models, training, closed-loop evaluation

Binding for the M2 modules. Conventions of `docs/data_contract.md` apply (SI units, 10 Hz,
`dv = v - v_lead`, semi-implicit Euler, acceleration clipped to `[-8, 4]`, collision `s <= 0`).

## 1. Model interface

`cf_stability.models.base.CFModel(nn.Module)` (exists): `window` (history steps needed),
`forward(state_history [B, steps >= window, 3]) -> [B]` acceleration in m/s^2 from raw SI
states `(s, dv, v)`, `acc(s, dv, v)` for memoryless use. Additions of M2 (in `base.py`):

| Member | Meaning |
|---|---|
| `name: str` | registry name |
| `memoryless: bool` | `True` when the output depends on the last state only (`window == 1`) |
| `trainable: bool` | `False` for models that are fitted without gradient steps (IDM, kNN, Persistence, Newell) |
| `extra_loss(batch) -> Tensor` | additional training loss of the model (default: zero); PIDL puts its physics term here |
| `fit(events, context) -> dict` | non-gradient fitting on the training events (kNN index, IDM / Newell calibration); returns a JSON-serialisable summary; default: nothing |
| `config() -> dict` | constructor arguments needed to rebuild the model from a checkpoint |

Every model standardises its inputs internally with `InputScaler` (buffers `center [3]`,
`scale [3]`), so that callers always pass raw SI states and the stability analysis can
differentiate through the whole model. Two kinds of scaler values:

* data scaler: mean and standard deviation of `(s, dv, v)` over the training samples
  (MLP, GRU, LSTM, kNN, PIDL, PERL);
* fixed physical scaler: `center = (0, 0, 0)`, `scale = (20 m, 2 m/s, 10 m/s)` (ResidualIDM:
  the bound of its Jacobian must not depend on the dataset, see section 3).

## 2. Models (`cf_stability/models/`)

| Name | Module | Definition |
|---|---|---|
| `idm` | `idm.py` | exists. New: `bounded=True` stores the parameters as logits and maps them into `IDM_BOUNDS` with a sigmoid (needed whenever the parameters are learnable). `fit` = global calibration on the training events (`calibrate_global`). |
| `newell` | `newell.py` | Newell's model in the adapted form of PERL (Long et al. 2023, eq. 6): the follower repeats the acceleration of its leader after the time a wave needs to travel the gap, `a(t) = a_lead(t - s(t) / w)`; one parameter, the wave speed `w` in [1, 60] m/s; window 30 (the delay is capped at 2.9 s); leader acceleration from `v_lead = v - dv` of the history. `fit` = one-dimensional search of the closed-loop objective of the IDM calibration (log grid of 40 values, then golden section). No feedback on the gap: string-neutral by construction, drifts in closed loop. |
| `ovm` | `ovm.py` | Optimal velocity model with Newell's triangular speed-gap relation: `a = (min(v0, (s - s0) / tau) - v) / tau`, parameters `(v0, tau, s0)`, bounds `v0 [10, 45]`, `tau [0.3, 3]`, `s0 [0.5, 10]`; `fit` = global calibration like the IDM. String unstable on the whole congested branch (`f_s = 1 / tau^2`, `f_v = -1 / tau`, `f_dv = 0`): the known-unstable reference of the stability module. |
| `persistence` | `persistence.py` | exists. |
| `knn` | `knn.py` | k = 10 nearest training samples in the standardised `(s, dv, v)` space, prediction = mean of their accelerations; brute force in torch in chunks; at most `max_points` (200 000) training samples, drawn with the run seed. |
| `mlp` | `mlp.py` | 3 -> 64 -> 64 -> 1, tanh, linear output. |
| `gru`, `lstm` | `recurrent.py` | one layer, hidden 64, window 30 steps, linear head on the last hidden state. |
| `pidl` | `pidl.py` | MLP (as `mlp`) plus an IDM. Prediction = MLP. `extra_loss = alpha * MSE(mlp(x_c) - idm(x_c))` on collocation states `x_c` drawn uniformly from the box of the training states (1 % - 99 % quantiles), as many as the batch has samples; `alpha` default 1.0; IDM parameters fixed to the global calibration of the training fold or learnable (bounded), `idm_learnable` default `false`. |
| `perl` | `perl.py` | `a = newell(history) + lstm_residual(history)`; LSTM as in `lstm`, head initialised to zero; the wave speed comes from the calibration on the training events and stays fixed. |
| `residual_idm` | `residual_idm.py` | `a = idm(theta) + r_max * tanh(lipschitz * g(x_tilde))`, `g` = MLP 3 -> 64 -> 64 -> 1 with tanh and spectral normalisation (`torch.nn.utils.parametrizations.spectral_norm`) on every linear layer, `x_tilde` from the fixed physical scaler; `r_max` default 1.0 m/s^2, `lipschitz` default 1.0; `theta` from the global calibration of the training fold, `idm_learnable` default `false` (bounded when learnable). Method `jacobian_bound() -> Tensor [3]`: `r_max * lipschitz * prod(exact spectral norms of the effective weights) / scale`, the bound of `|d r / d(s, dv, v)|`. |

Factory: `cf_stability.models.build_model(model_cfg: Mapping, context: ModelContext) -> CFModel`
with `ModelContext(center, scale, box_low, box_high, idm_params, newell_params, dt, seed)`;
registry by `model_cfg["name"]`. Checkpoint: `{"name", "config", "state_dict"}`;
`load_model(path) -> CFModel` (eval mode, CPU unless `map_location` is given).

Notes from the implementation:

* PIDL clips the IDM acceleration of the physics term to `[-8, 4]` m/s^2: in the corners of the
  collocation box the unclipped IDM returns tens of m/s^2 (highD box, global parameters:
  -72 m/s^2 at s = 7 m, v = 27 m/s, dv = 2.5 m/s).
* The residual of ResidualIDM starts at exactly zero through an antisymmetric initialisation
  (duplicated hidden units with opposite output weights): with spectral normalisation the
  scale of a weight matrix cannot be used to make the initial output small.
* Spectral normalisation runs a power iteration in every forward pass in training mode; the
  stability analysis and the certificate read the model in eval mode and use exact norms (SVD).
* Newell does not depend on `dv` (`f_dv = 0`).

## 3. Training (`cf_stability/train/trainer.py`)

* Tensors of a fold: `EventTensors` built once per event set on the device (`pad_events` plus
  the index of every valid window end).
* Loss of a step: `MSE(a_hat, a)` over a batch of `batch_size` (4096) window ends drawn without
  replacement per epoch + `rollout.weight` x rollout loss + `model.extra_loss(batch)`.
* Rollout loss (default weight 1.0, `0` disables it): `rollout.segments` (256) segments per
  step, each starts at a random valid window end, observed warm-up of `window` states,
  teacher-free rollout of `rollout.horizon` = 50 steps (5 s) behind the recorded leader;
  `MSE(s) / var_s + MSE(v) / var_v` with the variances of the training samples. Segments that
  would pass the end of their event are shortened by masking.
* Adam, learning rate 1e-3, at most `max_epochs` (100) epochs; one epoch = one pass over the
  window ends, capped at `steps_per_epoch` (300) steps.
* Early stopping on the validation closed-loop spacing RMSE (mean over the validation events of
  the full-event rollout), patience 10 epochs, best weights restored.
* Seeds 0-4: `torch.manual_seed`, numpy generator and the generators of the samplers are all
  derived from the run seed; cuDNN deterministic.
* Models with `trainable = False` skip the gradient loop: `fit`, then evaluation.

## 4. Closed-loop evaluation (`cf_stability/train/evaluate.py`)

`evaluate_closed_loop(model, events, *, warmup=30, batch_events=2048, device) -> DataFrame`
(one row per event: `event_id, follower_id, site, n_scored, rmse_s, rmse_v, collided, min_s`)
and `summarise(df) -> dict` (`rmse_s_mean`, `rmse_s_median`, `rmse_s_pooled`, the same for `v`,
`collision_rate`, `n_events`).

**Every model is scored on the same samples:** the first `warmup = 30` samples (3 s) of an event
are observed and given to the model as history; the follower is rolled from sample 29 to the end
of the event; errors are computed over the samples 29 ... n - 1. Memoryless models use the state
of sample 29 only. Events are at least 150 samples long, so at least 121 samples are scored.

## 5. Entry point and outputs

`python scripts/train.py data=<event set> model=<name> fold=<k> seed=<s> [split=driver|site]`
(hydra, multirun capable). Output directory
`runs/<experiment>/<event set>/<model>/<split>_fold<k>_seed<s>/` with

* `metrics.json`: resolved config, `config_hash`, git revision, seed, fold, numbers of events
  and samples per part, best epoch, history per epoch (training losses by term, validation
  closed-loop RMSE), closed-loop summaries of the validation and test parts, fit summary,
  number of parameters, wall time, device;
* `model.pt`: checkpoint of the best weights;
* `test_events.parquet`: per-event closed-loop metrics on the test part (needed for the
  bootstrap over drivers in M6).

`experiment` defaults to `dev`. IDM, Newell, kNN and persistence run through the same script.
