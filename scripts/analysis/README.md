# One-off analysis scripts

Scripts behind the tables of `docs/m3_report.md`. They read the run directories under `runs/`
and print tables; nothing here is part of the pipeline or of the tests. The experiment scripts
of M4 replace them. First argument of every script: the repository root.

```bash
R=$(pwd -W)   # Git Bash, in the repository root
python scripts/analysis/m3_tables.py            $R                       # sections 3.2 and 3.3
python scripts/analysis/m3_per_speed.py         $R m3_linear_gain/gru    # equilibria of one run, speed by speed
python scripts/analysis/m3_partials_range.py    $R m2/idm m2/mlp         # ranges of the derivatives
python scripts/analysis/m3_steady_band.py       $R scripts/analysis/m3_outputs/steady_band.json   # section 3.6
python scripts/analysis/m3_crossings_in_band.py $R scripts/analysis/m3_outputs/steady_band.json   # section 3.6, all equilibria
python scripts/analysis/m3_eq_curves.py         $R                       # equilibrium gap of every model
python scripts/analysis/m3_platoon_gpu.py       $R scripts/analysis/m3_outputs/platoon_gpu.json        # section 3.5, reference models
python scripts/analysis/m3_platoon_penalised.py $R scripts/analysis/m3_outputs/platoon_penalised.json  # section 3.5, penalised models
python scripts/analysis/m3_platoon_gpu_table.py scripts/analysis/m3_outputs/platoon_penalised.json
python scripts/analysis/m3_profile_speeds.py    $R                       # leader speeds of the platoon profiles
```

`m3_recover_lstm_metrics.py <root> <output json>` rebuilt `metrics.json` of the penalised LSTM
after the incident E7 of `docs/decisions.md`.
