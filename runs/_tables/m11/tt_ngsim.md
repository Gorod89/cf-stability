# Two-timescale certified hybrid fine-tuned on NGSIM I-80 (D129)

- e4_tt_ft: e4_tt fine-tuned on ngsim_i80 (init_from, the checkpoint as it is: core, scalers, certified budget; configs/queue/e4_tt.yaml), five folds, seed 0. Reference: the certified hybrid fine-tuned the same way (runs/e4_stable_ft/ngsim_i80/residual_idm, seed 0). The certificate is that of scripts/certificate.py after the fine-tuning (scanned a priori certificate at the 21 frozen latents and on the reachable range); the exact one at the 21 latents: table certificate_tt.
- Stability columns as in tt_pilot.

| fold | val RMSE s (m) | test RMSE s (m) | E4 fine-tuned test RMSE (m) | ratio to E4 | epochs | unstable eq. z=-1 | unstable eq. z=-0.5 | unstable eq. z=0 | unstable eq. z=0.5 | unstable eq. z=1 | band stable z=-1 | band stable z=-0.5 | band stable z=0 | band stable z=0.5 | band stable z=1 | stable at every frozen z | unstable eq. (online) | max gain (online) | platoon growth error | certificate z = 0 | certificate: z of 21 | certificate at all 21 z | certificate on the reachable range | min guaranteed margin, reachable (s⁻²) | z failing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|---:|---|
| 0 | 4.830 | 4.810 | 4.916 | 0.978 | 19 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | yes | 0.000 | 1.005 | 0.172 | yes | 13 |  | yes | 0.0055 | -1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 |
| 1 | 4.793 | 4.818 | 4.931 | 0.977 | 26 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | yes | 0.000 | 1.000 | 0.175 | yes | 13 |  | yes | 0.0057 | -1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 |
| 2 | 4.758 | 4.797 | 4.892 | 0.981 | 26 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | yes | 0.000 | 1.008 | 0.175 | yes | 13 |  | yes | 0.0057 | -1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 |
| 3 | 4.748 | 4.760 | 4.849 | 0.981 | 33 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | yes | 0.000 | 1.000 | 0.175 | yes | 13 |  | yes | 0.0058 | -1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 |
| 4 | 4.800 | 4.746 | 4.845 | 0.979 | 34 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | yes | 0.000 | 1.000 | 0.171 | yes | 13 |  | yes | 0.0057 | -1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 |
| mean | 4.786 | 4.786 | 4.887 | 0.979 | 27 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | yes | 0.000 | 1.003 | 0.174 | yes | 13 |  | yes | 0.0057 |  |
