# Windowed-gain penalty with the maximum over the frequencies (D134, H-N3'): paired differences against the mean over the frequencies (M11) and against no penalty

- Windowed-gain penalty with the maximum over the frequencies (D134, H-N3'): paired differences against the mean over the frequencies (M11) and against no penalty. Candidate and reference per corridor, paired by scenario and seed (unit run): the builder of ablation_pairs (cf_stability/eval/corridor_tables.py) with these pairs. gru_wmax, lstm_wmax, perl_wmax: the recurrent laws trained on follownet_highd with the penalty of D134 at the weight chosen on its pilot and fine-tuned on ngsim_i80 with it (configs/queue/m12_ft_wmax.yaml); gru_windowed, lstm_windowed, perl_windowed: the same with the mean over the frequencies (M11, D128); gru, lstm, perl: no penalty (D97). PERL is reported. Descriptive, no verdict.
- Runs: 1910 of 1910 expected runs have a macro.json (I-80: 31 laws x 3 scenarios x 10 seeds + 4 laws (idm_global_p0, residual_idm_certified_p0, mlp_p0, gru_p0) x 2 scenarios (i80_p1, i80_p2) x 10 seeds; US-101: 30 laws x 3 scenarios x 10 seeds); missing runs, files and values: missing.txt.
- Analysis window (scenario.json): 180-840 s, 90-870 s.
- pair: candidate - reference by the letters of the variants, candidate and reference: their laws (ablation_pairs: [candidate, reference]; a pair with a law not in the tables has no rows, missing_corridor.txt); corridor: the corridor of the runs; metric: macro error (mean of the absolute components of the macro-error vector against the ground truth of the run's scenario, as in laws), macro error (dynamic) (of FD, wave speed, waves and wave amplitude only) and collisions / 1000 veh-km (contact episodes of followers per 1000 vehicle-km inside the window, as in laws).
- pairs: the runs (scenario and seed) of both laws with a value of the metric; mean candidate, mean reference: the means over these pairs; difference: candidate - reference per pair, mean with its interval over the pairs (negative: the candidate has the smaller error or fewer collisions); relative: mean candidate / mean reference - 1 with its interval over the pairs, only with a reference mean above 0 (no interval when resamples have a reference mean of 0); p (Wilcoxon): two-sided signed-rank test of the differences (1 when every difference is 0; not adjusted for the multiple comparisons); outcome: lower / higher when the interval of the difference lies below / above 0, else no difference.
- Intervals: 95 % percentile bootstrap, 1000 resamples, seed 0.

| corridor | pair | candidate | reference | metric | pairs | mean candidate | mean reference | difference | relative | p (Wilcoxon) | outcome |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| I-80 | gru_wmax - gru_windowed | gru_wmax | gru_windowed | macro error | 30 | 0.597 | 0.132 | 0.465 [0.395, 0.541] | +352.1 % [+284.2 %, +443.2 %] | <0.001 | higher |
| I-80 | gru_wmax - gru_windowed | gru_wmax | gru_windowed | macro error (dynamic) | 30 | 0.468 | 0.233 | 0.236 [0.170, 0.298] | +101.2 % [+65.3 %, +148.6 %] | <0.001 | higher |
| I-80 | gru_wmax - gru_windowed | gru_wmax | gru_windowed | collisions / 1000 veh-km | 30 | 6383.555 | 270.474 | 6113.081 [5490.423, 6722.843] | +2260.1 % [+1864.0 %, +2748.5 %] | <0.001 | higher |
| I-80 | gru_wmax - gru | gru_wmax | gru | macro error | 30 | 0.597 | 0.192 | 0.405 [0.344, 0.472] | +211.1 % [+172.7 %, +253.0 %] | <0.001 | higher |
| I-80 | gru_wmax - gru | gru_wmax | gru | macro error (dynamic) | 30 | 0.468 | 0.312 | 0.156 [0.086, 0.222] | +49.9 % [+24.5 %, +80.0 %] | <0.001 | higher |
| I-80 | gru_wmax - gru | gru_wmax | gru | collisions / 1000 veh-km | 30 | 6383.555 | 205.968 | 6177.586 [5572.357, 6775.493] | +2999.3 % [+2528.6 %, +3652.4 %] | <0.001 | higher |
| I-80 | lstm_wmax - lstm_windowed | lstm_wmax | lstm_windowed | macro error | 30 | 0.481 | 0.423 | 0.058 [-0.018, 0.131] | +13.7 % [-4.0 %, +33.1 %] | 0.198 | no difference |
| I-80 | lstm_wmax - lstm_windowed | lstm_wmax | lstm_windowed | macro error (dynamic) | 30 | 0.412 | 0.511 | -0.099 [-0.178, -0.030] | -19.3 % [-31.4 %, -6.5 %] | 0.019 | lower |
| I-80 | lstm_wmax - lstm_windowed | lstm_wmax | lstm_windowed | collisions / 1000 veh-km | 30 | 3386.466 | 2665.883 | 720.583 [327.993, 1148.714] | +27.0 % [+11.8 %, +45.0 %] | 0.002 | higher |
| I-80 | lstm_wmax - lstm | lstm_wmax | lstm | macro error | 30 | 0.481 | 0.716 | -0.234 [-0.314, -0.158] | -32.8 % [-41.9 %, -23.0 %] | <0.001 | lower |
| I-80 | lstm_wmax - lstm | lstm_wmax | lstm | macro error (dynamic) | 30 | 0.412 | 0.657 | -0.244 [-0.355, -0.159] | -37.2 % [-47.7 %, -26.8 %] | <0.001 | lower |
| I-80 | lstm_wmax - lstm | lstm_wmax | lstm | collisions / 1000 veh-km | 30 | 3386.466 | 4664.165 | -1277.700 [-2835.868, 74.921] | -27.4 % [-45.1 %, +2.3 %] | 0.382 | no difference |
| I-80 | perl_wmax - perl_windowed | perl_wmax | perl_windowed | macro error | 30 | 0.464 | 0.574 | -0.110 [-0.142, -0.078] | -19.2 % [-24.3 %, -13.7 %] | <0.001 | lower |
| I-80 | perl_wmax - perl_windowed | perl_wmax | perl_windowed | macro error (dynamic) | 30 | 0.471 | 0.471 | 0.000 [-0.072, 0.069] | +0.0 % [-14.1 %, +15.7 %] | 0.887 | no difference |
| I-80 | perl_wmax - perl_windowed | perl_wmax | perl_windowed | collisions / 1000 veh-km | 30 | 12583.725 | 12559.937 | 23.788 [-520.639, 737.178] | +0.2 % [-4.1 %, +5.9 %] | 0.903 | no difference |
| I-80 | perl_wmax - perl | perl_wmax | perl | macro error | 30 | 0.464 | 0.431 | 0.033 [-0.007, 0.074] | +7.7 % [-1.7 %, +18.0 %] | 0.092 | no difference |
| I-80 | perl_wmax - perl | perl_wmax | perl | macro error (dynamic) | 30 | 0.471 | 0.652 | -0.180 [-0.266, -0.104] | -27.7 % [-37.7 %, -17.2 %] | <0.001 | lower |
| I-80 | perl_wmax - perl | perl_wmax | perl | collisions / 1000 veh-km | 30 | 12583.725 | 11683.262 | 900.463 [268.607, 1687.434] | +7.7 % [+2.2 %, +14.6 %] | 0.040 | higher |
| US-101 | gru_wmax - gru_windowed | gru_wmax | gru_windowed | macro error | 30 | 1.118 | 0.260 | 0.859 [0.710, 1.023] | +331.0 % [+283.6 %, +387.8 %] | <0.001 | higher |
| US-101 | gru_wmax - gru_windowed | gru_wmax | gru_windowed | macro error (dynamic) | 30 | 0.975 | 0.481 | 0.494 [0.347, 0.651] | +102.6 % [+70.2 %, +141.4 %] | <0.001 | higher |
| US-101 | gru_wmax - gru_windowed | gru_wmax | gru_windowed | collisions / 1000 veh-km | 30 | 6462.941 | 297.655 | 6165.286 [5576.851, 6757.463] | +2071.3 % [+1679.3 %, +2581.3 %] | <0.001 | higher |
| US-101 | gru_wmax - gru | gru_wmax | gru | macro error | 30 | 1.118 | 0.421 | 0.698 [0.566, 0.846] | +165.8 % [+128.6 %, +212.5 %] | <0.001 | higher |
| US-101 | gru_wmax - gru | gru_wmax | gru | macro error (dynamic) | 30 | 0.975 | 0.517 | 0.459 [0.321, 0.584] | +88.8 % [+56.5 %, +130.5 %] | <0.001 | higher |
| US-101 | gru_wmax - gru | gru_wmax | gru | collisions / 1000 veh-km | 30 | 6462.941 | 211.229 | 6251.712 [5680.348, 6850.009] | +2959.7 % [+2350.9 %, +3589.7 %] | <0.001 | higher |
| US-101 | lstm_wmax - lstm_windowed | lstm_wmax | lstm_windowed | macro error | 30 | 0.926 | 0.679 | 0.247 [0.142, 0.370] | +36.4 % [+21.2 %, +53.6 %] | <0.001 | higher |
| US-101 | lstm_wmax - lstm_windowed | lstm_wmax | lstm_windowed | macro error (dynamic) | 30 | 0.766 | 0.654 | 0.112 [-0.052, 0.302] | +17.1 % [-7.9 %, +47.6 %] | 0.919 | no difference |
| US-101 | lstm_wmax - lstm_windowed | lstm_wmax | lstm_windowed | collisions / 1000 veh-km | 30 | 4108.284 | 2836.030 | 1272.254 [821.127, 1759.407] | +44.9 % [+27.8 %, +64.3 %] | <0.001 | higher |
| US-101 | lstm_wmax - lstm | lstm_wmax | lstm | macro error | 30 | 0.926 | 0.888 | 0.038 [-0.064, 0.142] | +4.3 % [-6.9 %, +15.8 %] | 0.516 | no difference |
| US-101 | lstm_wmax - lstm | lstm_wmax | lstm | macro error (dynamic) | 30 | 0.766 | 0.837 | -0.071 [-0.170, 0.037] | -8.5 % [-21.4 %, +4.3 %] | 0.070 | no difference |
| US-101 | lstm_wmax - lstm | lstm_wmax | lstm | collisions / 1000 veh-km | 30 | 4108.284 | 2857.501 | 1250.783 [571.486, 1915.691] | +43.8 % [+18.3 %, +76.3 %] | 0.001 | higher |
| US-101 | perl_wmax - perl_windowed | perl_wmax | perl_windowed | macro error | 30 | 0.789 | 0.904 | -0.115 [-0.205, -0.040] | -12.7 % [-21.9 %, -4.2 %] | 0.007 | lower |
| US-101 | perl_wmax - perl_windowed | perl_wmax | perl_windowed | macro error (dynamic) | 30 | 0.726 | 0.767 | -0.041 [-0.188, 0.101] | -5.3 % [-22.9 %, +13.5 %] | 0.382 | no difference |
| US-101 | perl_wmax - perl_windowed | perl_wmax | perl_windowed | collisions / 1000 veh-km | 30 | 12195.916 | 12350.048 | -154.132 [-828.666, 577.597] | -1.2 % [-6.6 %, +4.7 %] | 0.393 | no difference |
| US-101 | perl_wmax - perl | perl_wmax | perl | macro error | 30 | 0.789 | 0.661 | 0.128 [0.047, 0.235] | +19.3 % [+7.3 %, +35.0 %] | 0.005 | higher |
| US-101 | perl_wmax - perl | perl_wmax | perl | macro error (dynamic) | 30 | 0.726 | 0.685 | 0.041 [-0.086, 0.219] | +6.0 % [-12.4 %, +31.9 %] | 0.543 | no difference |
| US-101 | perl_wmax - perl | perl_wmax | perl | collisions / 1000 veh-km | 30 | 12195.916 | 13481.471 | -1285.554 [-1890.050, -605.369] | -9.5 % [-13.9 %, -4.6 %] | 0.002 | lower |
