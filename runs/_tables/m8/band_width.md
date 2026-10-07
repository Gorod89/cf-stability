# Relative width of the spacing band and per-driver dispersion (D123)

- Spacing band of the data (D78, D123 part a): per grid speed the 5 %, 50 % and 95 % quantiles of the spacing of the near-steady samples of all events of the set (|dv| < 0.5 m/s, |a| < 0.3 m/s², speed within 0.5 m/s of the grid speed; cf_stability.train.tensors.spacing_band, dv = v - v_lead); a speed with fewer than 200 such samples has no band. relative width = (s95 - s5) / s50.
- Per-driver dispersion: the standard deviation (ddof 1) over the drivers with at least 20 near-steady samples at the speed of the driver's median near-steady spacing there, in m and relative to s50 (none with fewer than two such drivers). Drivers are follower ids; follownet_highd has none, every event (15 s) is its own driver (D19).

| data | v (m/s) | near-steady samples | s 5 % (m) | s 50 % (m) | s 95 % (m) | relative width | drivers (>= 20 samples) | driver std (m) | driver std / s 50 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| follownet_highd | 5 | 3090 | 4.76 | 9.91 | 21.77 | 1.716 | 62 | 5.44 | 0.548 |
| follownet_highd | 6 | 4568 | 5.42 | 11.46 | 23.05 | 1.538 | 78 | 6.41 | 0.560 |
| follownet_highd | 7 | 5589 | 6.11 | 12.38 | 33.23 | 2.190 | 114 | 8.83 | 0.713 |
| follownet_highd | 8 | 4973 | 6.62 | 13.94 | 27.36 | 1.487 | 92 | 6.51 | 0.467 |
| follownet_highd | 9 | 5737 | 7.46 | 15.87 | 39.20 | 2.000 | 111 | 9.37 | 0.590 |
| follownet_highd | 10 | 7342 | 8.37 | 17.66 | 35.62 | 1.543 | 133 | 9.48 | 0.536 |
| follownet_highd | 11 | 8777 | 8.54 | 17.38 | 40.84 | 1.858 | 168 | 11.56 | 0.665 |
| follownet_highd | 12 | 9185 | 8.57 | 18.94 | 43.64 | 1.852 | 173 | 11.48 | 0.606 |
| follownet_highd | 13 | 7662 | 9.41 | 21.53 | 45.32 | 1.668 | 151 | 11.69 | 0.543 |
| follownet_highd | 14 | 8250 | 9.22 | 21.91 | 53.09 | 2.002 | 169 | 14.15 | 0.646 |
| follownet_highd | 15 | 8392 | 9.97 | 21.79 | 51.42 | 1.902 | 170 | 13.14 | 0.603 |
| follownet_highd | 16 | 7935 | 10.20 | 22.12 | 49.78 | 1.790 | 141 | 13.41 | 0.606 |
| follownet_highd | 17 | 8260 | 10.24 | 22.22 | 47.26 | 1.666 | 152 | 11.76 | 0.529 |
| follownet_highd | 18 | 14580 | 11.64 | 26.19 | 77.25 | 2.505 | 259 | 19.36 | 0.739 |
| follownet_highd | 19 | 16349 | 11.68 | 26.31 | 63.15 | 1.956 | 302 | 17.40 | 0.662 |
| follownet_highd | 20 | 25911 | 12.50 | 28.74 | 62.56 | 1.742 | 483 | 16.14 | 0.562 |
| follownet_highd | 21 | 51293 | 12.94 | 29.80 | 61.20 | 1.620 | 876 | 15.27 | 0.512 |
| follownet_highd | 22 | 115163 | 13.49 | 30.69 | 58.07 | 1.453 | 1883 | 13.91 | 0.453 |
| follownet_highd | 23 | 175314 | 13.63 | 29.51 | 52.20 | 1.307 | 2782 | 11.89 | 0.403 |
| follownet_highd | 24 | 150589 | 12.37 | 25.62 | 43.48 | 1.214 | 2554 | 9.84 | 0.384 |
| follownet_highd | 25 | 83881 | 10.91 | 21.64 | 37.13 | 1.212 | 1470 | 8.30 | 0.384 |
| follownet_highd | 26 | 22046 | 10.18 | 17.99 | 31.13 | 1.165 | 420 | 6.16 | 0.343 |
| follownet_highd | 27 | 3137 | 10.76 | 17.77 | 30.86 | 1.131 | 59 | 5.91 | 0.333 |
| follownet_highd | 28 | 287 | 9.06 | 21.14 | 33.08 | 1.136 | 4 | 4.61 | 0.218 |
| follownet_highd | 29 | 356 | 9.23 | 20.61 | 45.40 | 1.755 | 4 | 9.88 | 0.479 |
| follownet_highd | 30 | 121 |  |  |  |  | 2 | 5.73 |  |
| ngsim_i80 | 5 | 77496 | 3.58 | 8.63 | 17.23 | 1.582 | 1404 | 4.35 | 0.504 |
| ngsim_i80 | 6 | 89785 | 4.55 | 10.01 | 19.89 | 1.533 | 1635 | 4.95 | 0.494 |
| ngsim_i80 | 7 | 35518 | 5.01 | 11.32 | 23.20 | 1.607 | 552 | 5.75 | 0.507 |
| ngsim_i80 | 8 | 55459 | 5.69 | 12.39 | 25.91 | 1.632 | 1035 | 6.04 | 0.487 |
| ngsim_i80 | 9 | 42397 | 6.63 | 13.68 | 28.89 | 1.627 | 789 | 7.06 | 0.516 |
| ngsim_i80 | 10 | 11187 | 6.98 | 15.09 | 31.67 | 1.636 | 115 | 6.62 | 0.439 |
| ngsim_i80 | 11 | 15091 | 7.98 | 16.12 | 35.41 | 1.701 | 243 | 8.32 | 0.516 |
| ngsim_i80 | 12 | 10327 | 9.10 | 18.80 | 37.62 | 1.516 | 161 | 8.39 | 0.446 |
| ngsim_i80 | 13 | 2823 | 8.79 | 20.90 | 38.97 | 1.444 | 19 | 7.25 | 0.347 |
| ngsim_i80 | 14 | 4067 | 10.24 | 22.51 | 44.89 | 1.539 | 45 | 8.16 | 0.363 |
| ngsim_i80 | 15 | 3183 | 11.54 | 23.58 | 56.33 | 1.900 | 30 | 10.43 | 0.442 |
| ngsim_i80 | 16 | 1640 | 11.80 | 24.98 | 59.68 | 1.917 | 11 | 7.59 | 0.304 |
| ngsim_i80 | 17 | 2040 | 12.14 | 27.58 | 64.12 | 1.884 | 23 | 13.62 | 0.494 |
| ngsim_i80 | 18 | 1017 | 13.80 | 28.99 | 68.40 | 1.883 | 7 | 17.31 | 0.597 |
| ngsim_i80 | 19 | 418 | 15.96 | 31.74 | 53.96 | 1.197 | 3 | 2.66 | 0.084 |
| ngsim_i80 | 20 | 426 | 18.05 | 32.26 | 69.72 | 1.602 | 5 | 21.00 | 0.651 |
| ngsim_i80 | 21 | 282 | 16.50 | 30.44 | 79.20 | 2.060 | 3 | 8.13 | 0.267 |
| ngsim_i80 | 22 | 152 |  |  |  |  | 0 |  |  |
| ngsim_i80 | 23 | 153 |  |  |  |  | 0 |  |  |
| ngsim_i80 | 24 | 75 |  |  |  |  | 0 |  |  |
| ngsim_i80 | 25 | 8 |  |  |  |  | 0 |  |  |
| ngsim_i80 | 26 | 3 |  |  |  |  | 0 |  |  |
| ngsim_i80 | 27 | 0 |  |  |  |  | 0 |  |  |
| ngsim_i80 | 28 | 0 |  |  |  |  | 0 |  |  |
| ngsim_i80 | 29 | 0 |  |  |  |  | 0 |  |  |
| ngsim_i80 | 30 | 0 |  |  |  |  | 0 |  |  |
