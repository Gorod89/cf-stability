# Verdicts of the hypotheses

- Verdicts of the tables e1 (H1.1), e2 (H1.2), e2_lowfreq (H1.2 (combined), D110), e4 (H1.5) and e5 (H1.3); an empty verdict lacks data; complete: drawn from every run of the design.
- Bases: H1.1 unstable among equilibria and the RMSE against the reference; H1.2 and H1.2 (combined) band share not stable with penalty and the RMSE change; H1.3 the reduction of the growth error on the collision-free prefix of the platoons (D109; open with fewer than 3 pairs of runs that both have one); H1.5 band share unstable. information: the verdict of H1.1 on band share not stable.

| hypothesis | unit | verdict | complete | basis | information |
|---|---|---|---|---|---|
| H1.1 | mlp | holds | yes | unstable among equilibria 0.94 [0.91, 0.97], RMSE vs idm -7.7 % [-8.9 %, -6.5 %] | band share not stable 0.94 [0.91, 0.97]: holds |
| H1.1 | gru | holds | yes | unstable among equilibria 0.86 [0.80, 0.91], RMSE vs idm -32.6 % [-33.7 %, -31.4 %] | band share not stable 0.89 [0.84, 0.93]: holds |
| H1.1 | lstm | holds | yes | unstable among equilibria 0.92 [0.89, 0.96], RMSE vs idm -29.0 % [-30.0 %, -28.0 %] | band share not stable 0.95 [0.93, 0.97]: holds |
| H1.1 | overall | holds | yes | holds for at least 2 of 3 | band share not stable: holds |
| H1.2 | mlp | confirmed | yes | band share not stable 0.00 [0.00, 0.00], RMSE change -0.5 % [-0.9 %, -0.1 %] |  |
| H1.2 | pidl | confirmed | yes | band share not stable 0.00 [0.00, 0.00], RMSE change -0.3 % [-0.7 %, +0.0 %] |  |
| H1.2 | gru | refuted | yes | band share not stable 0.66 [0.55, 0.76], RMSE change +14.7 % [+13.2 %, +15.9 %] |  |
| H1.2 | lstm | refuted | yes | band share not stable 0.39 [0.31, 0.48], RMSE change +25.0 % [+23.7 %, +26.3 %] |  |
| H1.2 | perl | refuted | yes | band share not stable 1.00 [0.99, 1.00], RMSE change +26.9 % [+25.5 %, +28.3 %] |  |
| H1.2 | residual_idm | confirmed | yes | band share not stable 0.00 [0.00, 0.01], RMSE change +1.5 % [+1.1 %, +2.0 %] |  |
| H1.2 (combined) | gru | open | yes | weight 0.1 (no weight reaches stable >= 0.9: largest stable share): band share not stable 0.45 [0.25, 0.65], RMSE change +17.3 % [+15.1 %, +19.3 %] against E1 (seed 0) |  |
| H1.2 (combined) | lstm | refuted | yes | weight 0.1 (stable >= 0.9: smallest validation RMSE): band share not stable 0.50 [0.24, 0.76], RMSE change +29.0 % [+27.1 %, +30.6 %] against E1 (seed 0) |  |
| H1.2 (combined) | perl | refuted | yes | weight 1 (no weight reaches stable >= 0.9: largest stable share): band share not stable 0.97 [0.92, 1.00], RMSE change +28.3 % [+26.2 %, +30.2 %] against E1 (seed 0) |  |
| H1.5 | overall | holds | yes | certified, fine-tuned: band share unstable 0.00 [0.00, 0.00]; fine-tuned MLP: 0.98 [0.96, 0.99] |  |
| H1.3 | openacc_acc/mlp | refuted | yes | reduction -2.6 % [-29.1 %, +14.9 %] over 5 pairs (collision-free prefix); whole curves (D107) -30.9 % [-95.4 %, +1.4 %] over 2 pairs; collided profiles 7/10 without, 0/10 with penalty; first collided position 36.5 without, 51.0 with penalty |  |
| H1.3 | openacc_acc/gru | open | yes | reduction -2.2 % [-2.2 %, -2.2 %] over 1 pairs (collision-free prefix); whole curves (D107) n/a over 0 pairs; collided profiles 8/10 without, 10/10 with penalty; first collided position 14.2 without, 24.4 with penalty; open: fewer than 3 pairs |  |
| H1.3 | openacc_acc/lstm | open | yes | reduction +52.5 % [+52.5 %, +52.5 %] over 1 pairs (collision-free prefix); whole curves (D107) n/a over 0 pairs; collided profiles 8/10 without, 9/10 with penalty; first collided position 17.4 without, 23.7 with penalty; open: fewer than 3 pairs |  |
| H1.3 | openacc_acc/pooled | open | yes | reduction +12.7 % [-14.7 %, +34.3 %] over 7 pairs (collision-free prefix); whole curves (D107) -30.9 % [-95.4 %, +1.4 %] over 2 pairs; collided profiles 23/30 without, 19/30 with penalty |  |
| H1.3 | openacc_human/mlp | refuted | yes | reduction -48.2 % [-123.9 %, +3.2 %] over 5 pairs (collision-free prefix); whole curves (D107) n/a over 0 pairs; collided profiles 11/15 without, 13/15 with penalty; first collided position 18.0 without, 13.5 with penalty |  |
| H1.3 | openacc_human/gru | confirmed | yes | reduction +48.5 % [+11.5 %, +76.5 %] over 5 pairs (collision-free prefix); whole curves (D107) +29.9 % [+29.9 %, +29.9 %] over 1 pairs; collided profiles 13/15 without, 12/15 with penalty; first collided position 13.1 without, 13.9 with penalty |  |
| H1.3 | openacc_human/lstm | open | yes | reduction +38.4 % [-61.6 %, +68.1 %] over 5 pairs (collision-free prefix); whole curves (D107) -46.4 % [-46.4 %, -46.4 %] over 1 pairs; collided profiles 14/15 without, 13/15 with penalty; first collided position 9.9 without, 13.2 with penalty |  |
| H1.3 | openacc_human/pooled | open | yes | reduction +12.1 % [-31.1 %, +44.1 %] over 15 pairs (collision-free prefix); whole curves (D107) +16.3 % [-46.4 %, +29.9 %] over 2 pairs; collided profiles 38/45 without, 38/45 with penalty |  |
