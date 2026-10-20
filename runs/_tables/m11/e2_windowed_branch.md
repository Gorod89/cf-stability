# H-N3: the branch of the decision rule (D128)

Decision rule of H-N3 (D128): confirmed when GRU and LSTM reach a stable band share >= 0.9 at a test spacing RMSE increase <= 10 % (5-fold means against E1) and the long-window check holds; partial when the share is reached at a larger increase; refuted otherwise. PERL is reported, not judged.

- gru: stable share 0.091 (< 0.9), RMSE change +17.3 % (> 10 %), long-window check fails
- lstm: stable share 0.664 (< 0.9), RMSE change +36.2 % (> 10 %), long-window check holds
- perl (not judged): stable share 0.000 (< 0.9), RMSE change +31.9 % (> 10 %), long-window check fails

Branch: the numbers fall in the branch 'refuted' (otherwise).
