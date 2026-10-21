# H-N3': the branch of the decision rule (D134)

Decision rule of H-N3' (D134): confirmed when GRU and LSTM reach a stable band share >= 0.9 at a test spacing RMSE increase <= 10 % (5-fold means against E1; the penalty with the maximum over the frequencies at the weights chosen on its pilot) and the long-window check holds; partial when the share is reached at a larger increase; refuted otherwise. PERL is reported, not judged.

- gru: stable share 1.000 (>= 0.9), RMSE change +46.1 % (> 10 %), long-window check holds
- lstm: stable share 0.991 (>= 0.9), RMSE change +29.6 % (> 10 %), long-window check holds
- perl (not judged): stable share 0.036 (< 0.9), RMSE change +34.4 % (> 10 %), long-window check fails

Branch: the numbers fall in the branch 'partial' (the share is reached at a larger increase).
