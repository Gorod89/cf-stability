# Residual budget allocation for the margin core of E (D130)

- Budgets (B_s, B_dv, B_v) of the residual's Jacobian bounds |dr/d(s, dv, v)| that maximise B_s + B_dv + B_v for the IDM core of every fold member of residual_idm_certified (E), by coordinate ascent with bisection (cf_stability.stability.robust.allocate_budget), reported, not trained. Criterion of D130: a verified certificate at gamma = 1 + 1e-3 on the band of the member's training context, box or hull (as in robust_certificate), for every speed (per speed) or over the whole band; feasible false: the core alone has none, every budget is 0.
- Supplement: the same ascent under the pointwise criterion of the a priori certificate on I(v) (r_max 0.3; certificate.guaranteed_margin >= 0 and the guaranteed signs at 400 spacings of every speed's I(v), without a common storage function).
- uniform: the member's own bounds at r_max 0.3 (r_max lipschitz prod(layer norms) / (20, 2, 10), the physical scaler of ResidualIDM); uniform holds: the criterion holds with them; ratio: the total of the allocation over the uniform total.

| member | polytope | scope | feasible | B_s | B_dv | B_v | total | uniform total | uniform holds | ratio |
|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|
| 0 | box | whole |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0653 |  | 0.00 |
| 0 | box | per speed |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0653 |  | 0.00 |
| 0 | hull | whole |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0653 |  | 0.00 |
| 0 | hull | per speed |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0653 |  | 0.00 |
| 1 | box | whole |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0660 |  | 0.00 |
| 1 | box | per speed |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0660 |  | 0.00 |
| 1 | hull | whole |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0660 |  | 0.00 |
| 1 | hull | per speed |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0660 |  | 0.00 |
| 2 | box | whole |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0662 |  | 0.00 |
| 2 | box | per speed |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0662 |  | 0.00 |
| 2 | hull | whole |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0662 |  | 0.00 |
| 2 | hull | per speed |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0662 |  | 0.00 |
| 3 | box | whole |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0667 |  | 0.00 |
| 3 | box | per speed |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0667 |  | 0.00 |
| 3 | hull | whole |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0667 |  | 0.00 |
| 3 | hull | per speed |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0667 |  | 0.00 |
| 4 | box | whole |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0661 |  | 0.00 |
| 4 | box | per speed |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0661 |  | 0.00 |
| 4 | hull | whole |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0661 |  | 0.00 |
| 4 | hull | per speed |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0661 |  | 0.00 |
| 0 | pointwise (a priori criterion) | I(v), every spacing | yes | 0.0144 | 0.0743 | 0.0000 | 0.0887 | 0.0653 | yes | 1.36 |
| 1 | pointwise (a priori criterion) | I(v), every spacing | yes | 0.0141 | 0.0763 | 0.0000 | 0.0904 | 0.0660 | yes | 1.37 |
| 2 | pointwise (a priori criterion) | I(v), every spacing | yes | 0.0142 | 0.0764 | 0.0000 | 0.0905 | 0.0662 | yes | 1.37 |
| 3 | pointwise (a priori criterion) | I(v), every spacing | yes | 0.0142 | 0.0770 | 0.0000 | 0.0912 | 0.0667 | yes | 1.37 |
| 4 | pointwise (a priori criterion) | I(v), every spacing | yes | 0.0149 | 0.0738 | 0.0000 | 0.0886 | 0.0661 | yes | 1.34 |
