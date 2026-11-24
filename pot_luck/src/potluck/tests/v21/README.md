# v21: RF vs logistic cross-table transfer

Question: does the random forest transfer across yield tables better than a
linear model? If yes, keep the RF. If not, the RF is fitting table-specific
structure and the logistic is the safer model.

Setup: general 8 (`Zn, Si, Al, S, Ca, Ti, Cr, Ni`), Gaussian element-level
noise sigma=0.2, `y_f05`. Six yield policies. Train on one full table, test on
every other table. The threshold is chosen on the training negatives at
FPR = 0.1%; the test FPR is reported because it can shift. The `tpr_test_thr`
column chooses the threshold on the test negatives, so both models are compared
at the same test FPR.

## Summary

| metric | RF | logistic |
|---|---:|---:|
| mean cross-table AUC | 0.961 | **0.964** |
| mean cross-table TPR @ FPR 0.1% | 0.375 | **0.377** |
| AUC wins (of 30 pairs) | 17 | 13 |
| TPR wins (of 30 pairs) | 15 | 15 |

The two models are effectively tied on the mean. The RF wins a few more AUC
pairs, but the TPR comparison is 15/15. There is no transfer advantage.

Within-table the RF is slightly better (mean AUC about 0.975 vs 0.970; mean TPR
about 0.45 vs 0.43). That gain does not survive the table change.

## The WW95 test

WW95 is the hardest transfer target. Logistic AUC is consistently better:

| train | RF AUC | logistic AUC | RF TPR | logistic TPR |
|---|---:|---:|---:|---:|
| Limongi18 | 0.894 | **0.955** | 0.343 | 0.360 |
| Nomoto13 CK20 blend | 0.924 | **0.957** | 0.378 | 0.285 |
| Nomoto13 CK20 clamp | 0.930 | **0.954** | 0.354 | 0.261 |
| Nomoto13 blend | 0.918 | **0.960** | 0.398 | 0.433 |
| Nomoto13 clamp | 0.927 | **0.960** | 0.361 | 0.390 |

The logistic has much better AUC on WW95, but the tail TPR is mixed: the RF
sometimes catches more at the 0.1% cut despite worse overall ranking. The AUC
gap is the clearer sign: the RF is fitting table-specific structure that does
not move to WW95.

## Conclusion

The RF does not transfer better than the logistic, but it is not worse on
average either. It is retained as the primary classifier. The logistic on the
general 8 remains the transparent baseline. The within-table gain is real but
does not transfer, and WW95 is the known weak target.

Figures: `auc_rf.png`, `auc_logit.png`, `delta_auc.png`, `tpr_rf.png`,
`tpr_logit.png`, `delta_tpr.png`.

Data: `cross_table.csv`, `cross_table.json`, `summary.json`.
