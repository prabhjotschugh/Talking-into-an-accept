# Results Summary

Auto-generated from analyze_results.py. Verify every number against the workbook before quoting it in the paper.

## Data quality

- Total logged calls: 16200
- Errored calls: 2

## Baseline reliability

- Krippendorff's alpha (interval, across 9 models, original abstracts): 0.105
- Baseline decision agreement with real venue verdict: 0.336

## Headline result: Verdict Switch Rate

- **hedged**: 2/53 correctly-identified rejections flipped to accept (3.8%)
- **assertive**: 9/53 correctly-identified rejections flipped to accept (17.0%)
- **overclaiming**: 2/53 correctly-identified rejections flipped to accept (3.8%)
- **native_fluent**: 8/53 correctly-identified rejections flipped to accept (15.1%)
- **l2_grounded**: 9/53 correctly-identified rejections flipped to accept (17.0%)

## Manipulation check

- hedged_vs_original_hedge_density: n=100, wilcoxon p=3.895e-18
- overclaiming_vs_original_hype_density: n=100, wilcoxon p=3.89e-18

## Content invariance (TF-IDF cosine similarity to original)

                 mean     std     min     max
tone                                         
assertive      0.9458  0.0456  0.7236  0.9936
hedged         0.8995  0.0562  0.5207  0.9643
l2_grounded    0.9441  0.0351  0.7628  0.9864
native_fluent  0.9175  0.0707  0.7075  1.0000
overclaiming   0.9423  0.0484  0.6329  0.9824

## McNemar's exact test (reject tier, decision-level)

See mcnemar_results sheet in the workbook for the full table, per model, per tone.

## Paired score tests (Wilcoxon, pooled across models)

See score_tests sheet in the workbook. Includes Cohen's d and observed power per tone.

## Mixed-effects model

Fallback used: False

```
                                  Mixed Linear Model Regression Results
==========================================================================================================
Model:                              MixedLM                 Dependent Variable:                 mean_score
No. Observations:                   5400                    Method:                             REML      
No. Groups:                         100                     Scale:                              0.3112    
Min. group size:                    54                      Log-Likelihood:                     -5800.3251
Max. group size:                    54                      Converged:                          Yes       
Mean group size:                    54.0                                                                  
----------------------------------------------------------------------------------------------------------
                                                               Coef.  Std.Err.    z    P>|z| [0.025 0.975]
----------------------------------------------------------------------------------------------------------
Intercept                                                       7.798    0.060 130.854 0.000  7.682  7.915
C(text_type, Treatment(reference='original'))[T.hedged]        -0.367    0.026 -13.957 0.000 -0.419 -0.315
C(text_type, Treatment(reference='original'))[T.assertive]      0.091    0.026   3.479 0.001  0.040  0.143
C(text_type, Treatment(reference='original'))[T.overclaiming]  -0.265    0.026 -10.063 0.000 -0.316 -0.213
C(text_type, Treatment(reference='original'))[T.native_fluent]  0.058    0.026   2.211 0.027  0.007  0.110
C(text_type, Treatment(reference='original'))[T.l2_grounded]   -0.013    0.026  -0.493 0.622 -0.065  0.039
C(quality_tier)[T.reject_tier]                                  0.005    0.068   0.073 0.941 -0.129  0.139
model_name Var                                                  0.829    0.082                            
==========================================================================================================

```

