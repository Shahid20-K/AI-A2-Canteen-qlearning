# Validation Notes — AI Assignment 2

## Changes made in this revision

1. **Stricter environment validation:** configuration checks now reject empty demand profiles/actions, invalid preparation quantities, duplicate actions, invalid demand regimes/probabilities, invalid inventory capacity, and non-finite or negative reward coefficients. Actions must be integer indices.
2. **State decoder:** added `decode_state()` so state IDs can be traced back to `(time_slot, inventory, outlook_index)` and inspected in the viva.
3. **Training/evaluation input checks:** the training config validates learning rate, discount, epsilon schedule, and episode count; evaluation validates its episode count and summary arrays.
4. **Separately tuned rule baseline:** the forecast-aware baseline's target fraction is selected from a predeclared grid using 1,000 validation days, separate from the 2,000 held-out test days. For these fixed assumptions, validation selected 100% of the outlook-adjusted mean.
5. **Corrected demand-shift comparison:** during 0.75x and 1.25x stress tests, the forecast-aware baseline retains the nominal planning means rather than reading the shifted environment means. This avoids giving the baseline advance knowledge of the distribution shift.
6. **Paired evaluation:** policies share the same nominal test demand draws, and the source saves paired reward-difference confidence intervals for each training seed.
7. **Across-seed learning curve:** the learning plot now shows 200-episode mean return bins averaged across five training seeds, with a ±1 SD band.
8. **Interpretability outputs:** generated `policy_heatmap.png` for learned actions across inventory/time/outlook and `sample_episode_trace.csv` for one matched eight-slot day.
9. **Additional tests:** expanded the suite to 13 tests, including config validation, state round trips, invalid non-integer actions, training-config checks, and keeping the rule baseline's planning model fixed under shifted demand.
10. **Report:** regenerated the two-page PDF with the tuned baseline, paired intervals, current stress-test results, the reward/service/waste trade-off, and explicit synthetic-data limitations.
11. **Reward sensitivity:** added `run_reward_sensitivity.py`, which trains agents with waste penalties 0.75, 1.50, and 2.25 using three seeds per setting. All policies are scored on the same held-out demand scenarios with the original reward weights; this makes base-reward comparisons comparable and reveals the waste-versus-stockout trade-off. It is described as a small exploratory sensitivity analysis, not a coefficient-selection procedure.
12. **Preparation-batch assumption:** documented that `{0, 5, 10, 15, 20}` is an assumed action set, with 20 additional portions the modelled per-slot batch limit (not an observed kitchen capacity); total inventory separately caps at 40.
13. **Report:** added a compact reward-sensitivity table to the two-page report.
14. **Packaging:** final ZIP should exclude `.venv`, Python bytecode caches, and nested ZIPs; the trained Q-table, source, tests, graphs, metrics, and PDF remain included.

## Current nominal held-out evaluation

2,000 matched synthetic days per policy; primary Q-learning agent trained with seed 2026.

| Policy | Reward/day | Sold/day | Unmet demand/day | Waste/day |
|---|---:|---:|---:|---:|
| Q-learning (seed 2026) | 30.43 | 120.90 | 34.58 | 7.34 |
| Fixed 15-portion batch | -5.06 | 106.41 | 49.07 | 13.59 |
| Forecast-aware tuned schedule (100% target) | 30.76 | 122.60 | 32.88 | 8.67 |

The tuned baseline slightly exceeds Q-learning on mean reward in the nominal environment. The primary-seed paired reward difference (Q-learning minus tuned rule) is -0.329 (approximate 95% CI -0.563 to -0.096). Q-learning has less waste, but also fewer sales and more unmet demand. Q-learning substantially beats the fixed-batch baseline.

Across five training seeds, Q-learning mean reward is 30.12 with standard deviation 0.80. The seed-2026 Q-learning reward has an approximate 95% confidence interval of 29.79 to 31.08.

## Demand-shift stress test

The Q-table and tuned rule's planning demand means are held at nominal values while the environment's demand means are changed.

| Scenario | Q-learning reward/day | Tuned rule reward/day |
|---|---:|---:|
| 0.75x demand | 20.86 | 16.30 |
| 1.00x demand | 30.58 | 30.96 |
| 1.25x demand | 8.75 | 11.58 |

These checks suggest the policies have different strengths under changed demand. All scenarios remain synthetic; this is not field validation.

## Reproducibility commands

```powershell
python -m unittest discover -s tests -v
python run_experiment.py
python run_reward_sensitivity.py
python make_report.py
```

The final report still needs the team member/roll-number field and GitHub repository URL filled in within `make_report.py` before submission.


## Reward sensitivity run (three training seeds per setting)

All learned policies are evaluated under the base reward coefficients (waste penalty 1.50), on the same held-out synthetic days. Values below are mean across three independently trained policies; `SD` is the standard deviation of the three seed-level means.

| Training waste penalty | Base reward/day | Waste/day | Unmet demand/day |
|---:|---:|---:|---:|
| 0.75 | 29.62 (SD 0.04) | 9.05 (SD 0.30) | 33.28 (SD 0.35) |
| 1.50 | 28.53 (SD 1.03) | 6.45 (SD 1.57) | 36.85 (SD 2.39) |
| 2.25 | 27.49 (SD 0.11) | 4.01 (SD 0.03) | 40.22 (SD 0.10) |

As the training waste penalty rises, learned policies tend to discard less food but leave more demand unmet, and their return under the common base reward decreases. This is evidence that the learned policy is sensitive to reward design; it is not evidence that any one coefficient is objectively optimal.
