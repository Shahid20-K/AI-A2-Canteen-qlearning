# Validation Notes — AI Assignment 2

## Updated implementation

- The hidden daily demand regime is sampled from probabilities `0.25/0.50/0.25` with demand multipliers `0.70/1.00/1.30`.
- The agent receives a noisy categorical forecast that is correct with configured probability `0.70`; if wrong, another category is selected uniformly.
- State contains time slot, exact inventory, observed forecast, sales-pace category, and a prior-stockout flag: `8 × 41 × 3 × 3 × 2 = 5,904` states.
- Sales pace is cumulative sales from completed slots divided by the posterior-expected cumulative demand based on the noisy forecast. Categories are below expected (<85%), on track (85%-115%), or above expected (>115%). The prior-stockout flag signals that low sales may be censored by unavailable food.
- This history summary is not a full belief state; the environment remains partially observable.
- The baseline receives the same observable forecast and history signals as Q-learning. Its target fraction/history adjustment are selected on validation episodes, separate from held-out tests.
- A small four-candidate grid selects the Q-learning learning rate and epsilon decay using two tuning seeds. Five final agents are then evaluated on the same held-out scenarios.

## Latest main evaluation (6,000 training episodes per final seed)

| Policy | Reward/day | Sold/day | Unmet demand/day | Waste/day |
|---|---:|---:|---:|---:|
| Q-learning (seed 2026) | 25.30 | 118.04 | 37.72 | 7.41 |
| Fixed 15-portion batch | -5.08 | 106.48 | 49.29 | 13.52 |
| Forecast-aware history rule | 26.60 | 121.93 | 33.84 | 10.17 |

The tuned history-aware rule has higher nominal reward. Q-learning produces less waste but sells fewer portions and has more unmet demand. Report this as a trade-off, not as an unconditional RL win.

## Demand-shift stress test (mean reward/day)

| Scenario | Q-learning | History-aware rule |
|---|---:|---:|
| 0.75x demand | 8.60 | 12.69 |
| 1.00x demand | 25.07 | 26.30 |
| 1.25x demand | 3.53 | 9.03 |

The trained policy and the rule's planning means remain nominal while environmental demand means shift.

## Reward-weight sensitivity (4,000 training episodes per seed)

| Training waste penalty | Base reward/day (SD) | Waste/day (SD) | Unmet demand/day (SD) |
|---:|---:|---:|---:|
| 0.75 | 24.77 (0.50) | 8.23 (0.37) | 37.23 (0.62) |
| 1.50 | 23.93 (1.51) | 6.55 (1.02) | 39.63 (2.05) |
| 2.25 | 20.99 (0.85) | 4.61 (0.10) | 43.57 (0.52) |

All policies are evaluated under the original reward weights. The SD is across three trained seeds, not a confidence interval. Higher waste penalties generally reduce waste but increase unmet demand.

## Validation

All 20 unit tests pass. They cover forecast accuracy/posterior inference, sales-pace categorization, state encoding/decoding, stockout-history representation, environment transitions, reward arithmetic, valid configuration, training output, and reproducible matched-seed evaluation.

From the project root:

```bash
python -m unittest discover -s tests -v
python run_experiment.py
python run_reward_sensitivity.py
python make_report.py
```

All outcomes are from the synthetic simulator, not actual campus data. The model uses a compact history summary rather than a full belief state and does not establish real-world savings.
