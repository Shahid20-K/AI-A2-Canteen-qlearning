# Validation Notes — AI Assignment 2

## Current implementation

- The simulator samples a hidden daily demand regime from prior probabilities `0.25/0.50/0.25` with multipliers `0.70/1.00/1.30`.
- The agent sees a noisy categorical forecast that matches the hidden regime with probability `0.70`; if wrong, one of the other two categories is selected uniformly. The true regime is used to generate Poisson demand but is not part of the agent's state.
- The observation state remains `(time slot, exact inventory, observed forecast)` with 984 states and five preparation actions. The tabular Q-learning policy is memoryless over this observation and does not maintain a belief state; this partial-observability limitation is disclosed.
- The forecast-aware baseline receives the same observed forecast and uses the posterior expected regime multiplier computed by Bayes' rule. Its target fraction is selected from a predeclared grid on a separate validation set, not the held-out test episodes.
- Main evaluation uses 2,000 matched held-out simulated days per policy, five independent Q-learning training seeds, paired difference intervals, and 0.75x/1.25x demand-shift stress tests.
- Reward sensitivity trains three agents per waste penalty and scores all learned policies using the same original reward weights. This is an exploratory diagnostic, not a claim that one coefficient is objectively optimal.

## Latest nominal held-out evaluation

| Policy | Reward/day | Sold/day | Unmet demand/day | Waste/day |
|---|---:|---:|---:|---:|
| Q-learning (seed 2026) | 25.66 | 117.68 | 38.09 | 6.89 |
| Fixed 15-portion batch | -5.08 | 106.48 | 49.29 | 13.52 |
| Forecast-aware tuned schedule | 26.60 | 121.93 | 33.84 | 10.17 |

The tuned forecast-aware baseline has higher nominal mean reward than Q-learning. Q-learning has less waste, but serves fewer portions and has higher unmet demand. The comparison must be described as a trade-off rather than an unconditional RL win. Values are from the synthetic simulator.

## Demand-shift stress test

| Scenario | Q-learning reward/day | Tuned rule reward/day |
|---|---:|---:|
| 0.75x demand | 21.43 | 12.69 |
| 1.00x demand | 25.26 | 26.30 |
| 1.25x demand | 3.15 | 9.03 |

The true regime remains generated using the configured prior, while the slot demand means are shifted. The learned policy and baseline planning means are not given advance knowledge of the shift.

## Reward-weight sensitivity

| Training waste penalty | Base reward/day (SD) | Waste/day (SD) | Unmet demand/day (SD) |
|---:|---:|---:|---:|
| 0.75 | 25.71 (0.38) | 11.74 (0.45) | 32.72 (0.43) |
| 1.50 | 26.07 (0.65) | 8.11 (0.65) | 36.57 (1.11) |
| 2.25 | 23.52 (0.59) | 5.98 (0.65) | 40.50 (0.98) |

Values are means and standard deviations across three independent training seeds, evaluated under common base reward weights. Greater waste penalties generally reduce waste but increase unmet demand.

## Tests and reproducibility

All 17 unit tests pass in the current working version. The tests cover environment transitions, state encoding/decoding, reward arithmetic, configuration validation, forecast accuracy and posterior interpretation, training output, matched-seed policy evaluation, and reproducibility.

Run from the project root:

```powershell
python -m unittest discover -s tests -v
python run_experiment.py
python run_reward_sensitivity.py
python make_report.py
```

The report is two pages and uses synthetic simulation results. It does not establish real-world savings. The current tabular observation does not track belief updates from previous sales; this is a limitation and reasonable next extension.
