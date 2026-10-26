# Validation Notes — AI Assignment 2

## Updated implementation and assumptions

- The true daily demand regime is hidden. The agent receives a noisy categorical forecast, correct with assumed probability 0.70; if wrong, another category is selected uniformly.
- State contains time slot, exact inventory, observed forecast, cumulative sales-pace category, and prior-stockout flag: `8 × 41 × 3 × 3 × 2 = 5,904` states. This is a compact history summary, not a full Bayesian belief state.
- Sales pace is based only on completed slots and cumulative expected demand under the forecast posterior. The stockout flag helps identify when low sales may reflect unavailable inventory.
- The per-slot preparation actions are `{0, 5, 10, 15, 20}`; the 20-portion action maximum is an explicit modelling assumption. Total inventory is separately capped at 40.
- Reward weights are illustrative utility values, not measured financial costs. All demand and outcomes are synthetic; no real canteen records were used.

## Training-duration validation

The Q-learning configuration selected on the existing hyperparameter validation sweep was held fixed while comparing 6,000 and 60,000 episodes. Each duration was evaluated on the same 1,000 validation scenarios across five training seeds, with the final test seed range excluded.

| Training episodes | Mean validation reward (SD across seeds) | Waste/day | Unmet demand/day | Sold/day |
|---:|---:|---:|---:|---:|
| 6,000 | 24.10 (0.58) | 7.35 | 38.64 | 117.45 |
| 60,000 | 25.73 (0.60) | 7.48 | 37.51 | 118.58 |

The paired reward gain was 1.63 per simulated day on average across five matched training seeds (SD of paired seed-level differences 0.71). Longer training was selected for the final run. Waste rose slightly, so this is a trade-off and validation-only evidence, not proof of real-world benefit.

## Latest main held-out evaluation (60,000 training episodes per final seed)

| Policy | Reward/day | Sold/day | Unmet demand/day | Waste/day |
|---|---:|---:|---:|---:|
| Q-learning (seed 2026) | 26.93 | 119.54 | 36.23 | 7.86 |
| Fixed 15-portion batch | -5.08 | 106.48 | 49.29 | 13.52 |
| Validation-tuned forecast-aware rule | 26.60 | 121.93 | 33.84 | 10.17 |

Q-learning's primary-seed reward was 26.93 (95% CI 26.23–27.64); the tuned rule's was 26.60 (95% CI 25.81–27.38). The paired Q-learning-minus-rule difference was 0.33 (95% CI −0.01 to 0.68), which includes zero. Across five Q-learning seeds, mean test reward was 26.64 (SD 0.43). Q-learning produced less waste but also fewer sales and more unmet demand than the tuned rule. Report this as a trade-off rather than an unconditional RL win.

The validation-selected rule target was 100% and the history adjustment was 0%. Thus the implemented baseline is accurately described as forecast-aware and validation-tuned; its selected action rule did not use the optional history correction to alter preparation quantities.

## Demand-shift stress test (mean reward/day)

| Scenario | Q-learning | Forecast-aware rule |
|---|---:|---:|
| 0.75× demand | 10.83 | 12.69 |
| 1.00× demand | 26.90 | 26.30 |
| 1.25× demand | 5.10 | 9.03 |

The learned policy and rule keep nominal planning assumptions while environmental demand means shift. The stress test shows the result is sensitive to the demand distribution.

## Reward-weight sensitivity (4,000 training episodes per seed)

| Training waste penalty | Base reward/day (SD) | Waste/day (SD) | Unmet demand/day (SD) |
|---:|---:|---:|---:|
| 0.75 | 24.77 (0.50) | 8.23 (0.37) | 37.23 (0.62) |
| 1.50 | 23.93 (1.51) | 6.55 (1.02) | 39.63 (2.05) |
| 2.25 | 20.99 (0.85) | 4.61 (0.10) | 43.57 (0.52) |

All policies are evaluated under the original reward weights. SD is across three trained seeds, not a confidence interval. Higher waste penalties generally reduce waste but increase unmet demand.

## Validation and reproduction

All 20 unit tests pass. They cover noisy-forecast accuracy/posterior inference, sales-pace state features, state encoding/decoding, stockout history, environment transitions, reward arithmetic, configuration validation, training, and reproducible matched-seed evaluation.

From the project root:

```bash
python -m unittest discover -s tests -v
python run_experiment.py
python compare_training_duration.py
python run_reward_sensitivity.py
python make_report.py
```

All outcomes are from the synthetic simulator, not actual campus data. The model uses a compact history summary rather than a full belief state and does not establish real-world savings.
