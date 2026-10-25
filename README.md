# Adaptive Meal Preparation in a Campus Canteen

A custom tabular Q-learning project exploring how a simulated campus canteen can balance meal availability, preparation cost, unmet demand, and discarded food.

> **Evidence boundary:** this project uses a custom synthetic simulator. The demand means, demand-outlook probabilities, carryover rule, and reward coefficients are illustrative assumptions, not measured data from IIT Jodhpur or any real canteen. Results demonstrate behavior only within this simulator.

## Project contents

```text
canteen_rl/
  environment.py       Canteen simulator, transition rules, state encoder/decoder
  agent.py             Tabular Q-learning, baselines, training and evaluation helpers
tests/
  test_environment.py  Environment transition, reward, configuration and state tests
  test_agent.py        Agent, reproducibility and baseline tests
results/
  metrics.json                  Metrics, confidence intervals, tuning and stress tests
  q_table.npy                   Saved Q-table for the seed-2026 agent
  training_returns.npy          Training return series for seed 2026
  training_returns_by_seed.npy  Training return series for all five seeds
  learning_curve.png            Learning curve mean and ±1 SD across five seeds
  policy_comparison.png         Test metrics across three policies
  policy_heatmap.png            Chosen action by inventory, time slot and outlook
  sample_episode_trace.csv      Eight-slot Q-learning vs baseline trace on matched demand
  reward_sensitivity.json       Reward-penalty sensitivity metrics on a common test score
  reward_sensitivity.png        Trade-off between waste penalty, waste and unmet demand
run_experiment.py      Validation, training, evaluation, stress tests and plots
run_reward_sensitivity.py  Reward-weight sensitivity experiment (3 penalties x 3 training seeds)
make_report.py         Two-page PDF report generator
AI_A2_Report.pdf       Generated two-page report with team and repository details
requirements.txt       Python dependencies
SUBMISSION_CHECKLIST.md
VALIDATION_NOTES.md
```

## Environment setup (Windows PowerShell / VS Code)

Use Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, open a Command Prompt terminal instead:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run tests and reproduce the experiment

Run from the project root (the folder containing `run_experiment.py`):

```powershell
python -m unittest discover -s tests -v
python run_experiment.py
python run_reward_sensitivity.py
python make_report.py
```

The test suite currently contains **13 tests**. The main experiment trains five Q-learning agents (seeds 2026-2030) for 20,000 episodes each. The optional sensitivity experiment trains nine additional agents (three waste-penalty settings × three seeds) for 20,000 episodes each; it is a targeted diagnostic and may take longer than the unit tests.

## RL formulation

- **State:** `(time_slot, exact_inventory, demand_outlook)`. There are 8 time slots, 41 inventory levels (0-40 portions), and 3 outlook categories: `8 × 41 × 3 = 984` states.
- **Action:** request an additional preparation batch from `{0, 5, 10, 15, 20}` portions. The maximum 20-portion batch per time slot is an explicit modelling assumption, not a measured kitchen limit. Actual production is also clipped by the separate 40-portion total-inventory capacity.
- **Demand:** at the start of a simulated day, a low/normal/high regime is sampled using probabilities `0.25/0.50/0.25` and multipliers `0.70/1.00/1.30`. The observed outlook identifies the demand regime in this deliberately optimistic simulator. Slot demand is independently sampled from a Poisson distribution around the adjusted mean.
- **Carryover:** 30% of leftover portions, rounded down, carries to the next slot; the rest is discarded. Nothing carries after the final slot.
- **Reward:** `sold - 0.35 × prepared - 1.00 × unmet_demand - 1.50 × discarded_waste`. These are illustrative utility weights, not measured currency costs.
- **Algorithm:** tabular Q-learning with epsilon-greedy exploration, learning rate `0.12`, discount factor `0.92`, initial epsilon `1.0`, minimum epsilon `0.05`, and multiplicative decay `0.9992`. Terminal transitions do not bootstrap from a future Q-value.

## Validation methodology

1. **Baseline tuning is separated from the final test.** The forecast-aware rule selects its target fraction from `50%, 60%, 70%, 80%, 90%, 100%, 110%, 120%` of the outlook-adjusted demand mean on a separate 1,000-day validation set (seed range beginning at 300,000). The best validation target is 100% for the current configuration.
2. **Held-out comparison.** The three policies are evaluated on the same 2,000 synthetic test days (seed range beginning at 500,000): Q-learning, a fixed 15-portion batch, and the validation-tuned forecast-aware rule. Matched per-episode demand draws support paired difference intervals.
3. **Training-seed robustness.** Five independent Q-learning agents are evaluated on the same held-out episodes; the learning curve shows the mean and ±1 standard deviation across seeds.
4. **Demand-shift stress test.** The nominally trained Q-table and the tuned rule (whose planning means remain nominal) are tested when the environment's demand means are scaled to `0.75×` and `1.25×`. This does not give either policy the newly shifted means.
5. **Uncertainty reporting.** The metrics JSON reports mean, standard deviation, approximate normal 95% confidence intervals, and matched per-episode reward differences. The intervals quantify simulator variability, not uncertainty about real-world impact.

## Current reference results

These are generated results from the included simulator and fixed random seeds. Re-run `python run_experiment.py` to reproduce them.

| Policy | Reward/day | Sold/day | Unmet demand/day | Waste/day |
|---|---:|---:|---:|---:|
| Q-learning (seed 2026) | 30.43 | 120.90 | 34.58 | 7.34 |
| Fixed 15-portion batch | -5.06 | 106.41 | 49.07 | 13.59 |
| Forecast-aware tuned schedule (100% target) | 30.76 | 122.60 | 32.88 | 8.67 |

On the nominal test set, the Q-learning agent clearly improves on the fixed-batch baseline. However, the validation-tuned forecast-aware rule has a slightly higher mean reward (30.76 versus 30.43); the primary-seed paired reward difference for Q-learning minus that rule is `-0.33` with an approximate 95% CI of `[-0.56, -0.10]`. Q-learning discards fewer portions than the tuned rule (7.34 versus 8.67) but serves fewer portions and has more unmet demand. This trade-off is a result, not something to conceal.

Under the demand-shift tests, mean rewards for Q-learning vs the nominal-planning tuned rule are `20.86 vs 16.30` at `0.75×` demand and `8.75 vs 11.58` at `1.25×` demand. Thus neither policy is best across every operating condition.

## Reward-weight sensitivity analysis

Run `python run_reward_sensitivity.py` after the main experiment. It trains Q-learning policies with waste penalties `0.75`, `1.50` (the main setting), and `2.25`, using three independent training seeds per setting. Every resulting policy is evaluated on the same 2,000 held-out demand scenarios **under the original reward weights** (including waste penalty 1.50), so reward values are comparable across rows. The operational differences are reported as means and standard deviations across three independently trained policies; this small sweep is exploratory, not a proof of statistical superiority and not a post-hoc method for selecting a final model.

In the current run, a larger waste penalty generally reduces discarded portions while increasing unmet demand. This illustrates why reward coefficients matter: they encode priorities, and a policy cannot be judged from reward alone. See `results/reward_sensitivity.json` and `results/reward_sensitivity.png`.

## Output interpretation

- `results/metrics.json`: all primary metrics, validation candidates, five-seed summaries, paired comparisons and demand-shift tests.
- `results/policy_heatmap.png`: visual summary of the learned greedy actions across the state dimensions.
- `results/sample_episode_trace.csv`: readable per-slot side-by-side decisions for one matched synthetic day. It is illustrative, not statistically representative.

## Conclusion

This project explores the use of tabular Q-learning to balance food availability and waste in a simulated campus canteen. It compares the learned policy with fixed-batch and forecast-aware preparation strategies and evaluates the effects of changing demand and reward-function parameters.

The experiments highlight a trade-off between reducing food waste and meeting student demand. Q-learning reduces waste compared with the tuned forecast-aware strategy, while the latter achieves a slightly higher reward and fewer unmet orders under nominal simulated conditions.

All results are based on synthetic demand and the assumptions defined in the simulation. They should not be interpreted as evidence of actual canteen savings. Future work could incorporate historical demand data, imperfect forecasts, food freshness, and realistic preparation constraints.