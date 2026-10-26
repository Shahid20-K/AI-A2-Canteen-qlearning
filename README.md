# Adaptive Meal Preparation in a Campus Canteen

A custom tabular Q-learning project exploring how a simulated campus canteen can balance meal availability, preparation cost, unmet demand, and discarded food.

> **Evidence boundary:** this project uses a custom synthetic simulator. Demand means, outlook probabilities, forecast accuracy, carryover, and reward coefficients are illustrative assumptions, not measurements from IIT Jodhpur or a real canteen. The results do not establish actual savings.

## Project contents

```text
canteen_rl/
  environment.py       Canteen simulator, state/history features, transition rules
  agent.py             Tabular Q-learning, history-aware baseline, training/evaluation helpers
tests/
  test_environment.py  Environment, state encoding, forecast, transition and reward tests
  test_agent.py        Training, evaluation, baseline and reproducibility tests
results/
  metrics.json                  Validation tuning, test metrics, paired comparisons and stress tests
  q_table.npy                   Saved Q-table for the seed-2026 agent
  training_returns.npy          Training returns for seed 2026
  training_returns_by_seed.npy  Training returns across five seeds
  learning_curve.png            Mean learning curve and +/- 1 SD across five seeds
  policy_comparison.png         Held-out metrics across three policies
  policy_heatmap.png             Policy slices across forecast and sales-pace categories
  sample_episode_trace.csv      One matched eight-slot policy trace
  reward_sensitivity.json       Results for three waste-penalty settings
  reward_sensitivity.png        Waste / service trade-off under reward sensitivity
run_experiment.py      Validation tuning, training, held-out evaluation, stress tests and plots
run_reward_sensitivity.py  Reward-weight sensitivity experiment
make_report.py         Reproducible two-page PDF report generator
AI_A2_Report.pdf       Generated two-page technical report
requirements.txt       Python dependencies
VALIDATION_NOTES.md    Experiment assumptions, findings and limitations
```

## Environment setup

Use Python 3.10 or newer. Run commands from the project root (the folder containing `run_experiment.py`).

### Windows PowerShell / VS Code

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run tests and reproduce the experiments

```bash
python -m unittest discover -s tests -v
python run_experiment.py
python run_reward_sensitivity.py
python make_report.py
```

The test suite contains 20 tests. The main run trains five final agents for 6,000 episodes each. Before final training, a four-candidate learning-rate/epsilon-decay sweep is evaluated on separate validation scenarios using two training seeds. The rule-based baseline's target fraction and sales-history correction are also selected on validation scenarios. Final test episodes are held out from this tuning.

The reward-sensitivity run trains three seeds for each of three waste-penalty values, for 4,000 episodes per seed. It is an exploratory diagnostic. If you rerun experiments, regenerate the report so its figures and metrics remain synchronized.

## RL formulation

- **State:** `(time_slot, exact_inventory, noisy_forecast, sales_pace_category, prior_stockout_flag)`. There are 8 time slots, 41 inventory levels (0-40), 3 forecast categories, 3 sales-pace categories, and 2 prior-stockout states: `8 × 41 × 3 × 3 × 2 = 5,904` encoded observations.
- **Sales pace:** cumulative sales in completed slots are compared with cumulative expected demand computed from the Bayesian posterior implied by the noisy forecast. Below 85% is labelled `below_expected`; above 115% is `above_expected`; otherwise it is `on_track`. Before the first slot the category is `on_track`.
- **Prior-stockout feature:** a Boolean flag records whether unmet demand occurred in an earlier slot. It helps prevent low sales caused by unavailable meals from being mistaken for low demand.
- **Action:** request an additional preparation batch from `{0, 5, 10, 15, 20}` portions. The 20-portion maximum per slot is a modelling assumption, not a measured kitchen limit. Actual production is clipped by the separate 40-portion inventory capacity.
- **Demand and forecast:** the true low/normal/high daily regime is sampled from prior probabilities `0.25/0.50/0.25` with multipliers `0.70/1.00/1.30`. The agent sees a noisy category forecast that is correct with assumed probability `0.70`; when wrong, one of the other categories is selected uniformly. Slot demand follows a Poisson distribution around the hidden regime-adjusted mean.
- **Partial observability:** the true regime persists through the day. The sales-pace and prior-stockout features summarize history but do not maintain a full Bayesian belief state, so this remains a compact observation-based approximation.
- **Carryover:** 30% of leftover portions, rounded down, carries to the next slot; the rest is discarded. Nothing carries past the final slot.
- **Reward:** `sold - 0.35 × prepared - 1.00 × unmet_demand - 1.50 × discarded_waste`. These are illustrative utility weights, not currency costs.
- **Algorithm:** tabular Q-learning with epsilon-greedy exploration. The learning rate and epsilon decay are selected from a small validation sweep; the chosen values are recorded in `results/metrics.json`. Terminal transitions do not bootstrap a future value.

## Validation methodology

1. **Baseline tuning:** the history-aware rule receives the same forecast, sales-pace and prior-stockout observations as the Q-learning agent. Its target fraction and history correction are selected on a separate validation set, not the final test set.
2. **Q-learning hyperparameter selection:** four learning-rate/epsilon-decay combinations are screened using two tuning seeds and separate validation episodes. The selected configuration is then trained with five final seeds.
3. **Held-out comparison:** the learned policy, a fixed 15-portion batch, and the validation-tuned history-aware rule are evaluated on the same 2,000 synthetic test days. Matched per-episode demand draws support paired comparisons.
4. **Demand-shift stress test:** the trained policy and rule are evaluated when nominal demand means are scaled to `0.75×` and `1.25×`. Neither receives advance knowledge of the shifted means.
5. **Reward sensitivity:** each learned policy is scored using the same base reward weights, even when its training waste penalty differs. This illustrates reward sensitivity, not an objectively optimal reward coefficient.
6. **Uncertainty reporting:** metrics include means, standard deviations, approximate 95% confidence intervals, and paired reward differences. These quantify variability in the simulator, not uncertainty about real-world impact.

## Current reference results

Generated with the current code and fixed seeds. Run `python run_experiment.py` to reproduce.

| Policy | Reward/day | Sold/day | Unmet demand/day | Waste/day |
|---|---:|---:|---:|---:|
| Q-learning (seed 2026) | 25.30 | 118.04 | 37.72 | 7.41 |
| Fixed 15-portion batch | -5.08 | 106.48 | 49.29 | 13.52 |
| Forecast-aware history rule | 26.60 | 121.93 | 33.84 | 10.17 |

On the held-out nominal test set, Q-learning's mean reward is `25.30` (95% CI `24.56–26.05`) versus `26.60` (`25.81–27.38`) for the tuned history-aware rule. The paired difference (Q-learning minus the tuned rule) is `-1.29` (95% CI `-1.66–-0.93`). Q-learning discards less food (7.41 vs 10.17 portions/day) but serves fewer portions and has more unmet demand. This is a trade-off, not an unconditional RL win.

Across five final training seeds, Q-learning reward averages `24.91` with standard deviation `0.39`. Under demand shifts, mean rewards for Q-learning versus the nominal-planning history rule are `8.60 vs 12.69` at `0.75×` demand and `3.53 vs 9.03` at `1.25×` demand. These results are limited to this simulator.

## Reward-weight sensitivity

The sensitivity experiment varies the waste penalty used during training (`0.75`, `1.50`, `2.25`) and scores every resulting policy using the original reward weights. Latest means across three seeds are:

| Training waste penalty | Base reward/day (SD) | Waste/day (SD) | Unmet demand/day (SD) |
|---:|---:|---:|---:|
| 0.75 | 24.77 (0.50) | 8.23 (0.37) | 37.23 (0.62) |
| 1.50 | 23.93 (1.51) | 6.55 (1.02) | 39.63 (2.05) |
| 2.25 | 20.99 (0.85) | 4.61 (0.10) | 43.57 (0.52) |

Higher waste penalties generally reduce discarded portions but increase unmet demand. The SD values are across trained seeds, not confidence intervals. See `results/reward_sensitivity.json` and `results/reward_sensitivity.png`.

## Output interpretation

- `results/metrics.json`: selected hyperparameters, baseline tuning, primary metrics, five-seed summaries, paired comparisons and demand-shift tests.
- `results/policy_heatmap.png`: policy slices by forecast and sales-pace category with the prior-stockout flag fixed to false; it is a slice of the full policy, not every state.
- `results/sample_episode_trace.csv`: per-slot actions and outcomes for one matched synthetic day. It is illustrative, not statistically representative.

## Conclusion

This project uses a custom simulator to study tabular Q-learning for balancing food availability and waste in a campus canteen. The sales-pace and prior-stockout state features let the policy use a compact summary of previous outcomes under a noisy forecast. The learned policy reduces waste relative to the tuned baseline under nominal conditions but has lower reward and higher unmet demand. All findings are synthetic and do not establish measured canteen savings.
