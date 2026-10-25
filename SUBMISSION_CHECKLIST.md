# AI-A2 Submission Checklist

**Deadline in the assignment PDF: 26 October 2026, 10:00 AM sharp.**

## 1. Run and inspect locally

From the project root (the folder containing `run_experiment.py`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
python run_reward_sensitivity.py
python make_report.py
```

All **13 tests** should pass. Inspect `results/metrics.json`, `results/reward_sensitivity.json`, the policy comparison, learning curve, sensitivity plot, policy heatmap, and sample trace. All result data is synthetic; it is not real canteen evidence.

## 2. Complete the report

Edit the constants near the top of `make_report.py`:

- `TEAM_MEMBERS = "[Enter name(s) and roll number(s)]"`
- `GITHUB_URL = "[Insert repository URL after publishing]"`

Use your actual details and repository URL. Re-run `python make_report.py`, open `AI_A2_Report.pdf`, confirm it is exactly two pages, and verify no placeholders remain.

## 3. Publish the repository

Create a GitHub repository, then from the project root:

```bash
git init
git add README.md requirements.txt .gitignore canteen_rl tests run_experiment.py run_reward_sensitivity.py make_report.py results AI_A2_Report.pdf SUBMISSION_CHECKLIST.md VALIDATION_NOTES.md
git commit -m "Add canteen simulation and Q-learning evaluation"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

Make genuine follow-up commits as you test and revise the work. Do not manufacture development history. If the repository is private, follow the access-sharing instructions in the assignment PDF.

## 4. Final verification

- [ ] The implemented project matches the approved idea.
- [ ] The source code, requirements, tests, saved Q-table, metrics and figures are in GitHub.
- [ ] A clean environment can run the setup/test/experiment commands.
- [ ] `README.md` accurately describes the environment, algorithm, baselines and limitations.
- [ ] The report is exactly two pages, includes the required RL formulation and graphs, and contains the actual team details and GitHub URL.
- [ ] The report describes the synthetic simulator and reports the tuned baseline trade-off accurately.
- [ ] The course's policy on AI-assisted code and disclosure is followed; understand the code well enough for a viva.
- [ ] Submit before 26 October 2026, 10:00 AM.
