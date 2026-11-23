# Deadline Forecaster

A project-planning tool that combines task-delay prediction with Monte Carlo simulation to estimate the probability of meeting a project deadline.

## What it does

- Trains gradient-boosting models on synthetic historical task data to estimate delay probability and expected schedule slip.
- Simulates the project dependency network using PERT-style duration estimates.
- Compares a baseline forecast with an ML-adjusted forecast.
- Produces deadline confidence metrics, task-risk rankings, criticality scores, charts, and CSV outputs.

## Sample project

The included example is a synthetic InsurTech broker-portal project with 18 tasks and 400 synthetic historical task records. No real company or customer data is included.

## Run locally

```bash
pip install -r requirements.txt
python generate_data.py
python run_all.py --as-of 2026-11-23 --deadline 2027-01-22 --sims 10000
```

The pipeline writes charts and CSV reports to `outputs/`.

## Main files

- `generate_data.py` — creates the synthetic historical dataset and sample project plan.
- `delay_predictor.py` — trains and evaluates task-delay classification and slip-ratio regression models.
- `forecaster.py` — performs dependency-aware PERT / Monte Carlo schedule simulation.
- `run_all.py` — runs the end-to-end workflow and generates reports.
- `data/` — sample synthetic datasets.
- `outputs/` — example charts and CSV results.

## Important limitations

- The included data is synthetic; model performance and forecast probabilities are illustrative, not operational predictions.
- The simulation does not perform resource levelling, so tasks assigned to the same owner may overlap.
- Weekends are excluded, but public holidays are not modelled.
- Applying ML-derived slip adjustments to PERT estimates can double-count risk if estimates already include contingency. Calibrate with real project history before relying on absolute probabilities.

## Potential extensions

- Import task data from Jira or ClickUp exports.
- Add resource-constrained scheduling and holiday calendars.
- Build a dashboard for comparing scenarios and tracking forecast changes.
