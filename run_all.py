"""End-to-end pipeline: historical data -> ML delay model -> Monte Carlo forecast."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import delay_predictor as dp
import forecaster as fc

ap = argparse.ArgumentParser()
ap.add_argument("--as-of", default="2026-11-23")
ap.add_argument("--deadline", default="2027-01-22")
ap.add_argument("--sims", type=int, default=10_000)
args = ap.parse_args()

if not Path("data/project_plan.csv").exists():
    import generate_data  # noqa: F401
if not Path("data/history.csv").exists():
    import generate_data  # noqa: F401
Path("outputs").mkdir(exist_ok=True)
history = pd.read_csv("data/history.csv")
plan = fc.load_plan("data/project_plan.csv")
clf, reg, metrics, importance = dp.train(history)
risk = dp.score_plan(clf, reg, plan)
risk.to_csv("outputs/task_risk_scores.csv", index=False)
mult = {r.task_id: float(np.clip(r.expected_slip_ratio, 0.9, 1.5)) for r in risk.itertuples()}
base = fc.simulate(plan, args.sims)
adj = fc.simulate(plan, args.sims, risk_multiplier=mult)
s_base = fc.summarize(base["total"], args.as_of, args.deadline)
s_adj = fc.summarize(adj["total"], args.as_of, args.deadline)

print(f"Deadline {args.deadline} | as of {args.as_of} | {s_base['budget_days']} working days available")
print(f"ML model: AUC {metrics['auc']:.2f} | slip-ratio MAE {metrics['mae_model']:.3f} vs naive {metrics['mae_naive']:.3f}")
print("Top delay drivers:", ", ".join(f"{k} ({v:.3f})" for k, v in importance.head(4).items()))
for label, result in (("Naive estimates", s_base), ("ML-adjusted", s_adj)):
    print(f"{label}: P(on time)={result['p_on_time']:.0%} | " +
          " | ".join(f"{k} {v}" for k, v in result["dates"].items()))
print("Top 5 riskiest tasks:")
print(risk.head(5)[["task_id","task_name","delay_probability","expected_slip_ratio"]].to_string(index=False))
pd.DataFrame([{"scenario": label, "p_on_time": result["p_on_time"],
               **{k: str(v) for k,v in result["dates"].items()}}
              for label,result in (("naive",s_base),("ml_adjusted",s_adj))]).to_csv(
                  "outputs/forecast_summary.csv", index=False)

BLUE, ORANGE, RED, GREY = "#2563eb", "#ea580c", "#dc2626", "#6b7280"
names = dict(zip(plan.task_id, plan.task_name))
budget = s_base["budget_days"]
fig, ax = plt.subplots(figsize=(9,4.8))
bins = np.linspace(min(base["total"].min(),adj["total"].min()),max(base["total"].max(),adj["total"].max()),60)
ax.hist(base["total"],bins,alpha=.55,color=BLUE,label=f"Naive ({s_base['p_on_time']:.0%} on time)")
ax.hist(adj["total"],bins,alpha=.55,color=ORANGE,label=f"ML-adjusted ({s_adj['p_on_time']:.0%} on time)")
ax.axvline(budget,color=RED,lw=2,ls="--",label=f"Deadline {args.deadline}")
ax.set(xlabel="Working days to finish",ylabel="Simulations",title="Simulated project finish distribution")
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("outputs/1_finish_distribution.png",dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(9,4.8))
for total,color,label in ((base["total"],BLUE,"Naive"),(adj["total"],ORANGE,"ML-adjusted")):
    xs=np.sort(total); ax.plot(xs,np.arange(1,len(xs)+1)/len(xs)*100,color=color,lw=2.2,label=label)
ax.axvline(budget,color=RED,lw=2,ls="--"); ax.axhline(80,color=GREY,lw=1,ls=":")
ax.set(xlabel="Working days from today",ylabel="Chance of finishing by then (%)",title="Deadline confidence curve")
ax.legend(frameon=False,loc="lower right"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("outputs/2_confidence_curve.png",dpi=150); plt.close(fig)
fig, axes = plt.subplots(1,2,figsize=(12,4.8))
top=pd.Series(adj["criticality"]).sort_values().tail(8)
axes[0].barh([f"{t} {names[t][:24]}" for t in top.index],top.values*100,color=RED)
axes[0].set(xlabel="% of simulations on critical path",title="Critical path (ML-adjusted)")
top=risk.head(8).iloc[::-1]
axes[1].barh([f"{r.task_id} {r.task_name[:24]}" for r in top.itertuples()],top.delay_probability*100,color=ORANGE)
axes[1].set(xlabel="Predicted chance of running >20% over (%)",title="Task delay risk")
for axis in axes: axis.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("outputs/3_critical_path_and_risk.png",dpi=150); plt.close(fig)
print("Saved charts and CSV reports to outputs/.")
