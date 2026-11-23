"""Monte Carlo deadline forecaster (PERT-beta durations + dependency network)."""
import numpy as np
import pandas as pd


def load_plan(path):
    df = pd.read_csv(path)
    df["deps"] = df["depends_on"].fillna("").apply(lambda s: [x for x in s.split(";") if x])
    df["num_dependencies"] = df["deps"].apply(len)
    df["estimate_days"] = df["likely"]
    return df


def topo_order(df):
    deps = dict(zip(df.task_id, df.deps))
    order, seen = [], set()

    def visit(t, stack=()):
        if t in stack:
            raise ValueError(f"Circular dependency at {t}")
        if t in seen:
            return
        for d in deps[t]:
            visit(d, stack + (t,))
        seen.add(t)
        order.append(t)

    for t in df.task_id:
        visit(t)
    return order


def pert_samples(o, l, p, n, rng):
    """PERT = beta distribution stretched between optimistic and pessimistic."""
    if p <= o:
        return np.full(n, float(l))
    a = 1 + 4 * (l - o) / (p - o)
    b = 1 + 4 * (p - l) / (p - o)
    return o + rng.beta(a, b, n) * (p - o)


def simulate(plan, n=10_000, seed=7, risk_multiplier=None):
    """Simulate project duration, criticality, and task sensitivity."""
    rng = np.random.default_rng(seed)
    order = topo_order(plan)
    row = plan.set_index("task_id")
    dur = {}
    for t in order:
        r = row.loc[t]
        if r.status == "Done":
            d = np.zeros(n)
        else:
            d = pert_samples(r.optimistic, r.likely, r.pessimistic, n, rng)
            if r.status == "In Progress":
                d = d * (1 - r.pct_complete / 100)
            if risk_multiplier:
                d = d * risk_multiplier.get(t, 1.0)
        dur[t] = d
    finish = {}
    for t in order:
        d = row.loc[t, "deps"]
        start = np.max([finish[x] for x in d], axis=0) if d else np.zeros(n)
        finish[t] = start + dur[t]
    total = np.max([finish[t] for t in order], axis=0)
    succ = {t: [] for t in order}
    for t in order:
        for x in row.loc[t, "deps"]:
            succ[x].append(t)
    latest_finish = {}
    for t in reversed(order):
        latest_finish[t] = (total.copy() if not succ[t]
                            else np.min([latest_finish[s] - dur[s] for s in succ[t]], axis=0))
    crit = {t: float(np.mean(np.isclose(latest_finish[t] - finish[t], 0, atol=1e-9) & (dur[t] > 0)))
            for t in order}
    sens = {t: (float(np.corrcoef(dur[t], total)[0, 1]) if dur[t].std() > 0 else 0.0) for t in order}
    return {"total": total, "criticality": crit, "sensitivity": sens}


def summarize(total, as_of, deadline):
    """Convert simulated working days into dates and deadline probability."""
    as_of, deadline = np.datetime64(as_of), np.datetime64(deadline)
    budget = np.busday_count(as_of, deadline)
    to_date = lambda days: np.busday_offset(as_of, int(np.ceil(days)), roll="forward")
    pct = {f"P{q}": to_date(np.percentile(total, q)) for q in (50, 80, 90)}
    return {"budget_days": int(budget), "p_on_time": float(np.mean(total <= budget)),
            "dates": pct, "mean_days": float(total.mean())}
