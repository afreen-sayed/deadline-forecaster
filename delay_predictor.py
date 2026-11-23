"""Train ML models to estimate task delay probability and expected schedule slip."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CAT = ["task_type", "priority", "owner"]
NUM = ["estimate_days", "owner_open_tasks", "num_dependencies", "client_dependency", "scope_changes"]
DELAY_THRESHOLD = 1.2


def _pipe(model):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                             ("num", "passthrough", NUM)])
    return Pipeline([("pre", pre), ("model", model)])


def train(history):
    X = history[CAT + NUM]
    ratio = history.actual_days / history.estimate_days
    y_cls, y_reg = (ratio > DELAY_THRESHOLD).astype(int), np.log(ratio)
    Xtr, Xte, ctr, cte, rtr, rte = train_test_split(
        X, y_cls, y_reg, test_size=0.25, random_state=1, stratify=y_cls)
    clf = _pipe(GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.08, random_state=0)).fit(Xtr, ctr)
    reg = _pipe(GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.08, random_state=0)).fit(Xtr, rtr)
    auc = roc_auc_score(cte, clf.predict_proba(Xte)[:, 1])
    mae_model = mean_absolute_error(np.exp(rte), np.exp(reg.predict(Xte)))
    mae_naive = mean_absolute_error(np.exp(rte), np.full(len(rte), np.exp(rtr).mean()))
    imp = permutation_importance(clf, Xte, cte, scoring="roc_auc", n_repeats=20, random_state=0)
    importance = pd.Series(imp.importances_mean, index=CAT + NUM).sort_values(ascending=False)
    metrics = {"auc": auc, "mae_model": mae_model, "mae_naive": mae_naive,
               "delay_rate": float(y_cls.mean()), "mean_ratio": float(np.exp(y_reg).mean()),
               "n_train": len(Xtr), "n_test": len(Xte)}
    return clf, reg, metrics, importance


def score_plan(clf, reg, plan):
    out = plan[["task_id", "task_name", "task_type", "owner", "status"]].copy()
    X = plan[CAT + NUM]
    out["delay_probability"] = clf.predict_proba(X)[:, 1].round(3)
    out["expected_slip_ratio"] = np.exp(reg.predict(X)).round(3)
    out["risk_level"] = pd.cut(out.delay_probability, [-1, .35, .6, 2], labels=["Low", "Medium", "High"])
    return out.sort_values("delay_probability", ascending=False)
