"""Generate reproducible synthetic historical tasks and a sample project plan."""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
Path("data").mkdir(exist_ok=True)
TYPES = ["Backend", "Frontend", "Integration", "QA", "Docs", "Compliance"]
OWNERS = ["Aisha", "Rohan", "Meera", "Kabir", "Neha"]
TYPE_EFFECT = {"Backend": 0.02, "Frontend": 0.0, "Integration": 0.12, "QA": 0.03, "Docs": -0.06, "Compliance": 0.07}
OWNER_EFFECT = {"Aisha": -0.04, "Rohan": 0.0, "Meera": -0.02, "Kabir": 0.06, "Neha": 0.02}

n = 400
h = pd.DataFrame({
    "task_id": [f"H{i:03d}" for i in range(n)],
    "project": rng.choice(["Portal-A", "Portal-B", "Claims-App", "Renewals-API"], n),
    "task_type": rng.choice(TYPES, n, p=[.26, .20, .18, .16, .10, .10]),
    "priority": rng.choice(["Low", "Medium", "High"], n, p=[.25, .5, .25]),
    "owner": rng.choice(OWNERS, n),
    "estimate_days": rng.choice([2, 3, 4, 5, 6, 8, 10, 12], n),
    "owner_open_tasks": rng.integers(1, 10, n),
    "num_dependencies": rng.integers(0, 5, n),
    "client_dependency": rng.choice([0, 1], n, p=[.7, .3]),
    "scope_changes": rng.choice([0, 1, 2, 3], n, p=[.5, .3, .15, .05]),
})
log_ratio = (-0.12 + h.task_type.map(TYPE_EFFECT) + h.owner.map(OWNER_EFFECT)
             + 0.14 * h.client_dependency + 0.04 * h.scope_changes
             + 0.018 * np.maximum(h.owner_open_tasks - 4, 0)
             + 0.015 * h.num_dependencies + 0.02 * (h.priority == "High")
             + rng.normal(0, 0.11, n))
h["actual_days"] = (h.estimate_days * np.exp(log_ratio)).round(1).clip(lower=1)
h.to_csv("data/history.csv", index=False)

P = [
("T01","Requirements & BRD","Docs","Meera","High",(3,5,8),"","Done",100,3,1,1),
("T02","Broker onboarding API design","Backend","Rohan","High",(3,5,9),"T01","Done",100,4,0,0),
("T03","UI wireframes","Frontend","Aisha","Medium",(3,4,7),"T01","Done",100,3,0,1),
("T04","Quote engine integration","Integration","Kabir","High",(6,9,16),"T02","In Progress",60,7,1,2),
("T05","Broker login & KYC module","Backend","Rohan","High",(5,8,14),"T02","In Progress",40,6,1,1),
("T06","Broker dashboard UI","Frontend","Aisha","Medium",(5,7,12),"T03","In Progress",30,4,0,1),
("T07","Commission calculation module","Backend","Neha","Medium",(4,6,11),"T02","Not Started",0,5,0,2),
("T08","Insurer API integration","Integration","Kabir","High",(7,11,20),"T04","Not Started",0,8,1,1),
("T09","Payment gateway integration","Integration","Neha","High",(4,6,12),"T05","Not Started",0,5,1,0),
("T10","Document upload & e-sign","Backend","Rohan","Medium",(4,6,10),"T05","Not Started",0,6,0,1),
("T11","Regulatory compliance review","Compliance","Meera","High",(4,7,15),"T05;T10","Not Started",0,3,1,0),
("T12","Quote comparison UI","Frontend","Aisha","Medium",(4,6,10),"T04;T06","Not Started",0,4,0,1),
("T13","Reports module","Backend","Neha","Low",(3,5,9),"T07","Not Started",0,5,0,0),
("T14","Integration testing","QA","Aisha","High",(5,8,14),"T08;T09;T12","Not Started",0,5,0,1),
("T15","Security testing (VAPT)","QA","Meera","High",(4,6,12),"T10;T11","Not Started",0,3,1,0),
("T16","UAT with brokers","QA","Meera","High",(5,8,15),"T13;T14","Not Started",0,4,1,1),
("T17","Training docs","Docs","Neha","Low",(2,3,6),"T12","Not Started",0,4,0,0),
("T18","Go-live preparation","Integration","Kabir","High",(3,4,8),"T15;T16;T17","Not Started",0,6,1,0)]
cols = ["task_id","task_name","task_type","owner","priority","_est","depends_on","status","pct_complete","owner_open_tasks","client_dependency","scope_changes"]
plan = pd.DataFrame(P, columns=cols)
plan[["optimistic","likely","pessimistic"]] = pd.DataFrame(plan.pop("_est").tolist())
plan.to_csv("data/project_plan.csv", index=False)
print(f"history.csv: {len(h)} rows | project_plan.csv: {len(plan)} tasks")
