"""Load, join and clean the four DATATHON CSVs; derive SLA flags and model features."""
import pandas as pd

# ASSUMPTION (not in the dataset): maximum acceptable wait in minutes per triage level.
SLA = {"Critical": 15, "High": 30, "Medium": 60, "Low": 90}
FEATURES = ["triage_level", "department_name", "visit_type", "hour", "dow", "age",
            "sex", "chronic_condition", "insurance_type", "dept_load"]


def load_and_clean(data_dir):
    p = pd.read_csv(f"{data_dir}/patients.csv")
    e = pd.read_csv(f"{data_dir}/encounters.csv")
    d = pd.read_csv(f"{data_dir}/departments.csv")
    t = pd.read_csv(f"{data_dir}/diagnostic_tests.csv")
    for c in ["arrival_time", "service_start_time", "service_end_time", "discharge_time"]:
        e[c] = pd.to_datetime(e[c])
    t["tat"] = (pd.to_datetime(t.completion_time) - pd.to_datetime(t.requested_time)).dt.total_seconds() / 60
    m = e.merge(p, on="patient_id", how="left").merge(d, on="department_id", how="left")
    m["chronic_condition"] = m.chronic_condition.fillna("None")
    m["sla"] = m.triage_level.map(SLA)
    m["breach"] = (m.wait_minutes > m.sla).astype(int)   # target variable
    m["hour"] = m.arrival_time.dt.hour
    m["dow"] = m.arrival_time.dt.dayofweek
    m["day"] = m.arrival_time.dt.date
    m["dept_load"] = m.groupby(["department_id", "day"]).encounter_id.transform("count")
    return p, e, d, t, m


def model_matrix(m):
    X = m[FEATURES].copy()
    for c in X.columns[X.dtypes.map(lambda t: not pd.api.types.is_numeric_dtype(t))]:
        X[c] = X[c].astype("category")
    return X
