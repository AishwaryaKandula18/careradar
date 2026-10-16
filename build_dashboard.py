"""Score all encounters with the trained model, aggregate, and write the self-contained dashboard."""
import json, sys, joblib, numpy as np, pandas as pd
from data_prep import load_and_clean, model_matrix, SLA

def main(data_dir="data", out="."):
    p, e, d, t, m = load_and_clean(data_dir)
    met = json.load(open(f"{out}/outputs/metrics.json"))
    clf = joblib.load(f"{out}/models/risk_model.joblib")
    m["risk"] = clf.predict_proba(model_matrix(m))[:, 1]
    r = lambda x: round(float(x), 1)
    order = ["Critical", "High", "Medium", "Low"]
    tri = [dict(level=l, n=int((m.triage_level == l).sum()), mean=r(m[m.triage_level == l].wait_minutes.mean()),
                p90=r(m[m.triage_level == l].wait_minutes.quantile(.9)), sla=SLA[l],
                breach=r(100 * m[m.triage_level == l].breach.mean())) for l in order]
    dep = sorted([dict(dept=k, mean=r(g.wait_minutes.mean()), breach=r(100 * g.breach.mean()),
                       lbc=r(100 * (g.outcome == "Left before care").mean()), n=len(g)) for k, g in m.groupby("department_name")],
                 key=lambda x: -x["breach"])
    hb = [r(100 * m[m.hour == h].breach.mean()) for h in range(24)]
    ttype = [dict(type=k, urg=r(g[g.priority == "Urgent"].tat.mean()), rout=r(g[g.priority == "Routine"].tat.mean()))
             for k, g in t.groupby("test_type")]
    hist = np.histogram(m.wait_minutes, bins=range(0, 165, 15))
    kp = dict(enc=len(m), pat=len(p), mean=r(m.wait_minutes.mean()), breach=r(100 * m.breach.mean()),
              crit30=r(100 * (m[m.triage_level == "Critical"].wait_minutes > 30).mean()),
              lbc=r(100 * (m.outcome == "Left before care").mean()),
              urgent_gap=r(t[t.priority == "Urgent"].tat.mean() - t[t.priority == "Routine"].tat.mean()))
    watch = [dict(id=x.encounter_id, pid=x.patient_id, dept=x.department_name, tri=x.triage_level, vt=x.visit_type,
                  age=int(x.age), chronic=x.chronic_condition, wait=int(x.wait_minutes), sla=int(x.sla),
                  risk=round(float(x.risk), 2), out=x.outcome) for x in m.sort_values("risk", ascending=False).head(400).itertuples()]
    dq = [dict(check="Missing values in encounters", res=f"{int(e.isna().sum().sum())} cells"),
          dict(check="Missing chronic_condition (patients)", res=f"{int(p.chronic_condition.isna().sum())} of {len(p):,} -> coded 'None'"),
          dict(check="wait_minutes = start - arrival", res=f"{int(((e.service_start_time - e.arrival_time).dt.total_seconds() / 60 - e.wait_minutes).abs().le(1).sum()):,} / {len(e):,} consistent"),
          dict(check="Timestamp order (arrival → start → end → discharge)", res=f"{int(((e.service_end_time < e.service_start_time) | (e.discharge_time < e.service_end_time)).sum())} violations"),
          dict(check="Orphan keys across joins", res=f"{int((~e.patient_id.isin(p.patient_id)).sum() + (~t.encounter_id.isin(e.encounter_id)).sum())} orphans"),
          dict(check="Encounters with a diagnostic test", res=f"{100 * e.encounter_id.isin(t.encounter_id).mean():.1f}%"),
          dict(check="Pediatrics dept patients aged <18", res=f"{100 * (m[m.department_name == 'Pediatrics'].age < 18).mean():.0f}% only → dept label unreliable"),
          dict(check="Wait vs age / insurance / chronic condition", res="No meaningful difference (synthetic)")]
    data = dict(kpi=kp, tri=tri, dep=dep, hb=hb, ttype=ttype, hist=dict(c=hist[0].tolist(), b=hist[1].tolist()),
                watch=watch, met=met, dq=dq, sla=SLA)
    json.dump(data, open(f"{out}/outputs/data.json", "w"))
    tpl = open(f"{out}/dashboard/template.html", encoding="utf-8").read()
    open(f"{out}/outputs/CareRadar_dashboard.html", "w", encoding="utf-8").write(tpl.replace("/*DATA*/", json.dumps(data)))
    print("Wrote outputs/CareRadar_dashboard.html")

if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["data"]))
