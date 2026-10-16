"""Train the SLA-breach risk model FROM SCRATCH (no pretrained weights, no API calls).
Time-based split: train Jan-Sep 2025, test Oct-Dec 2025. Baseline = triage-level breach rate."""
import json, sys, joblib, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from data_prep import load_and_clean, model_matrix

def main(data_dir="data", out="."):
    *_, m = load_and_clean(data_dir)
    X = model_matrix(m)
    tr = m.arrival_time < "2025-10-01"; te = ~tr
    rate = m[tr].groupby("triage_level").breach.mean()
    base = m.loc[te, "triage_level"].map(rate)
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                         categorical_features="from_dtype", random_state=7)
    clf.fit(X[tr], m.breach[tr])
    pr = clf.predict_proba(X[te])[:, 1]
    met = dict(n_train=int(tr.sum()), n_test=int(te.sum()), breach_rate=float(m.breach.mean()),
               base_auc=roc_auc_score(m.breach[te], base), base_ap=average_precision_score(m.breach[te], base),
               gb_auc=roc_auc_score(m.breach[te], pr), gb_ap=average_precision_score(m.breach[te], pr))
    joblib.dump(clf, f"{out}/models/risk_model.joblib")
    json.dump(met, open(f"{out}/outputs/metrics.json", "w"), indent=2)
    print(json.dumps(met, indent=2))

if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["data"]))
