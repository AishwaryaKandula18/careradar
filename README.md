# CareRadar: early warning for the "forgotten patient"

DATATHON 2026, Round 1 baseline prototype. Track: **Health – The Forgotten Patient**.

CareRadar flags patients at risk of waiting longer than their triage target, so staff can act before the breach.

**Team:** [Nexus] · [Padam Charan], [Kandula Hima Lakshmi Aishwarya]

## What it does
- Joins the four tables (encounters is the hub; keys per the dataset README).
- Cleans the data and derives an **SLA breach** flag (wait > triage target).
- Trains a **gradient-boosting risk model from scratch** and compares it with a triage-only baseline.
- Builds a single-file dashboard: Overview, Watchlist (ranked at-risk patients), Diagnostics and Data Quality.

## Run steps
```bash
git clone <your-repo-url> && cd careradar
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# copy the 4 dataset CSVs into data/
python run_all.py                  # or: python run_all.py path/to/csvs
```
Then open `outputs/CareRadar_dashboard.html` in any browser (no server needed).
A pre-built copy is in `docs/`.

## Repo layout
| Path | Purpose |
|---|---|
| `src/data_prep.py` | load, join, clean, SLA flag, features |
| `src/train.py` | trains and evaluates the model, saves `models/risk_model.joblib`, `outputs/metrics.json` |
| `src/build_dashboard.py` | scores encounters and writes the dashboard |
| `dashboard/template.html` | dashboard UI (plain HTML/CSS/JS) |
| `docs/screenshots/` | prototype screenshots |
| `DECLARATIONS.md` | APIs, pretrained models, extra data (none) |

## Model
- **Target:** wait exceeds triage target. **Features (known at arrival):** triage, department, visit type, hour, weekday, age, sex, chronic condition, insurance, daily department load.
- **Split:** time-based. Train Jan–Sep 2025 (5,980), test Oct–Dec 2025 (2,020).
- **Baseline:** triage-level breach rate.

| Metric (test) | Triage-only rule | Our model |
|---|---|---|
| ROC-AUC | 0.62 | 0.69 |
| PR-AUC | 0.33 | 0.43 |

The lift is modest. Waits in this synthetic data are mostly driven by triage level.

## Key findings
- 25.9% of 8,000 encounters wait past target; 44.9% of Critical patients wait over 15 min.
- Pediatrics (42%) and Laboratory (39.9%) breach most; Cardiology least (12.7%).
- Urgent tests are not faster than routine (69 vs 70 min).

## Limitations
Synthetic data; SLA targets are assumed (see `DECLARATIONS.md`); no clinical outcomes; decision support only, not for clinical decisions.

## Screenshots
![Overview]("docs/screenshots/01_overview.png")
![Watchlist](docs/screenshots/02_watchlist.png)
![Diagnostics](docs/screenshots/03_diagnostics.png)

## License
MIT
