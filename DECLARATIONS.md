# Declarations: APIs, pretrained models and data

| Item | Used? | Details |
|---|---|---|
| External APIs / API keys | **None** | No LLM, cloud or web API is called at any point. No keys are needed. |
| Pretrained models / weights | **None** | The risk model is trained from scratch in `src/train.py` (scikit-learn `HistGradientBoostingClassifier`, random_state=7). |
| Extra datasets | **None** | Only the four DATATHON 2026 CSVs: `patients.csv`, `encounters.csv`, `departments.csv`, `diagnostic_tests.csv`. |
| Third-party libraries | pandas, numpy, scikit-learn, joblib | Open source, listed in `requirements.txt`. The dashboard uses plain HTML/CSS/JS with no external libraries or CDNs. |
| Assumptions made by us | Triage SLA targets | Critical 15, High 30, Medium 60, Low 90 minutes. These are **our assumption**, not from the dataset; edit `SLA` in `src/data_prep.py`. |

The dataset is synthetic and educational. Nothing here is for clinical decisions.
