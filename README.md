# Musanze Harvest Intelligence Service

**Mohammed Osama Hasan**  
**Registration number:** 25/27014  
**Institutional username:** ug2527014@ines.ac.rw  
**Verification code:** AI-A2-7014  
**Course:** SWE 3513 — Artificial Intelligence, INES-Ruhengeri  
**Repository:** https://github.com/Mogadi/Musanze_Harvest_Intelligence_Service  
**Final commit hash:** `05c06d9a76a95bf3bdb8fde4d824ad8ba50c3825` on branch `dataset-generation-validation`  
This hash is the system commit that contains the trained models and the API. The report and evidence files are added after it. Show `git rev-parse HEAD` in the demonstration for the tip of the branch.

This repository is an individual classroom prototype. It estimates potato harvest quantity, flags farms at high production risk, and groups farms with similar growing conditions for a Musanze potato cooperative. The dataset is **synthetic**. It is generated from registration number 25/27014 and is not a record of real farms.

## Decision problem

A field officer receives farm records in inconsistent spreadsheets and needs a local tool that can:

1. Estimate harvest quantity in tons (`yield_tons`) — regression.
2. Identify farms whose yield per hectare is below the median (`high_risk`) — classification.
3. Group farms that share similar conditions — clustering.

Training and command-line prediction must work without internet. A small FastAPI service is included so a later mobile application can call the same saved models. The service supports a human decision. It does not replace an agronomist and must not be used to grant or deny finance, insurance, or public services.

## Individual data signature

| Item | Value |
| :--- | :--- |
| Registration number | 25/27014 |
| `LLLL` (last four digits) | 7014 |
| Random seed | `numpy.random.default_rng(7014)` |
| Train/test `random_state` | 7014 |
| Base rows | `240 + (7014 mod 41)` = 243 |
| Deliberate missing cells | `m = round(0.05 × 243)` = 12 positions each for `soil_moisture_pct` and `fertilizer_kg` |
| Exact duplicate rows appended | 3 |
| Verification code | `AI-A2-7014` |
| CSV numeric format | four decimal places (`float_format="%.4f"`) |
| Dataset fingerprint | `209caa6bff4616bdeee0e6de48554e870d27289974057935d7287f4655323b10` in `artifacts/data_fingerprint.txt` |

`python run_all.py` rejects any seed other than `7014`.

## Environment

| Item | Tested value |
| :--- | :--- |
| Python | 3.14.4 |
| Operating system | Ubuntu 26.04.1 LTS (Linux 7.0.0-38-generic, x86_64) |
| Main packages | NumPy, Pandas, scikit-learn, Matplotlib, joblib, FastAPI, uvicorn, pytest |
| Report tool | fpdf2, used only to typeset the report, test log, and declaration. Training and prediction do not import it. |

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --student-id ug2527014@ines.ac.rw --seed 7014
python predict.py --json samples/valid.json
pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000
```

`run_all.py` recreates the dataset, cleans it, trains regression, classification, and clustering, and regenerates every artifact. Running it twice must reproduce the same dataset fingerprint and the same metrics within normal floating-point tolerance.

## Repository layout

```text
AI_A2_2527014/
    README.md
    requirements.txt
    run_all.py
    predict.py
    app.py
    src/
    data/generate_data.py
    data/AI_A2_2527014.csv
    models/
    artifacts/
    samples/valid.json
    samples/invalid.json
    tests/
    docs/AI_A2_2527014_REPORT.pdf
    evidence/
```

## Artifact list

| Path | Content |
| :--- | :--- |
| `artifacts/data_fingerprint.txt` | SHA-256 of the generated CSV and the verification code |
| `artifacts/regression_metrics.json` | Mean baseline and candidate MAE, RMSE, and R², plus the selected model and reason |
| `artifacts/classification_metrics.json` | Majority baseline and candidate precision, recall, F1, confusion matrix, and ROC-AUC |
| `artifacts/clustering_report.json` | Silhouette scores for K = 2..6, selected K, and a plain-language profile of each cluster |
| `artifacts/test_predictions.csv` | Row ID, actual values, predictions, probabilities, errors, and cluster |
| `artifacts/plots/residuals.png` | Regression residual plot |
| `artifacts/plots/confusion_matrix.png` | Classification confusion matrix |
| `artifacts/plots/clusters.png` | Cluster plot |
| `models/` | Saved preprocessing and final models used by `predict.py` and `app.py` |
| `docs/AI_A2_2527014_REPORT.pdf` | Technical report. Its figures match the JSON artifacts. |
| `evidence/AI_A2_2527014_TEST_LOG.pdf` | Output of `pytest -q` and the named test list |
| `evidence/AI_A2_2527014_DECLARATION.pdf` | Individual declaration |
| `evidence/AI_USE.md` | Disclosure of how an AI coding assistant was used |

## API examples

Start the service, then call it locally.

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Model information (includes `AI-A2-7014`):

```bash
curl http://127.0.0.1:8000/model-info
```

Valid prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @samples/valid.json
```

A valid response includes the yield estimate, risk prediction, probability, cluster, and verification code `AI-A2-7014`. Invalid input — a missing field, a wrong type, or a value outside the allowed range — returns a 4xx error and does not produce a prediction. Interactive documentation is at `http://127.0.0.1:8000/docs`.

Command-line prediction uses the same saved artifacts:

```bash
python predict.py --json samples/valid.json
```

## Known limitations

- The dataset is synthetic. Coefficients, noise, missing values, and duplicate rows are assigned by the coursework rules. They are not measurements from Musanze farms.
- Models are fit on a few hundred generated rows. Metrics describe this generated hold-out set only.
- `high_risk` is defined from yield per hectare, so classification quality partly reflects how well the features explain yield. Yield, yield per hectare, and `high_risk` are never used as classification inputs.
- Internet is not required for training or for `predict.py`. The API is a local prototype for a future mobile client.
- A field officer must review every estimate before acting on it. This system must not replace an agronomist or decide access to finance, insurance, or public services.

## Responsible use

Intended use is classroom decision support for a Musanze potato cooperative scenario: estimate harvest quantity, highlight higher-risk farms, and describe groups of similar farms. Prohibited use includes automated approval or denial of credit, insurance, subsidies, or any public service, and any claim that the output is a real agronomic measurement.
