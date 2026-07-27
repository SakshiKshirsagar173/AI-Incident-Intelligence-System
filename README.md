# AI Incident Intelligence System

Predicts incident **Priority** and estimated **Resolution_Time_Hours** from
ticket-intake fields, served via a FastAPI REST API.

## ⚠️ Known limitation (read this first)

The provided `it_incident_dataset.csv` (1,200 rows) has `Priority` and
`Resolution_Time_Hours` values that are **very weakly correlated** with
`Incident_Type`, `Assigned_Department`, `Location`, or the reported timestamp
(see `models/training_summary.json` and `models/training_summary_reg.json`).

- Best Priority classifier (Random Forest): **~30% accuracy** vs. a 27%
  majority-class baseline.
- Best resolution-time regressor (Linear Regression): **R² ≈ 0.02**, i.e.
  effectively no explained variance.

This means the current models are not reliable in a real deployment — the
pipeline is fully wired and correct end-to-end, but the *data* doesn't
contain a strong signal for these targets. The `/predict` endpoint response
includes a `model_confidence_note` field surfacing this to API consumers
rather than hiding it. To get a model that's actually useful in production,
swap in a dataset where Priority/resolution time are meaningfully driven by
the incident's features.

## Project structure

```
project/
├── data/
│   ├── raw/it_incident_dataset.csv
│   └── processed/           # train/test splits (classification + regression)
├── models/                  # trained models, encoders, metadata, metrics
├── src/
│   ├── eda.py                    # Phase 1: exploratory analysis
│   ├── preprocess.py             # Phase 2: Priority classification preprocessing
│   ├── preprocess_regression.py  # Phase 2: Resolution time preprocessing
│   ├── train.py                  # Phase 3: Priority model training + comparison
│   └── train_regression.py       # Phase 3: Resolution time model training + comparison
├── app/
│   ├── schemas.py           # Pydantic request/response models
│   ├── model.py             # Model loading + inference logic
│   └── main.py               # FastAPI app (Phase 6-7)
└── requirements.txt
```

## Design notes

- **No data leakage**: `Resolved_Time`, `Status`, and `Resolution_Type` are
  excluded from all features because they don't exist yet when a ticket is
  first submitted. `Priority` IS used as a feature for the resolution-time
  model, since in a real workflow priority is assigned at intake, before
  resolution time is known.
- **Training/serving consistency**: `preprocessing_meta*.json` records the
  exact final feature column order used at training time; `app/model.py`
  reindexes to that same order at inference time.

## Running locally

```bash
pip install -r requirements.txt
cd app
uvicorn main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for interactive Swagger UI, or:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "incident_type": "Network Outage",
    "assigned_department": "Network Team",
    "location": "Data Center A"
  }'
```

## Reproducing the pipeline from scratch

```bash
cd src
python3 eda.py
python3 preprocess.py
python3 train.py
python3 preprocess_regression.py
python3 train_regression.py
```

## Running with Docker

```bash
docker build -t incident-intelligence .
docker run -p 8000:8000 incident-intelligence
# or:
docker compose up --build
```

Then hit it the same way as local dev: `curl http://localhost:8000/health`.

The image only copies `app/` and `models/` (not `data/` or `src/`, which are
training-time only) — see `.dockerignore`. It runs as a non-root user and
has a built-in `HEALTHCHECK` hitting `/health` every 30s.

> Note: I don't have Docker available in this sandbox to run an actual
> `docker build`, so I couldn't test the real image. Instead I verified
> correctness by recreating the exact file layout the Dockerfile produces
> (`COPY app/ ./app/` + `COPY models/ ./models/`, then running from `app/`
> as the `CMD` does) in an isolated directory and confirming `/health` and
> `/predict` both work from that layout. Please do run `docker build` +
> `docker run` yourself once to confirm on your machine before deploying.

## Deploying to Render (Phase 9)

Render needs your code in a GitHub repo, then builds/deploys from there.
I can't create the GitHub repo or Render account for you, but here's the
exact path:

### 1. Push this project to GitHub

```bash
cd project
git init
git add .
git commit -m "Initial commit: incident intelligence API"
```
Create a new empty repo on github.com (no README/license), then:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

### 2. Create the Render service

1. Sign up / log in at [render.com](https://render.com).
2. Click **New +** → **Web Service**.
3. Connect your GitHub account and select this repo.
4. Render should auto-detect `render.yaml` in this repo and pre-fill the
   settings below. If it doesn't, set them manually:
   - **Runtime**: Docker
   - **Dockerfile path**: `./Dockerfile`
   - **Health check path**: `/health`
   - **Instance type**: Free (fine for this project)
5. Click **Create Web Service**. Render will build the image and deploy it —
   watch the build logs; it should show the same "Model artifacts loaded
   successfully" line you saw locally.
6. Once live, Render gives you a public URL like
   `https://ai-incident-intelligence.onrender.com`. Test it:
   ```bash
   curl https://ai-incident-intelligence.onrender.com/health
   ```

### Notes on the free tier

- Free-tier services spin down after inactivity and take ~30-60s to wake up
  on the next request — the first `/predict` call after idle time will be
  slow, that's expected, not a bug.
- Render sets a `PORT` environment variable itself; the Dockerfile already
  reads it (`--port ${PORT:-8000}`), so no manual config needed there.

## Not yet built (next phases)

- **Phase 10 — RAG/LLM**: the source dataset has no free-text incident
  description field, so there's nothing to embed/retrieve yet. This phase
  would need either a synthesized text field (e.g. a templated sentence per
  incident) or a richer dataset with real ticket text.
