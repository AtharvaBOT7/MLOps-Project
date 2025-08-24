Movie Review Sentiment Analysis MLOps Project
==============================

A production-minded MLOps setup using **Cookiecutter**, **DVC**, **MLflow on DagsHub**, **GitHub Actions**, **Docker**, and **AWS S3**. Includes a lightweight **Flask API** container for quick local verification.

## 🚀 Tech Stack

- **Project Scaffolding:** Cookiecutter Data Science  
- **Version Control & CI/CD:** Git, GitHub Actions  
- **Experiment Tracking:** MLflow (DagsHub backend)  
- **Data/Model Versioning:** DVC (local + S3 remote)  
- **Modeling:** scikit-learn, XGBoost, NLTK  
- **Serving:** Flask app (containerized)  
- **Containerization:** Docker  
- **Cloud Storage:** AWS S3 (DVC remote)  
- **(Future) Orchestration & Monitoring:** ECR/EKS/EC2/CloudFormation + Prometheus/Grafana  

---

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

> If you renamed `src.models` → `src.model`, ensure imports reflect that change consistently.

---

## ✅ Prerequisites

- Python 3.10+
- Git + GitHub repository
- Docker Desktop (for local container run)
- AWS account, IAM user with programmatic access + S3 bucket
- DagsHub account connected to your GitHub repo

---

## 1) 🔧 Initialize the Project (Cookiecutter)

    pip install cookiecutter
    cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
    # Copy folders/files into your main repo, remove the cookiecutter-named folder if needed
    # Fix imports if you changed module names.

---

## 2) 🧪 MLflow on DagsHub (Experiment Tracking)

1. Create a DagsHub repo and connect it to your GitHub repo.  
2. MLflow Tracking URI (for this repo):

       https://dagshub.com/atharvachundurwar841/MLOps-Project.mlflow

3. Install deps:

       pip install mlflow dagshub

4. Auth for CI & scripts (non-interactive) — set these in **GitHub Actions → Secrets/Variables**:

   - `DAGSHUB_USERNAME`  
   - `DAGSHUB_TOKEN` (DagsHub → Settings → Tokens)  
   - `MLFLOW_TRACKING_URI=https://dagshub.com/atharvachundurwar841/MLOps-Project.mlflow`  
   - `MLFLOW_TRACKING_USERNAME=${{ secrets.DAGSHUB_USERNAME }}`  
   - `MLFLOW_TRACKING_PASSWORD=${{ secrets.DAGSHUB_TOKEN }}`  

**DagsHub + MLflow note:** newer MLflow clients call logged-model/registry endpoints that DagsHub doesn’t support yet.  
**Workaround 1 (recommended):** log models as **plain artifacts**:

    from mlflow.sklearn import save_model
    import tempfile, os, mlflow

    def log_model_as_artifacts(model, artifact_path="model"):
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = os.path.join(tmp, artifact_path)
            save_model(sk_model=model, path=local_dir)
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

**Workaround 2:** pin MLflow:

    pip install "mlflow<=2.13.1"

---

## 3) 📦 Data Versioning with DVC

    dvc init
    # (temporary local remote during development)
    mkdir -p local_s3
    dvc remote add -d mylocal local_s3

Add your **IMDB.csv** in `notebooks/`, implement pipeline modules in `src/`, and wire **dvc.yaml** (stages through `model_evaluation` with metrics) + **params.yaml**.

    dvc repro
    dvc status
    git add .
    git commit -m "Add DVC pipeline and stages"
    git push

### Switch DVC remote to S3 (recommended)

    pip install "dvc[s3]" awscli
    aws configure  # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / region
    dvc remote remove mylocal   # optional
    dvc remote add -d myremote s3://<your-bucket-name>
    dvc push    # data and models go to S3

---

## 4) 🧪 Experiments You Can Reproduce

- **BoW vs TF-IDF** comparison (LogReg, NB, XGB, RF, GBDT)  
- **LogReg hyperparameter tuning** (grid over `C`, `penalty`, `solver`)  
- Track **params + metrics** to MLflow and **save model as artifacts** (see note above)

> For unit tests or CI, avoid interactive OAuth: either provide `DAGSHUB_USERNAME/TOKEN` env or fall back to a local MLflow store when `os.getenv("CI")` is set.

---

## 5) 🤖 CI/CD (GitHub Actions)

Minimal CI with Python + DVC + MLflow env:

    # .github/workflows/ci.yaml
    name: CI

    on:
      push:
        branches: [ main ]
      pull_request:

    jobs:
      ci:
        runs-on: ubuntu-latest
        env:
          MLFLOW_TRACKING_URI: https://dagshub.com/atharvachundurwar841/MLOps-Project.mlflow
          MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
          MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
          DAGSHUB_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_DEFAULT_REGION: us-east-1

        steps:
          - uses: actions/checkout@v4

          - uses: actions/setup-python@v5
            with:
              python-version: '3.10'

          - name: Install deps
            run: |
              pip install -r requirements.txt
              pip install "dvc[s3]" awscli

          - name: Reproduce pipeline
            run: dvc repro -v

          - name: Run tests
            run: python -m unittest discover -s tests -p "test_*.py"

> If a module calls `dagshub.init(..., mlflow=True)` at import, guard it for CI or mock it in tests to avoid OAuth prompts.

---

## 6) 🌶️ Flask App (Local Serve)

A minimal API lives in `flask_app/` for quick verification of your trained model.

    cd flask_app
    pip install -r requirements.txt
    python app.py  # app should bind to 0.0.0.0 and chosen port

Recommended binding:

    # flask_app/app.py
    import os
    from flask import Flask
    app = Flask(__name__)

    PORT = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=PORT)

---

## 7) 🐳 Docker (Local Container Run)

We **only copy `flask_app/`** into the image to keep it small. Generate a minimal requirements file inside `flask_app/`:

    pip install pipreqs
    cd flask_app
    pipreqs . --force

Dockerfile (at repo root; copies only `flask_app/`):

    # syntax=docker/dockerfile:1
    FROM python:3.10-slim

    WORKDIR /app
    COPY flask_app/ /app/

    RUN pip install --no-cache-dir -r requirements.txt

    # Let app read a PORT env; default 5001
    ENV PORT=5001

    # Your app should bind to 0.0.0.0:PORT
    CMD ["python", "app.py"]

Build & run:

    # from repo root
    docker build -t atlas:latest .

    # IMPORTANT: pass the required env var your app checks (CAPSTONE_TEST)
    docker run --rm -d --name atlas_test \
      -p 8888:5001 \
      -e CAPSTONE_TEST="<token_or_flag_value>" \
      atlas:latest

    # Check logs and open http://localhost:8888
    docker logs -f atlas_test

> If you map `-p 8888:5001`, ensure the app inside listens on **5001** and **0.0.0.0** (not 127.0.0.1).

---

## 8) 🔐 Environment Variables

Create a local `.env` (don’t commit) for convenience:

    # MLflow / DagsHub
    MLFLOW_TRACKING_URI=https://dagshub.com/atharvachundurwar841/MLOps-Project.mlflow
    MLFLOW_TRACKING_USERNAME=<your_dagshub_user>
    MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
    DAGSHUB_USERNAME=<your_dagshub_user>
    DAGSHUB_TOKEN=<your_dagshub_token>

    # AWS
    AWS_ACCESS_KEY_ID=<...>
    AWS_SECRET_ACCESS_KEY=<...>
    AWS_DEFAULT_REGION=us-east-1

    # Flask/Docker app
    CAPSTONE_TEST=<token_or_flag_value>
    PORT=5001

Use with Docker:

    docker run --rm -d --name atlas_test -p 8888:5001 --env-file .env atlas:latest

---

## 9) ⚙️ Parameters & Pipeline

- **params.yaml** – centralize vectorizer choice (BoW/TF-IDF), model hyperparams (`C`, `penalty`, etc.), dataset paths.  
- **dvc.yaml** – stages: ingestion → preprocessing → features → train → evaluate (metrics artifact).  
  Use `dvc repro` to run end-to-end and `dvc push` to sync to S3.

---

## 🔒 FINAL REMINDER — Rotate/Remove Leaked Credentials

You noted MLflow credentials were committed. **Before publishing or sharing this repo**, remove/rotate any secrets exposed in:

- `exp1.ipynb`  
- `exp2_bow_VS_TfIdf.py`  
- `exp3_lor_bow_hp.py`

**Action items:**

1) Rotate tokens on DagsHub.  
2) Purge credentials from code/notebooks; use env vars instead.  
3) (Optional) Rewrite git history or at least invalidate those tokens.

---

## 📄 License

Add your preferred license (e.g., MIT) in `LICENSE`.