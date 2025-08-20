import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
import warnings
import tempfile
from mlflow.sklearn import save_model  # <-- added

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/atharvachundurwar841/MLOps-Project.mlflow",
    "dagshub_repo_owner": "atharvachundurwar841",
    "dagshub_repo_name": "MLOps-Project",
    "experiment_name": "Bow vs TfIdf"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

# ========================== LOAD & PREPROCESS DATA ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'negative': 0, 'positive': 1}).infer_objects(copy=False)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ========================== FEATURE ENGINEERING ==========================
VECTORIZERS = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),  # keep defaults; max_iter can be set if needed
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# ========================== SAFE MODEL LOGGING (NO REGISTRY ENDPOINTS) ==========================
def log_model_as_artifacts(model, artifact_path="model"):
    """
    Save the trained model locally and upload it as plain artifacts.
    Avoids MLflow 'logged models' endpoints that DagsHub doesn't support.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = os.path.join(tmpdir, artifact_path)
        save_model(sk_model=model, path=local_dir)
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

# ========================== TRAIN & EVALUATE MODELS ==========================
def train_and_evaluate(df):
    with mlflow.start_run(run_name="All Experiments") as parent_run:
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                    try:
                        # Feature extraction
                        X = vectorizer.fit_transform(df['review'])
                        y = df['sentiment']
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=CONFIG["test_size"], random_state=42
                        )

                        # Log preprocessing parameters
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })

                        # Train model
                        model = algorithm
                        model.fit(X_train, y_train)

                        # Log model parameters
                        log_model_params(algo_name, model)

                        # Evaluate model
                        y_pred = model.predict(X_test)
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred),
                            "f1_score": f1_score(y_test, y_pred)
                        }
                        mlflow.log_metrics(metrics)

                        # ---- SAFE LOGGING (replaces mlflow.sklearn.log_model(...)) ----
                        log_model_as_artifacts(model, artifact_path="model")
                        # ----------------------------------------------------------------

                        # Print results for verification
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

                    except Exception as e:
                        print(f"Error in training {algo_name} with {vec_name}: {e}")
                        mlflow.log_param("error", str(e))

def log_model_params(algo_name, model):
    """Logs hyperparameters of the trained model to MLflow."""
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log["C"] = getattr(model, "C", None)
        params_to_log["penalty"] = getattr(model, "penalty", None)
        params_to_log["solver"] = getattr(model, "solver", None)
        params_to_log["max_iter"] = getattr(model, "max_iter", None)
    elif algo_name == 'MultinomialNB':
        params_to_log["alpha"] = getattr(model, "alpha", None)
        params_to_log["fit_prior"] = getattr(model, "fit_prior", None)
    elif algo_name == 'XGBoost':
        params_to_log["n_estimators"] = getattr(model, "n_estimators", None)
        params_to_log["learning_rate"] = getattr(model, "learning_rate", None)
        params_to_log["max_depth"] = getattr(model, "max_depth", None)
        params_to_log["subsample"] = getattr(model, "subsample", None)
        params_to_log["colsample_bytree"] = getattr(model, "colsample_bytree", None)
    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = getattr(model, "n_estimators", None)
        params_to_log["max_depth"] = getattr(model, "max_depth", None)
        params_to_log["max_features"] = getattr(model, "max_features", None)
        params_to_log["min_samples_split"] = getattr(model, "min_samples_split", None)
    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = getattr(model, "n_estimators", None)
        params_to_log["learning_rate"] = getattr(model, "learning_rate", None)
        params_to_log["max_depth"] = getattr(model, "max_depth", None)

    # Remove None values (cleaner UI)
    params_to_log = {k: v for k, v in params_to_log.items() if v is not None}
    if params_to_log:
        mlflow.log_params(params_to_log)

# ========================== EXECUTION ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)
