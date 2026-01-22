import os
import argparse
import pandas as pd
from typing import Tuple

from zenml import step, pipeline
from zenml.client import Client

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/raw/heart.csv"


@step
def ingest_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


@step
def preprocess_data(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target_col = "num"
    if target_col not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_col}' introuvable. Colonnes: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = (df[target_col] > 0).astype(int)  # binarisation

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000
) -> Pipeline:
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(C=C, max_iter=max_iter))
    ])

    model.fit(X_train, y_train)
    return model


@step
def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000
) -> float:
    """
    Évalue le modèle + log dans MLflow (métriques + params).
    On récupère le tracking_uri depuis le tracker ZenML actif.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ ZenML - Accuracy test: {acc:.4f}")

    # --- MLflow logging (pour voir l'historique des runs ZenML dans MLflow) ---
    client = Client()
    tracker = client.active_stack.experiment_tracker

    if tracker is not None:
        tracking_uri = tracker.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("zenml-heart-pipeline")

        with mlflow.start_run(run_name=f"zenml_C={C}_maxiter={max_iter}"):
            mlflow.log_param("C", C)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_metric("accuracy", float(acc))

    return acc


@pipeline
def heart_disease_pipeline(C: float = 1.0, max_iter: int = 1000):
    df = ingest_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train, C=C, max_iter=max_iter)
    _ = evaluate_model(model, X_test, y_test, C=C, max_iter=max_iter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    # Dans cette version de ZenML, appeler le pipeline l’exécute directement
    heart_disease_pipeline(C=args.C, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
