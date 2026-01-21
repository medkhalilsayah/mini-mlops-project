import os
import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn


DATA_PATH = "data/raw/heart.csv"


def build_pipeline(C: float, max_iter: int):
    # pipeline preprocessing + modèle
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    def make_preprocessor(X):
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

    return make_preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Cible du dataset (num) -> binaire
    target_col = "num"
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' introuvable. Colonnes: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = (df[target_col] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    make_preprocessor = build_pipeline(args.C, args.max_iter)
    preprocessor = make_preprocessor(X_train)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(C=args.C, max_iter=args.max_iter))
    ])

    # ---- MLflow tracking ----
    mlflow.set_experiment("heart-disease-mlops")

    with mlflow.start_run():
        # Params
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("model", "LogisticRegression")

        # Train
        model.fit(X_train, y_train)

        # Predict + metric
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        print(f"✅ Accuracy test : {acc:.4f}")

        # Artefact 1: confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        os.makedirs("artifacts", exist_ok=True)
        cm_path = "artifacts/confusion_matrix.png"
        plt.title("Confusion Matrix")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path)

        # Artefact 2: modèle (MLflow) + copie joblib
        mlflow.sklearn.log_model(model, artifact_path="model")
        joblib_path = "artifacts/model.joblib"
        joblib.dump(model, joblib_path)
        mlflow.log_artifact(joblib_path)

        print("✅ Run enregistré dans MLflow.")


if __name__ == "__main__":
    main()
