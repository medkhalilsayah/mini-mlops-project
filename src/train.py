import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


DATA_PATH = "data/raw/heart.csv"
MODEL_PATH = "models/model.joblib"


def main():
    # 1. Vérifier la présence du dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    # 2. Charger les données
    df = pd.read_csv(DATA_PATH)

    # 3. Définir la colonne cible (spécifique à ce dataset)
    target_col = "num"

    if target_col not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_col}' introuvable. Colonnes disponibles : {list(df.columns)}"
        )

    # 4. Séparer features / target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5. Binarisation de la cible
    # 0 = pas de maladie, >0 = maladie
    y = (y > 0).astype(int)

    # 6. Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 7. Détection des types de colonnes
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    # 8. Prétraitement
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

    # 9. Modèle (baseline)
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # 10. Entraînement
    model.fit(X_train, y_train)

    # 11. Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy sur le jeu de test : {acc:.4f}")

    # 12. Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Modèle sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()
