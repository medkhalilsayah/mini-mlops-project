# Mini MLOps Project — Heart Disease (DVC • MLflow • ZenML • Docker • FastAPI)

Ce projet implémente un mini workflow MLOps de bout en bout :
- **Versioning des données** avec **DVC**
- **Tracking des expériences** avec **MLflow** (baseline + variation)
- **Pipeline** avec **ZenML** (ingest → preprocess → train → evaluate)
- **Serving** via **FastAPI**
- **Déploiement Docker** avec **docker-compose** (MLflow + API)
- **Démo d’inférence** via `curl`

---

## 1) Structure du projet

mini-mlops-project/
├── data/
│ └── raw/
│ ├── heart.csv
│ └── heart.csv.dvc
├── models/
│ └── model.joblib
├── src/
│ ├── train.py
│ ├── train_mlflow.py
│ ├── zenml_pipeline.py
│ └── api.py
├── mlruns/ # tracking MLflow local
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

yaml
Copier le code

---

## 2) Prérequis

- Python 3.8+ (recommandé)
- Git
- Docker + docker-compose
- DVC

---

## 3) Installation (mode local)

### 3.1 Créer un environnement virtuel
```bash
python3 -m venv .venv
source .venv/bin/activate
3.2 Installer les dépendances
bash
Copier le code
pip install --upgrade pip
pip install -r requirements.txt
4) Données — DVC
4.1 Récupérer les données (après clone)
bash
Copier le code
dvc pull
4.2 Vérifier les fichiers
bash
Copier le code
ls data/raw
# heart.csv  heart.csv.dvc
5) Entraînement local (baseline simple)
bash
Copier le code
python src/train.py
Sortie attendue :

Accuracy sur test

Modèle sauvegardé dans models/model.joblib

6) MLflow (livrable : baseline + variation)
6.1 Lancer l’UI MLflow
bash
Copier le code
mlflow ui --backend-store-uri ./mlruns --port 5000
Ouvrir : http://localhost:5000

6.2 Lancer deux runs (baseline + variation)
Run 1 (baseline)

bash
Copier le code
python src/train_mlflow.py --C 1.0 --max_iter 1000
Run 2 (variation)

bash
Copier le code
python src/train_mlflow.py --C 0.1 --max_iter 1000
Dans l’UI MLflow, on doit voir :

2 runs

métrique accuracy

artefacts : confusion_matrix.png + modèle

7) ZenML (pipeline obligatoire)
7.1 Initialiser ZenML
bash
Copier le code
zenml init
7.2 Exécuter le pipeline (2 runs)
Run 1

bash
Copier le code
python src/zenml_pipeline.py --C 1.0 --max_iter 1000
Run 2

bash
Copier le code
python src/zenml_pipeline.py --C 0.1 --max_iter 1000
7.3 Voir les runs (preuve)
bash
Copier le code
zenml pipeline runs list
7.4 Dashboard ZenML (optionnel mais recommandé)
bash
Copier le code
zenml up
Dashboard : http://127.0.0.1:8237

8) API d’inférence (FastAPI)
8.1 Lancer l’API localement
bash
Copier le code
uvicorn src.api:app --host 0.0.0.0 --port 8000
8.2 Tester l’API
Health

bash
Copier le code
curl http://127.0.0.1:8000/health
Predict

bash
Copier le code
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "age": 55,
    "sex": 1,
    "dataset": 0,
    "cp": 0,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalch": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }'
9) Docker (MLflow + API)
9.1 Lancer la stack
bash
Copier le code
docker-compose up -d --build
9.2 Vérifier les conteneurs
bash
Copier le code
docker ps
# mlflow sur :5000, api sur :8000
9.3 Tester

MLflow : http://localhost:5000

API :

curl http://127.0.0.1:8000/health

10) Livrables (checklist)

✅ Lien GitHub / GitLab

✅ Structure + README (ce fichier)

✅ Dockerfile + docker-compose.yml

✅ DVC : fichiers .dvc + preuve dvc pull

✅ MLflow : 2 runs (baseline + variation) + métriques + artefacts

✅ ZenML : pipeline exécuté + preuve (logs / dashboard)

✅ Démo d’inférence : commande curl + sortie

Auteur

Projet réalisé par SAYAH Mohamed Khalil dans le cadre du mini-projet MLOps.