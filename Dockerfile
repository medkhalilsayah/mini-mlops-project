FROM python:3.10-slim

WORKDIR /app

# dépendances système minimales (utile pour sklearn/matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY src ./src

# (Optionnel) documentation
COPY README.md ./README.md

# on ne copie pas data/models ici car on va les monter en volume via docker-compose
EXPOSE 8000 5000
