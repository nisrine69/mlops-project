#  ImmoPrix — Projet MLOps

##  Objectif
Mettre en œuvre un pipeline **MLOps complet** pour prédire le **prix médian des maisons en Californie**, depuis l’entraînement du modèle jusqu’à son déploiement et sa surveillance en production.

---

##  Données
Dataset **California Housing**  
Features : `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`  
Cible : `MedHouseVal`

---

##  Fonctionnalités clés
- Entraînement de modèles (Linear Regression, Random Forest)
- Suivi des expériences avec **MLflow**
- Analyse des features avec **SHAP**
- API de prédiction avec **FastAPI**
- Conteneurisation avec **Docker**
- CI/CD avec **GitHub Actions**
- Versioning du modèle via **Git LFS**
- Détection du data drift avec **Evidently**

---

##  Installation & exécution

### Prérequis
- Python 3.10
- Git
- Docker
- Poetry
- Git LFS

### Lancer le projet
```bash
git clone https://github.com/nisrine69/mlops-project.git
cd mlops-project
poetry install
poetry run python -m mon_mlops_project.models.train
poetry run uvicorn mon_mlops_project.serving.api:app --reload
```

### Docker
```bash
docker pull ghcr.io/nisrine69/mlops-project/immoprix-api:latest
docker run --rm -p 8000:8000 ghcr.io/nisrine69/mlops-project/immoprix-api:latest
```


## Suivi en production et gestion du data drift


MLOps – Stratégie en cas de data drift significatif (mission 6)
--------------------------------------------------

Si un data drift significatif est détecté par Evidently
(ex: dataset_drift = True ou plus de 30% des variables dérivent),
les actions suivantes sont proposées :

1) Réentraînement automatique du modèle
   - Lancer un nouveau pipeline d'entraînement avec les données
     de production les plus récentes.
   - Enregistrer le nouveau modèle dans MLflow Model Registry
     avec une nouvelle version.

2) Validation avant mise en production
   - Comparer les performances du nouveau modèle avec celui
     actuellement en production (RMSE, MAE, R²).
   - Promouvoir le modèle en Production uniquement s’il
     améliore ou maintient les performances.

3) Réentraînement périodique
   - Mettre en place un réentraînement préventif (mensuel ou hebdomadaire)
     pour anticiper les dérives progressives des données.

4) Supervision et alerting
   - Générer un rapport Evidently HTML à chaque analyse.
   - Alerter l’équipe (email / Slack) en cas de drift critique
     pour validation humaine.

Cette approche garantit la robustesse du modèle face à
l’évolution des données en production.

