# mlops-project


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

