# DST_MLOPS_04_Drift_Exam
Datascientest Cursus MLOPS - Sprint 4 - Drift Monitoring - Exam


## Exécution du script python pour générer les rapports
Depuis le folder contenant le fichier "drift_exam.py":
python3 drift_exam.py


## Questions
### Question après étape 4
*Après l'étape 4 (model drift sur chacune des 3 premières semaines de février), expliquez ce qui a changé au cours des semaines 1, 2 et 3.*
Lorsqu'on passe de la semaine 1 de février à la semaine 3, la Model Quality diminue.
De même, en observant le graphique "Predicted vs Actual in Time", on remarque qu'il y a de plus en plus de sous-estimation, notamment sur le weekend.


### Question après étape 4
*Après l'étape 5 (model drift et target drift sur la semaine la plus défavorable), tirez une conclusion concernant la cause potentielle de la dérive.*
On remarque que la corrélation de la variable d'intérêt réelle "current" (février) avec la température est plus importante que pour la variable d'intérêt de référence (janvier). 
Le poids des deux variables de température est minimisé dans le modèle entraîné sur les données de janvier, entraînant une dérive. On peut aussi noter une sous-estimation par le modèle de l'importance du jour de la semaine. 
La corrélation avec les autres grandeurs (humidité, vent, heure) ne semble pas beaucoup évoluer entre réalité et prédiction.

Note : on peut s'interroger sur le fait d'utiliser les deux variables "température" en même temps. Certes, la différence entre température réelle et température ressentie varie avec le vent, l'humidité, etc, mais au final, n'introduit-on pas de la colinéarité ?


### Question après étape 4
*Après l'étape 6 (data drift sur la 3ème semaine de février), concluez sur la cause potentielle de la dérive.*
Lorsqu'on s'attarde sur les conditions atmosphériques (température ressentie, humidité, vitesse du vent), on remarque une dérive dans la distribution des données, détectée par evidently.
Il semble que le vent et la température soient plus importants en février qu'en janvier alors que l'humidité est plus faible. 

On ne tient pas compte de la détection d'une dérive sur le mois, puisqu'elle est attendue du fait de deux mois différents.

En l'état, j'aurais tendance à préconiser un réentraînement du modèle avec une fenêtre de données peut-être glissante : les données de janvier ne semblent pas les plus adéquates pour prévoir sur le mois de février. 
Il conviendrait d'entraîner le modèle sur 2, 3 ou 4 semaines glissantes et de comparer les résultats. Certes, la diminution de la fenêtre réduit le nombre de données mais si celles-ci sont plus "proches" de ce qu'on cherche à prédire, cela peut amener à de meilleures prédictions.