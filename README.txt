==============================================================
🧠 PROJET DE SEGMENTATION PANOPTIQUE - MASTER SI T 2025
Auteur : Ahmed Walid BOUANZOUL 21318873
==============================================================

📘 Description :
----------------
Ce projet met en œuvre un pipeline complet de SEGMENTATION PANOPTIQUE 
basé sur l’approche de Kirillov et al. (CVPR 2019).
L’objectif est de combiner la segmentation d’instances (Things)
et la segmentation sémantique (Stuff) pour obtenir une vision globale de la scène.

Deux versions ont été développées :
 - Version 1 : Mask R-CNN + DeepLab Cityscapes
 - Version 2 : Mask R-CNN + DeepLab COCO (améliorée)

==============================================================
 Contenu du dossier :
-----------------------
images/         → Images de test (ville, foule, intérieur, nature)
dnn/            → Modèles pré-entraînés (.pb, .pbtxt)
results/        → Résultats de la première version
results_v2/     → Résultats de la version améliorée
main_v2.py      → Script principal à exécuter
compare_pq.py   → Script de comparaison PQ (v1 vs v2)
README.txt      → Ce fichier d’explication

==============================================================
⚙️ Dépendances :
----------------
Python >= 3.10
Modules nécessaires :
 - opencv-python
 - numpy
 - torch
 - torchvision
 - matplotlib

Installation rapide :
> pip install opencv-python numpy torch torchvision matplotlib

==============================================================
 Exécution :
--------------
Lancer le script principal :
   > python main.py

→ Les résultats seront affichés à l’écran et enregistrés dans le dossier "results/".

apres 
   > Lancer la version 2 (améliorée COCO) :
        > python main_v2.py
        → Les résultats seront affichés à l’écran et enregistrés dans le dossier "results_v2/".

Comparer les deux versions :
   > python compare_pq.py
   → Compare automatiquement les images communes entre "results/" et "results_v2/"
     et génère un graphique : comparaison_PQ_V1_vs_V2.png

=============================================================
Résultats :
--------------
- Version 2 obtient un gain moyen de +8% de Panoptic Quality (PQ)
  grâce à un modèle plus généraliste et une fusion adaptative.
- Les résultats améliorent particulièrement les scènes d’intérieur
  et les environnements naturels.

==============================================================

