# Projet : Détectez des faux billets
Projet de <i>machine learning</i> du parcours Data analyst d'OpenClassrooms

## Scénario du projet
<img src="imgs/oncfm.png"/>

Dans le scénario du projet, l’Organisation nationale de lutte contre le faux-monnayage (ONCFM) nous a chargé de mettre au point un algorithme capable de détecter les faux billets à partir de leurs dimensions géométriques.

## Commentaires

### Organisation des fichiers du projet
Les analyses ont été effectuées avec Python en utilisant Jupyter Notebook. Le projet a été divisé en deux notebooks qui contiennent :
* [l'imputation des données](https://github.com/vincent71219291/oc_counterfeit_notes_detection/blob/main/oc_counterfeit_notes_detection_01.ipynb) ;
* la mise au point de [l'algorithme de détection](https://github.com/vincent71219291/oc_counterfeit_notes_detection/blob/main/oc_counterfeit_notes_detection_02.ipynb) des faux billets.

Le fichier [`P10_functions.py`](https://github.com/vincent71219291/oc_counterfeit_notes_detection/blob/main/scripts/P10_functions.py) (dans le dossier `scripts`) contient des fonctions personnalisées dont la programmation représente une grosse partie du travail effectué pour ce projet.

<b>Note :</b> Les données du projet sont un peu trop "parfaites" (on obtient un score F1 de 0,99 avec une régression logistique sans transformer ou sélectionner les variables, et sans toucher aux hyperparamètres par défaut). En ce sens, le test de différents algorithmes ou l'utilisation de pipelines pourraient apparaître superflus, mais ils s'inscrivent dans une démarche d'apprentissage.

## Compétences :

- Réaliser une régression linéaire (méthode des moindres carrés et régression pondérée)
- Réaliser une classification binaire avec différents algorithmes
- Evaluer la performance de différents algorithmes de classification
- Créer un pipeline et sélectionner automatiquement le pipeline le plus performant