# Reccurent Neural Network for Bitcoin Price predictions

# **Please use the RNN.ipynb file, where all the analysis is made**

---

# Tentative de prédiction du Bitcoin par Analyse Technique et Intelligence Artificielle (RNN - LTSM)

Ce Notebook a pour intention d'explorer les possibilités des réseaux de neurones couplés aux indicateurs de l'analyse technique afin de définir une stratégie et de l'automatiser grâce aux bots de trading.

Ces recherches sont développées en plusieures parties : 
1.   Statistiques descriptives
  *   Comprendre le marché volatil Bitcoin
  *   Déduire des statistiques propres à ce marché
2.   Intégration des Indicateurs
  *   Comprendre leur fontionnement et leur pertinence
  *   Déterminer lesquels utiliser pour de l'algotrading
3.   Contruction du modèle RNN
  *   Ajuster les paramètres et les couches du modele TensorFlow-Keras pour le rendre plus performant
  *   Trouver un modele permettant de prédire à h+24 ou d+2
4.   Robot Trading
  *   Mise en place de la stratégie par scoring
  *   Backtesting sur les données historiques
  *   Optimisation de ses paramètres

Les données sont tirées de https://www.cryptodatadownload.com/

Les commentaires sont en français dans le texte, mais par habitudes le nom des variables et les commentaires dans le code sont en anglais.

---
