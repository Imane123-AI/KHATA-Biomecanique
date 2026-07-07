
# KHATA — Analyse Biomécanique de la Marche Humaine

> Projet Master 1 Machine Learning — Apprentissage par Renforcement Inverse (MaxEnt IRL)



## Description

Application web d'analyse biomécanique de la marche humaine par IRL.

- Upload vidéo → MediaPipe → extraction angles articulaires

- Modèle MaxEnt IRL (RewardNetwork) → score de normalité 0-100

- Dashboard interactif + rapport clinique automatique

## Résultats clés

- R_expert / R_random = ×3.07

- Vitesse optimale apprise : 1.2 m/s (Winter, 2009)

- Hanche dominante : 32.5% (hiérarchie biomécanique confirmée)

## Stack technique

- Backend : Python, Flask, PyTorch, MediaPipe

- Frontend : React.js, Recharts

- Dataset : Figshare 2022 — 9 sujets, 251 trajectoires, 190 618 frames

## Lancer le projet

\\\ash

# Backend

pip install -r requirements.txt

python api.py

# Frontend

cd frontend

npm install

npm start

\\\

## Perspectives

- Déploiement cloud (Heroku / AWS)

- Chatbot conversationnel intégré

- Extension à une population pathologique

