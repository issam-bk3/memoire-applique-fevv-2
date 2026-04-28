# 🛡️ SupplyGuard — Guide de déploiement

## Déploiement en ligne (Streamlit Cloud — GRATUIT)

### Étape 1 — Préparer le dépôt GitHub
1. Créez un compte sur https://github.com (si pas déjà fait)
2. Créez un nouveau dépôt public (ex: `supplygard`)
3. Uploadez les deux fichiers :
   - `app.py`
   - `requirements.txt`

### Étape 2 — Déployer sur Streamlit Cloud
1. Allez sur https://share.streamlit.io
2. Connectez-vous avec votre compte GitHub
3. Cliquez **"New app"**
4. Sélectionnez votre dépôt `supplygard`
5. Fichier principal : `app.py`
6. Cliquez **"Deploy"** → votre app est en ligne en ~3 minutes

**URL résultante** : `https://votrenom-supplygard.streamlit.app`

---

## Déploiement local (test)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Ce qui a changé vs l'ancienne version

### ✅ Interface
- **Dark theme premium** — Design luxury, comparable à Palantir / Tableau
- **Zéro jargon ML** — L'utilisateur voit "Score de risque", pas "UMAP" ou "HDBSCAN"
- **Navigation par onglets** — Vue d'ensemble · Cartographie · Liste · Fiche · Tendances · Rapport

### ✅ Fonctionnalités Power BI
- **KPI cards animées** avec indicateurs de tendance
- **Gauge interactif** du score moyen portefeuille
- **Heatmap Secteur × Région** — croisement géographique
- **Treemap** — taille = volume, couleur = risque
- **Cartographie scatter** — positionnement visuel de chaque fournisseur
- **Violin plots + Box plots** — distributions avancées
- **Radar de profil** par fournisseur
- **Matrice de corrélation** interactive
- **Filtres dynamiques** — alerte, score min/max, secteur
- **Tableau configurable** avec barres de progression intégrées

### ✅ Rapport Exécutif
- Synthèse narrative automatique selon le niveau de risque
- Top 5 fournisseurs prioritaires
- Recommandations stratégiques par niveau
- Export CSV + JSON
- Note méthodologique

### ✅ Multi-utilisateurs simultanés
- Utilisation de `st.session_state` pour isoler les sessions
- Compatible avec le multi-thread de Streamlit Cloud
- Chaque utilisateur a son propre pipeline isolé

---

## Structure des fichiers

```
supplygard/
├── app.py              ← Application principale
├── requirements.txt    ← Dépendances Python
└── README.md           ← Ce fichier
```

---

*Université Mohammed V — FSJES Agdal · Master M.I.E.L · 2025–2026*
