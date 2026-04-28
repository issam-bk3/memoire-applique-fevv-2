import warnings
warnings.filterwarnings("ignore")

import io
import time
import json
import base64
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

import umap
import hdbscan
import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SupplyGuard · Gestion des Risques Fournisseurs",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — LUXURY DARK THEME
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #080C14 !important;
    color: #E8EDF5 !important;
    font-family: 'Sora', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #0D1420 !important; border-right: 1px solid #1E2D45; }
[data-testid="stMainBlockContainer"] { padding: 0 2rem 4rem !important; max-width: 100% !important; }
section.main > div { padding-top: 0 !important; }

/* ── Hide Streamlit clutter ── */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stFileUploader"] label { color: #8BA3C7 !important; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Sora', sans-serif !important; font-weight: 700 !important; }
p, span, div { font-family: 'Sora', sans-serif !important; }
code { font-family: 'JetBrains Mono', monospace !important; }

/* ── Top Navigation Bar ── */
.sg-topbar {
    position: sticky; top: 0; z-index: 999;
    background: rgba(8, 12, 20, 0.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(56, 100, 180, 0.25);
    padding: 0 2rem;
    display: flex; align-items: center; justify-content: space-between;
    height: 64px; margin: 0 -2rem 2rem;
}
.sg-logo {
    display: flex; align-items: center; gap: 12px;
}
.sg-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #1E5EFF, #0A3CBF);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; box-shadow: 0 0 20px rgba(30,94,255,0.4);
}
.sg-logo-text { font-size: 1.1rem; font-weight: 800; color: #E8EDF5; letter-spacing: -0.02em; }
.sg-logo-sub  { font-size: 0.7rem; color: #4A6FA5; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }
.sg-nav-badge {
    background: rgba(30,94,255,0.15); border: 1px solid rgba(30,94,255,0.3);
    color: #6EA3FF; padding: 4px 12px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
}

/* ── Hero Upload Section ── */
.sg-hero {
    background: linear-gradient(135deg, #0D1830 0%, #0A1525 50%, #080C14 100%);
    border: 1px solid rgba(30,94,255,0.2);
    border-radius: 20px; padding: 60px 40px;
    text-align: center; margin: 2rem 0;
    position: relative; overflow: hidden;
}
.sg-hero::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(30,94,255,0.12) 0%, transparent 70%);
}
.sg-hero-title {
    font-size: 2.8rem; font-weight: 800; color: #E8EDF5;
    letter-spacing: -0.04em; line-height: 1.1; margin: 0 0 12px;
}
.sg-hero-title span { color: #1E5EFF; }
.sg-hero-sub { font-size: 1rem; color: #4A6FA5; font-weight: 400; max-width: 500px; margin: 0 auto 40px; }
.sg-hero-steps {
    display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;
    margin-bottom: 40px;
}
.sg-hero-step {
    display: flex; align-items: center; gap: 10px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 12px 20px;
    font-size: 0.82rem; color: #8BA3C7; font-weight: 500;
}
.sg-hero-step-num {
    width: 24px; height: 24px; border-radius: 50%;
    background: rgba(30,94,255,0.2); color: #6EA3FF;
    font-size: 0.72rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

/* ── KPI Cards ── */
.sg-kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin: 2rem 0; }
.sg-kpi {
    background: #0D1420; border: 1px solid #1E2D45;
    border-radius: 16px; padding: 20px 18px;
    position: relative; overflow: hidden;
    transition: all 0.3s ease;
}
.sg-kpi::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.sg-kpi.total::before  { background: linear-gradient(90deg, #1E5EFF, #6EA3FF); }
.sg-kpi.green::before  { background: linear-gradient(90deg, #00C566, #30E88A); }
.sg-kpi.orange::before { background: linear-gradient(90deg, #FF8C00, #FFB347); }
.sg-kpi.red::before    { background: linear-gradient(90deg, #FF2D55, #FF6B81); }
.sg-kpi.score::before  { background: linear-gradient(90deg, #9B59B6, #C39BD3); }
.sg-kpi-label { font-size: 0.7rem; color: #4A6FA5; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
.sg-kpi-value { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1; }
.sg-kpi.total  .sg-kpi-value { color: #6EA3FF; }
.sg-kpi.green  .sg-kpi-value { color: #00C566; }
.sg-kpi.orange .sg-kpi-value { color: #FF8C00; }
.sg-kpi.red    .sg-kpi-value { color: #FF2D55; }
.sg-kpi.score  .sg-kpi-value { color: #C39BD3; }
.sg-kpi-sub { font-size: 0.72rem; color: #2A3F5F; margin-top: 6px; font-weight: 500; }
.sg-kpi-trend { position: absolute; top: 18px; right: 16px; font-size: 1.2rem; opacity: 0.4; }

/* ── Alert Badges ── */
.badge-rouge  { background:rgba(255,45,85,0.15);  border:1px solid rgba(255,45,85,0.4);  color:#FF2D55; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; display:inline-block; }
.badge-orange { background:rgba(255,140,0,0.15);  border:1px solid rgba(255,140,0,0.4);  color:#FF8C00; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; display:inline-block; }
.badge-vert   { background:rgba(0,197,102,0.12);  border:1px solid rgba(0,197,102,0.35); color:#00C566; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; display:inline-block; }

/* ── Section Header ── */
.sg-section {
    display: flex; align-items: center; gap: 12px;
    padding: 24px 0 16px; border-bottom: 1px solid #1E2D45; margin-bottom: 20px;
}
.sg-section-icon {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center; font-size: 16px;
}
.sg-section-title { font-size: 1rem; font-weight: 700; color: #E8EDF5; }
.sg-section-sub   { font-size: 0.75rem; color: #4A6FA5; margin-top: 2px; }

/* ── Cards ── */
.sg-card {
    background: #0D1420; border: 1px solid #1E2D45;
    border-radius: 16px; padding: 24px; margin-bottom: 16px;
}
.sg-card-title { font-size: 0.85rem; font-weight: 700; color: #8BA3C7; margin-bottom: 16px; letter-spacing: 0.04em; text-transform: uppercase; }

/* ── Alert Panel ── */
.sg-alert {
    border-radius: 12px; padding: 16px 20px; margin: 12px 0;
    display: flex; align-items: flex-start; gap: 16px;
    border: 1px solid;
}
.sg-alert.critical { background: rgba(255,45,85,0.08);  border-color: rgba(255,45,85,0.3); }
.sg-alert.warning  { background: rgba(255,140,0,0.08);  border-color: rgba(255,140,0,0.3); }
.sg-alert.safe     { background: rgba(0,197,102,0.06);  border-color: rgba(0,197,102,0.25); }
.sg-alert-ico { font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }
.sg-alert-title { font-size: 0.85rem; font-weight: 700; margin-bottom: 4px; }
.sg-alert.critical .sg-alert-title { color: #FF2D55; }
.sg-alert.warning  .sg-alert-title { color: #FF8C00; }
.sg-alert.safe     .sg-alert-title { color: #00C566; }
.sg-alert-body { font-size: 0.78rem; color: #8BA3C7; line-height: 1.6; }

/* ── Score Bar ── */
.score-bar-wrap { background: #131E30; border-radius: 6px; height: 8px; overflow: hidden; margin: 4px 0 8px; }
.score-bar-fill { height: 8px; border-radius: 6px; transition: width 0.8s ease; }

/* ── Supplier Row ── */
.sup-row {
    background: #0D1420; border: 1px solid #1E2D45; border-radius: 12px;
    padding: 14px 20px; margin: 6px 0;
    display: flex; align-items: center; gap: 16px;
    transition: all 0.2s; cursor: pointer;
}
.sup-row:hover { border-color: rgba(30,94,255,0.4); background: #111827; }
.sup-name { font-size: 0.88rem; font-weight: 700; color: #E8EDF5; flex: 1; }
.sup-meta { font-size: 0.72rem; color: #4A6FA5; }
.sup-score-box {
    font-size: 1rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;
    min-width: 56px; text-align: right;
}

/* ── Progress ring ── */
.ring-wrap { display: flex; align-items: center; justify-content: center; }

/* ── Tabs ── */
[data-testid="stTabs"] { margin-top: 0; }
[data-baseweb="tab-list"] {
    background: #0D1420 !important;
    border-radius: 12px !important; padding: 4px !important;
    border: 1px solid #1E2D45 !important;
    gap: 2px !important;
}
[data-baseweb="tab"] {
    color: #4A6FA5 !important; font-family: 'Sora', sans-serif !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    border-radius: 8px !important; padding: 8px 16px !important;
    transition: all 0.2s !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #1E2D45 !important; color: #E8EDF5 !important;
}
[data-testid="stTabContent"] { padding: 16px 0 0 !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1E5EFF, #0A3CBF) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important; font-size: 0.85rem !important;
    padding: 12px 24px !important; width: 100% !important;
    box-shadow: 0 4px 24px rgba(30,94,255,0.3) !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 32px rgba(30,94,255,0.5) !important;
}
[data-testid="stDownloadButton"] > button {
    background: rgba(30,94,255,0.1) !important;
    border: 1px solid rgba(30,94,255,0.3) !important;
    color: #6EA3FF !important; border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important; font-weight: 600 !important;
    font-size: 0.8rem !important; width: 100% !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] .st-bd { color: #1E5EFF !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0D1420; border: 1px solid #1E2D45;
    border-radius: 12px; padding: 16px;
}
[data-testid="stMetricLabel"] { color: #4A6FA5 !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #E8EDF5 !important; font-weight: 700 !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #0D1420 !important; border: 1px solid #1E2D45 !important;
    border-radius: 10px !important; color: #E8EDF5 !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div > div {
    background: #0D1420 !important; border: 1px solid #1E2D45 !important;
    border-radius: 10px !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] > div {
    background: #0D1420 !important; border-radius: 12px !important;
    border: 1px solid #1E2D45 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] > section {
    background: #0D1420 !important; border: 2px dashed #1E2D45 !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"] > section:hover {
    border-color: rgba(30,94,255,0.4) !important;
    background: rgba(30,94,255,0.05) !important;
}

/* ── Info/Warning boxes ── */
[data-testid="stAlert"] {
    background: rgba(30,94,255,0.08) !important;
    border: 1px solid rgba(30,94,255,0.2) !important;
    border-radius: 12px !important;
    color: #8BA3C7 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0D1420 !important; border: 1px solid #1E2D45 !important;
    border-radius: 12px !important;
}
[data-testid="stExpanderDetails"] { background: #0D1420 !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #1E5EFF, #6EA3FF) !important;
    border-radius: 999px !important;
}
[data-testid="stProgress"] > div {
    background: #131E30 !important; border-radius: 999px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #6EA3FF !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] .stMarkdown h3 { color: #8BA3C7 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stSidebar"] hr { border-color: #1E2D45 !important; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] span { color: #8BA3C7 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: #1E2D45; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2A3F5F; }

/* ── Plotly charts dark ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Animate KPIs ── */
@keyframes fadeInUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
.sg-kpi { animation: fadeInUp 0.5s ease both; }
.sg-kpi:nth-child(1) { animation-delay: 0.05s; }
.sg-kpi:nth-child(2) { animation-delay: 0.1s; }
.sg-kpi:nth-child(3) { animation-delay: 0.15s; }
.sg-kpi:nth-child(4) { animation-delay: 0.2s; }
.sg-kpi:nth-child(5) { animation-delay: 0.25s; }

/* ── Risk Gauge ── */
.gauge-wrap { text-align: center; padding: 16px; }
.gauge-label { font-size: 0.7rem; color: #4A6FA5; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 8px; }

/* ── Timeline ── */
.timeline-item {
    display: flex; gap: 16px; padding: 12px 0;
    border-bottom: 1px solid #1E2D45;
}
.timeline-dot {
    width: 10px; height: 10px; border-radius: 50%;
    flex-shrink: 0; margin-top: 5px;
}
.timeline-content { flex: 1; }
.timeline-title { font-size: 0.82rem; font-weight: 600; color: #E8EDF5; }
.timeline-sub { font-size: 0.72rem; color: #4A6FA5; margin-top: 2px; }

/* ── Heatmap table ── */
.heatmap-cell {
    width: 32px; height: 32px; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68rem; font-weight: 700; cursor: pointer;
    transition: transform 0.15s;
}
.heatmap-cell:hover { transform: scale(1.2); z-index: 10; position: relative; }

/* ── Executive Report ── */
.exec-section {
    border-left: 3px solid #1E5EFF; padding: 12px 20px;
    margin: 12px 0; background: rgba(30,94,255,0.04);
    border-radius: 0 10px 10px 0;
}
.exec-section h4 { color: #6EA3FF; font-size: 0.85rem; margin: 0 0 8px; }
.exec-section p { color: #8BA3C7; font-size: 0.8rem; line-height: 1.7; margin: 0; }

/* ── Loading overlay ── */
.loading-overlay {
    position: fixed; inset: 0; z-index: 9999;
    background: rgba(8,12,20,0.92); backdrop-filter: blur(10px);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 24px;
}
.loading-title { font-size: 1.4rem; font-weight: 700; color: #E8EDF5; }
.loading-sub   { font-size: 0.85rem; color: #4A6FA5; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════
EXCL = [
    "ID_Fournisseur","Nom_Fournisseur","Secteur","Region_Maroc",
    "Cluster_Reel","Note_Risque_Pays","Certification",
    "Niveau_Alerte","Priorite_Action","Score_Risque",
    "Alerte_ML","Priorite_ML","Cluster_HDBSCAN",
]
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Sora, sans-serif", color="#8BA3C7", size=11),
    gridcolor="#1E2D45",
    zerolinecolor="#1E2D45",
    margin=dict(l=16, r=16, t=44, b=16),
)

def apply_theme(fig, title="", height=360):
    fig.update_layout(
        **CHART_THEME,
        title=dict(text=title, font=dict(size=13, color="#E8EDF5"), x=0.01, xanchor="left"),
        height=height,
        xaxis=dict(gridcolor="#1E2D45", zerolinecolor="#1E2D45"),
        yaxis=dict(gridcolor="#1E2D45", zerolinecolor="#1E2D45"),
    )
    return fig

def risk_color(score, seuil_v, seuil_o):
    if score <= seuil_v:   return "#00C566"
    if score <= seuil_o:   return "#FF8C00"
    return "#FF2D55"

def risk_label(score, seuil_v, seuil_o):
    if score <= seuil_v:   return ("Faible", "vert")
    if score <= seuil_o:   return ("Modéré", "orange")
    return ("Critique", "rouge")

# ═══════════════════════════════════════════════════════════════
#  TOP NAV
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sg-topbar">
  <div class="sg-logo">
    <div class="sg-logo-icon">🛡️</div>
    <div>
      <div class="sg-logo-text">SupplyGuard</div>
      <div class="sg-logo-sub">Gestion des Risques Fournisseurs</div>
    </div>
  </div>
  <div style="display:flex;gap:10px;align-items:center">
    <div class="sg-nav-badge">Intelligence Artificielle</div>
    <div class="sg-nav-badge">FSJES Agdal · MIEL 2025–26</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR — SETTINGS (collapsée par défaut, épurée)
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration avancée")
    st.markdown("---")
    st.markdown("#### Sensibilité du modèle")
    p_contam = st.slider("Taux d'anomalies détectées (%)", 1, 20, 5) / 100
    st.caption("Plus élevé = plus de fournisseurs signalés")
    st.markdown("#### Seuils d'alerte")
    p_seuil_vert   = st.slider("Seuil Vert → Orange", 10, 45, 29)
    p_seuil_orange = st.slider("Seuil Orange → Rouge", 40, 80, 59)
    st.markdown(f"""
<div style='background:#131E30;border-radius:8px;padding:10px;font-size:0.75rem;color:#8BA3C7;margin-top:4px'>
🟢 0–{p_seuil_vert} · 🟠 {p_seuil_vert+1}–{p_seuil_orange} · 🔴 {p_seuil_orange+1}–100
</div>""", unsafe_allow_html=True)
    st.markdown("#### Pondération du score")
    p_w_cl = st.slider("Comportement groupe (%)",   10, 60, 35) / 100
    p_w_an = st.slider("Déviance comportementale (%)", 10, 60, 30) / 100
    p_w_te = st.slider("Évolution temporelle (%)",   5, 40, 20) / 100
    p_w_dt = max(0.0, 1.0 - p_w_cl - p_w_an - p_w_te)
    p_shap = st.checkbox("Calculer les facteurs d'influence", value=True)
    p_shap_top = st.slider("Nombre de facteurs affichés", 5, 25, 10)
    st.markdown("---")
    st.caption("Université Mohammed V · Rabat")

# ═══════════════════════════════════════════════════════════════
#  HERO + UPLOAD
# ═══════════════════════════════════════════════════════════════
if "df_res" not in st.session_state:
    st.markdown("""
    <div class="sg-hero">
      <div class="sg-hero-title">Évaluez vos fournisseurs<br><span>en quelques secondes</span></div>
      <div class="sg-hero-sub">Importez votre fichier de données et obtenez un tableau de bord complet de gestion des risques, prêt à partager.</div>
      <div class="sg-hero-steps">
        <div class="sg-hero-step"><div class="sg-hero-step-num">1</div>Importez votre fichier Excel ou CSV</div>
        <div class="sg-hero-step"><div class="sg-hero-step-num">2</div>L'IA analyse et score chaque fournisseur</div>
        <div class="sg-hero-step"><div class="sg-hero-step-num">3</div>Explorez votre tableau de bord interactif</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

fichier = st.file_uploader(
    "Déposez votre fichier ici (.xlsx ou .csv)",
    type=["xlsx", "csv"],
    label_visibility="visible",
)

if fichier is None:
    if "df_res" not in st.session_state:
        st.stop()

# ═══════════════════════════════════════════════════════════════
#  DATA READING
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def lire(contenu, nom):
    if nom.endswith(".xlsx"):
        df_raw = pd.read_excel(io.BytesIO(contenu), header=None, nrows=15)
        header_row, best_score = 0, 0
        for i in range(len(df_raw)):
            row = df_raw.iloc[i]
            score = row.apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 1).sum()
            if score > best_score:
                best_score = score; header_row = i
        df = pd.read_excel(io.BytesIO(contenu), header=header_row)
    else:
        df = pd.read_csv(io.BytesIO(contenu))
    return df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

if fichier is not None:
    with st.spinner("Lecture du fichier…"):
        df = lire(fichier.read(), fichier.name)

    features = [c for c in df.columns
                if c not in EXCL and pd.api.types.is_numeric_dtype(df[c])]

    if len(features) == 0:
        st.error("❌ Aucune variable numérique détectée.")
        st.stop()

    X_brut = df[features].copy().apply(pd.to_numeric, errors="coerce").astype("float64")

    col_info, col_btn = st.columns([3, 1])
    col_info.success(f"✅ **{fichier.name}** · {df.shape[0]} fournisseurs · {len(features)} indicateurs")
    run = col_btn.button("🚀 Lancer l'analyse", type="primary")

    if not run and "df_res" not in st.session_state:
        with st.expander("👁️ Aperçu des données"):
            st.dataframe(df.head(), use_container_width=True)
        st.stop()

# ═══════════════════════════════════════════════════════════════
#  PIPELINE ML — SILENT (no ML jargon shown to user)
# ═══════════════════════════════════════════════════════════════
if fichier is not None and (run or "df_res" not in st.session_state):

    prog_container = st.empty()
    with prog_container.container():
        barre = st.progress(0, "⏳ Analyse en cours… (1/5) Préparation des données")

    t0 = time.time()

    # 1 · Preprocessing
    imputer = SimpleImputer(strategy="median")
    scaler  = RobustScaler()
    X_imp   = imputer.fit_transform(X_brut.values)
    X_sc    = scaler.fit_transform(X_imp)
    barre.progress(15, "⏳ Analyse en cours… (2/5) Identification des profils")

    # 2 · PCA
    pca   = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    # 3 · UMAP (silent)
    X_u2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_pca)
    X_u3 = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_pca)
    barre.progress(40, "⏳ Analyse en cours… (3/5) Segmentation des fournisseurs")

    # 4 · HDBSCAN (silent)
    cl       = hdbscan.HDBSCAN(min_cluster_size=max(3,int(len(df)*0.04)), min_samples=max(1,int(len(df)*0.02)), metric="euclidean", prediction_data=True)
    labels   = cl.fit_predict(X_u3)
    n_cl     = len(set(labels)) - (1 if -1 in labels else 0)
    n_br     = int((labels == -1).sum())
    mask     = labels != -1
    barre.progress(56, "⏳ Analyse en cours… (4/5) Détection des anomalies")

    # 5 · Isolation Forest (silent)
    iso  = IsolationForest(n_estimators=200, contamination=p_contam, random_state=42, n_jobs=-1)
    iso.fit(X_sc)
    if_r = iso.decision_function(X_sc)
    if_sc = 1 - (if_r - if_r.min()) / (if_r.max() - if_r.min())
    barre.progress(68, "⏳ Analyse en cours… (5/5) Calcul des scores de risque")

    # 6 · VAE (silent)
    class VAE(nn.Module):
        def __init__(self, d, l=8):
            super().__init__()
            h = max(d // 2, l * 4)
            self.enc = nn.Sequential(nn.Linear(d,h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1), nn.Dropout(0.2), nn.Linear(h,h//2), nn.LeakyReLU(0.1))
            self.mu  = nn.Linear(h//2, l); self.lv = nn.Linear(h//2, l)
            self.dec = nn.Sequential(nn.Linear(l,h//2), nn.LeakyReLU(0.1), nn.Linear(h//2,h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1), nn.Linear(h,d))
        def forward(self, x):
            h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
            z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
            return self.dec(z), mu, lv

    Xn  = torch.FloatTensor(X_sc[labels != -1])
    ld  = DataLoader(TensorDataset(Xn), batch_size=32, shuffle=True)
    vae = VAE(X_sc.shape[1])
    opt = optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    for _ in range(60):
        for (b,) in ld:
            opt.zero_grad(); r, mu, lv = vae(b)
            loss = nn.functional.mse_loss(r,b,reduction="sum") - 5e-4*torch.sum(1+lv-mu.pow(2)-lv.exp())
            loss.backward(); opt.step()
    vae.eval()
    with torch.no_grad():
        Xt = torch.FloatTensor(X_sc)
        r_all, mu_all, lv_all = vae(Xt)
        ve = nn.functional.mse_loss(r_all, Xt, reduction="none").mean(dim=1).numpy()
    vae_sc = (ve - ve.min()) / (ve.max() - ve.min())
    anom   = (if_sc + vae_sc) / 2

    # 7 · Score composite
    cr = {c: float(if_sc[labels==c].mean()) if c!=-1 else 1.0 for c in set(labels)}
    cc = np.array([cr[c] for c in labels])

    def gcol(n): return df[n].fillna(0).astype(float).values if n in df.columns else np.zeros(len(df))
    derive = (0.4*np.clip(gcol("PSI_Score")/0.5,0,1) + 0.4*gcol("Changepoint_PELT") + 0.2*np.clip(np.abs(gcol("Tendance_OTD_6M"))/10,0,1))

    score_100 = np.clip((p_w_cl*cc + p_w_an*anom + p_w_te*derive + p_w_dt*vae_sc)*100, 0, 100)

    def alerte(s):
        if s <= p_seuil_vert: return "🟢 Vert"
        if s <= p_seuil_orange: return "🟠 Orange"
        return "🔴 Rouge"

    alertes  = np.array([alerte(s) for s in score_100])
    n_vert   = int((alertes=="🟢 Vert").sum())
    n_orange = int((alertes=="🟠 Orange").sum())
    n_rouge  = int((alertes=="🔴 Rouge").sum())

    # 8 · Build result DF
    df_res = df.copy()
    df_res["_Cluster"]       = labels
    df_res["_Score_IF"]      = np.round(if_sc*100, 1)
    df_res["_Score_VAE"]     = np.round(vae_sc*100, 1)
    df_res["_Score_Anomalie"]= np.round(anom*100, 1)
    df_res["Score_Risque"]   = np.round(score_100, 1)
    df_res["Niveau_Alerte"]  = alertes
    df_res["Priorite"]       = ["Action immédiate" if a=="🔴 Rouge" else "Surveillance" if a=="🟠 Orange" else "Standard" for a in alertes]
    df_res["_U1"]            = X_u2[:, 0]
    df_res["_U2"]            = X_u2[:, 1]

    # 9 · SHAP (silent)
    shap_values, shap_df = None, None
    if p_shap:
        try:
            exp = shap.TreeExplainer(iso)
            shap_values = exp.shap_values(X_sc)
            ma = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({"Variable":features,"SHAP_abs":ma,"SHAP_pct":(ma/ma.sum()*100).round(1)}).sort_values("SHAP_abs",ascending=False).reset_index(drop=True)
            df_res["_Top_Facteur"] = [features[int(np.argmax(np.abs(shap_values[i])))] for i in range(len(df))]
        except: pass

    t_total = time.time() - t0
    barre.progress(100, f"✅ Analyse terminée en {t_total:.0f} secondes")
    time.sleep(0.8)
    prog_container.empty()

    # Store in session
    st.session_state["df_res"]       = df_res
    st.session_state["score_100"]    = score_100
    st.session_state["alertes"]      = alertes
    st.session_state["n_vert"]       = n_vert
    st.session_state["n_orange"]     = n_orange
    st.session_state["n_rouge"]      = n_rouge
    st.session_state["features"]     = features
    st.session_state["shap_values"]  = shap_values
    st.session_state["shap_df"]      = shap_df
    st.session_state["labels"]       = labels
    st.session_state["X_u2"]         = X_u2
    st.session_state["if_sc"]        = if_sc
    st.session_state["vae_sc"]       = vae_sc
    st.session_state["t_total"]      = t_total
    st.session_state["p_seuil_vert"] = p_seuil_vert
    st.session_state["p_seuil_orange"] = p_seuil_orange
    st.session_state["n_cl"]         = n_cl

# ═══════════════════════════════════════════════════════════════
#  LOAD FROM SESSION
# ═══════════════════════════════════════════════════════════════
if "df_res" not in st.session_state:
    st.stop()

df_res       = st.session_state["df_res"]
score_100    = st.session_state["score_100"]
alertes      = st.session_state["alertes"]
n_vert       = st.session_state["n_vert"]
n_orange     = st.session_state["n_orange"]
n_rouge      = st.session_state["n_rouge"]
features     = st.session_state["features"]
shap_values  = st.session_state["shap_values"]
shap_df      = st.session_state["shap_df"]
labels       = st.session_state["labels"]
X_u2         = st.session_state["X_u2"]
if_sc        = st.session_state["if_sc"]
vae_sc       = st.session_state["vae_sc"]
t_total      = st.session_state["t_total"]
p_seuil_vert   = st.session_state["p_seuil_vert"]
p_seuil_orange = st.session_state["p_seuil_orange"]
n_cl         = st.session_state["n_cl"]

# ═══════════════════════════════════════════════════════════════
#  KPI BAR
# ═══════════════════════════════════════════════════════════════
pct_rouge  = round(n_rouge/len(df_res)*100)
pct_orange = round(n_orange/len(df_res)*100)
pct_vert   = round(n_vert/len(df_res)*100)
score_moy  = score_100.mean()

st.markdown(f"""
<div class="sg-kpi-grid">
  <div class="sg-kpi total">
    <div class="sg-kpi-trend">🏭</div>
    <div class="sg-kpi-label">Fournisseurs analysés</div>
    <div class="sg-kpi-value">{len(df_res)}</div>
    <div class="sg-kpi-sub">Score moyen : {score_moy:.1f}/100</div>
  </div>
  <div class="sg-kpi green">
    <div class="sg-kpi-trend">✓</div>
    <div class="sg-kpi-label">Risque faible</div>
    <div class="sg-kpi-value">{n_vert}</div>
    <div class="sg-kpi-sub">{pct_vert}% du portefeuille</div>
  </div>
  <div class="sg-kpi orange">
    <div class="sg-kpi-trend">⚠</div>
    <div class="sg-kpi-label">Sous surveillance</div>
    <div class="sg-kpi-value">{n_orange}</div>
    <div class="sg-kpi-sub">{pct_orange}% du portefeuille</div>
  </div>
  <div class="sg-kpi red">
    <div class="sg-kpi-trend">🚨</div>
    <div class="sg-kpi-label">Action immédiate</div>
    <div class="sg-kpi-value">{n_rouge}</div>
    <div class="sg-kpi-sub">{pct_rouge}% du portefeuille</div>
  </div>
  <div class="sg-kpi score">
    <div class="sg-kpi-trend">📊</div>
    <div class="sg-kpi-label">Score de risque moyen</div>
    <div class="sg-kpi-value">{score_moy:.0f}</div>
    <div class="sg-kpi-sub">sur 100 · Analyse IA</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  GLOBAL ALERT BANNER
# ═══════════════════════════════════════════════════════════════
if pct_rouge > 20:
    al_class, al_ico, al_title = "critical", "🚨", "Situation Critique"
    al_body = f"{n_rouge} fournisseurs ({pct_rouge}%) nécessitent une action immédiate. Activez votre plan de contingence."
elif pct_rouge > 10 or pct_orange > 30:
    al_class, al_ico, al_title = "warning", "⚠️", "Situation Préoccupante"
    al_body = f"{n_rouge} en alerte rouge et {n_orange} en surveillance. Des mesures correctives ciblées s'imposent."
else:
    al_class, al_ico, al_title = "safe", "✅", "Portefeuille Sous Contrôle"
    al_body = f"Seulement {pct_rouge}% en alerte critique. Maintenez la vigilance sur les {n_orange} fournisseurs en surveillance."

st.markdown(f"""
<div class="sg-alert {al_class}">
  <div class="sg-alert-ico">{al_ico}</div>
  <div>
    <div class="sg-alert-title">{al_title} — {date.today().strftime("%d/%m/%Y")}</div>
    <div class="sg-alert-body">{al_body}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Vue d'ensemble",
    "🗺️ Cartographie",
    "📋 Liste des fournisseurs",
    "🔍 Fiche détaillée",
    "📈 Tendances",
    "📄 Rapport Exécutif",
])

tab_overview, tab_map, tab_list, tab_detail, tab_trends, tab_report = tabs

# ═══════════════════════════════════════════════════════════════
#  TAB 1 — VUE D'ENSEMBLE (Power BI style)
# ═══════════════════════════════════════════════════════════════
with tab_overview:

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        # ── Distribution Score ──
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=score_100, nbinsx=40,
            marker=dict(
                color=score_100,
                colorscale=[[0,"#00C566"],[0.4,"#FF8C00"],[1,"#FF2D55"]],
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            opacity=0.9, hovertemplate="Score %{x:.0f} · %{y} fournisseurs<extra></extra>",
        ))
        fig_hist.add_vline(x=p_seuil_vert,   line_dash="dash", line_color="#00C566", line_width=1.5)
        fig_hist.add_vline(x=p_seuil_orange, line_dash="dash", line_color="#FF8C00", line_width=1.5)
        fig_hist.add_vline(x=score_100.mean(), line_dash="dot", line_color="#6EA3FF", line_width=2,
                            annotation_text=f"Moy. {score_100.mean():.1f}", annotation_font_color="#6EA3FF")
        apply_theme(fig_hist, "Distribution des scores de risque", 280)
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Donut + Bar côte à côte ──
        c1, c2 = st.columns(2)
        with c1:
            fig_donut = go.Figure(go.Pie(
                labels=["Risque faible","Surveillance","Action immédiate"],
                values=[n_vert, n_orange, n_rouge],
                hole=0.65,
                marker=dict(colors=["#00C566","#FF8C00","#FF2D55"], line=dict(color="#080C14", width=3)),
                textinfo="label+percent",
                textfont=dict(size=10, family="Sora"),
                hovertemplate="%{label}<br>%{value} fournisseurs<br>%{percent}<extra></extra>",
            ))
            fig_donut.add_annotation(text=f"<b>{len(df_res)}</b><br><span style='font-size:10px'>total</span>",
                                      x=0.5, y=0.5, showarrow=False,
                                      font=dict(size=18, color="#E8EDF5", family="Sora"))
            apply_theme(fig_donut, "Répartition", 280)
            fig_donut.update_layout(showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            # Score par secteur (si disponible)
            if "Secteur" in df_res.columns:
                sect_df = df_res.groupby("Secteur")["Score_Risque"].mean().sort_values(ascending=True).reset_index()
                colors_sect = [risk_color(s, p_seuil_vert, p_seuil_orange) for s in sect_df["Score_Risque"]]
                fig_sect = go.Figure(go.Bar(
                    x=sect_df["Score_Risque"], y=sect_df["Secteur"],
                    orientation="h",
                    marker_color=colors_sect,
                    text=sect_df["Score_Risque"].round(1), textposition="outside",
                    textfont=dict(color="#E8EDF5", size=10),
                    hovertemplate="%{y}<br>Score moyen : %{x:.1f}<extra></extra>",
                ))
                apply_theme(fig_sect, "Score par secteur", 280)
                fig_sect.update_layout(yaxis=dict(tickfont=dict(size=10)))
                st.plotly_chart(fig_sect, use_container_width=True)
            else:
                # Fallback: top 10 risqués
                top10 = df_res.nlargest(8, "Score_Risque")
                id_col = "ID_Fournisseur" if "ID_Fournisseur" in df_res.columns else df_res.index.name or "Index"
                labels_top = top10[id_col].astype(str) if id_col in top10.columns else top10.index.astype(str)
                fig_top = go.Figure(go.Bar(
                    x=top10["Score_Risque"], y=labels_top,
                    orientation="h",
                    marker_color=[risk_color(s, p_seuil_vert, p_seuil_orange) for s in top10["Score_Risque"]],
                    text=top10["Score_Risque"].round(1), textposition="outside",
                    textfont=dict(color="#E8EDF5", size=10),
                ))
                apply_theme(fig_top, "Top 8 fournisseurs à risque", 280)
                st.plotly_chart(fig_top, use_container_width=True)

    with col_r:
        # ── Gauge Score Moyen ──
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score_moy,
            delta={"reference": p_seuil_vert, "increasing": {"color": "#FF2D55"}, "decreasing": {"color": "#00C566"}},
            number={"font": {"size": 44, "family": "Sora", "color": "#E8EDF5"}, "suffix": "/100"},
            gauge={
                "axis": {"range": [0,100], "tickcolor": "#4A6FA5", "tickwidth": 1},
                "bar": {"color": risk_color(score_moy, p_seuil_vert, p_seuil_orange), "thickness": 0.3},
                "bgcolor": "#131E30",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, p_seuil_vert],   "color": "rgba(0,197,102,0.08)"},
                    {"range": [p_seuil_vert, p_seuil_orange], "color": "rgba(255,140,0,0.08)"},
                    {"range": [p_seuil_orange, 100], "color": "rgba(255,45,85,0.08)"},
                ],
                "threshold": {"line": {"color": "#6EA3FF", "width": 2}, "thickness": 0.8, "value": score_moy},
            },
            title={"text": "Score de risque moyen<br><span style='font-size:12px;color:#4A6FA5'>Portefeuille global</span>",
                   "font": {"size": 14, "family": "Sora", "color": "#E8EDF5"}},
        ))
        fig_gauge.update_layout(**{k:v for k,v in CHART_THEME.items() if k not in ("gridcolor","zerolinecolor")}, height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Top facteurs SHAP ──
        if shap_df is not None:
            st.markdown('<div class="sg-card"><div class="sg-card-title">🎯 Principaux facteurs de risque</div>', unsafe_allow_html=True)
            for _, row in shap_df.head(p_shap_top).iterrows():
                pct = row["SHAP_pct"]
                col = "#FF2D55" if pct > 10 else "#FF8C00" if pct > 5 else "#6EA3FF"
                width_pct = int(pct * 3)
                st.markdown(f"""
                <div style='margin-bottom:10px'>
                  <div style='display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:4px'>
                    <span style='color:#E8EDF5;font-weight:600'>{row["Variable"]}</span>
                    <span style='color:{col};font-family:JetBrains Mono,monospace;font-weight:700'>{pct:.1f}%</span>
                  </div>
                  <div class='score-bar-wrap'>
                    <div class='score-bar-fill' style='width:{min(width_pct,100)}%;background:{col}'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Score percentile breakdown
            st.markdown('<div class="sg-card"><div class="sg-card-title">📊 Répartition détaillée</div>', unsafe_allow_html=True)
            for label, threshold, color in [("0–20 (très faible)", 20, "#00C566"),
                                             ("21–40 (faible)", 40, "#30E88A"),
                                             ("41–60 (modéré)", 60, "#FF8C00"),
                                             ("61–80 (élevé)", 80, "#FF6B35"),
                                             ("81–100 (critique)", 101, "#FF2D55")]:
                lo = int(label.split("–")[0])
                hi = threshold
                count = int(((score_100 >= lo) & (score_100 < hi)).sum())
                pct_b = count/len(df_res)*100
                st.markdown(f"""
                <div style='margin-bottom:10px'>
                  <div style='display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:3px'>
                    <span style='color:#8BA3C7'>{label}</span>
                    <span style='color:{color};font-weight:700'>{count} ({pct_b:.0f}%)</span>
                  </div>
                  <div class='score-bar-wrap'>
                    <div class='score-bar-fill' style='width:{pct_b:.0f}%;background:{color}'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Heatmap Secteur × Région ──
    if "Secteur" in df_res.columns and "Region_Maroc" in df_res.columns:
        st.markdown("---")
        st.markdown('<div class="sg-section"><div class="sg-section-icon" style="background:rgba(30,94,255,0.1)">🗺️</div><div><div class="sg-section-title">Heatmap Secteur × Région</div><div class="sg-section-sub">Score de risque moyen par croisement secteur / région géographique</div></div></div>', unsafe_allow_html=True)
        pivot = df_res.pivot_table(values="Score_Risque", index="Secteur", columns="Region_Maroc", aggfunc="mean").round(1)
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0,"#002B1A"],[0.3,"#00C566"],[0.6,"#FF8C00"],[1,"#FF2D55"]],
            text=pivot.values.round(1), texttemplate="%{text}",
            textfont=dict(size=10, family="JetBrains Mono"),
            hovertemplate="Secteur: %{y}<br>Région: %{x}<br>Score: %{z:.1f}<extra></extra>",
            colorbar=dict(title="Score", tickfont=dict(color="#8BA3C7")),
            zmin=0, zmax=100,
        ))
        apply_theme(fig_hm, "Heatmap des risques — Secteur × Région Maroc", 400)
        fig_hm.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── Scatter Anomalie Score ──
    st.markdown("---")
    st.markdown('<div class="sg-section"><div class="sg-section-icon" style="background:rgba(255,45,85,0.1)">🔬</div><div><div class="sg-section-title">Cartographie des anomalies</div><div class="sg-section-sub">Positionnement de chaque fournisseur selon son profil de risque global</div></div></div>', unsafe_allow_html=True)

    hover_cols = [c for c in ["ID_Fournisseur","Secteur","Region_Maroc"] if c in df_res.columns]
    dp = df_res.copy()
    dp["_X"] = X_u2[:, 0]; dp["_Y"] = X_u2[:, 1]
    dp["_Alerte_Label"] = dp["Niveau_Alerte"].map({"🟢 Vert":"Risque faible","🟠 Orange":"Surveillance","🔴 Rouge":"Action immédiate"})

    color_map = {"Risque faible":"#00C566","Surveillance":"#FF8C00","Action immédiate":"#FF2D55"}
    fig_scatter = px.scatter(
        dp, x="_X", y="_Y", color="_Alerte_Label",
        color_discrete_map=color_map,
        size="Score_Risque", size_max=18,
        hover_data=hover_cols + ["Score_Risque","Priorite"],
        labels={"_X":"Axe 1","_Y":"Axe 2","_Alerte_Label":"Niveau de risque","Score_Risque":"Score"},
    )
    fig_scatter.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color="#080C14")))
    fig_scatter.update_layout(
        **{k:v for k,v in CHART_THEME.items()},
        height=440,
        title=dict(text="Cartographie des fournisseurs — Taille = Niveau de risque", font=dict(size=13, color="#E8EDF5")),
        legend=dict(orientation="h", y=-0.12, font=dict(color="#8BA3C7")),
        xaxis=dict(showgrid=True, gridcolor="#1E2D45", zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="#1E2D45", zeroline=False, showticklabels=False, title=""),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 2 — CARTOGRAPHIE GÉOGRAPHIQUE
# ═══════════════════════════════════════════════════════════════
with tab_map:
    st.markdown('<div class="sg-section"><div class="sg-section-icon" style="background:rgba(0,197,102,0.1)">🗺️</div><div><div class="sg-section-title">Cartographie géographique</div><div class="sg-section-sub">Distribution des risques par région du Maroc</div></div></div>', unsafe_allow_html=True)

    if "Region_Maroc" in df_res.columns:
        reg_df = df_res.groupby("Region_Maroc").agg(
            Score_moyen=("Score_Risque","mean"),
            Nb_total=("Score_Risque","count"),
            Nb_rouge=("Niveau_Alerte", lambda x: (x=="🔴 Rouge").sum()),
            Nb_orange=("Niveau_Alerte", lambda x: (x=="🟠 Orange").sum()),
            Nb_vert=("Niveau_Alerte", lambda x: (x=="🟢 Vert").sum()),
        ).reset_index()
        reg_df["Score_moyen"] = reg_df["Score_moyen"].round(1)
        reg_df["Taux_critique_%"] = (reg_df["Nb_rouge"]/reg_df["Nb_total"]*100).round(1)

        c_map1, c_map2 = st.columns([3, 2])

        with c_map1:
            fig_reg = go.Figure(go.Bar(
                x=reg_df["Score_moyen"],
                y=reg_df["Region_Maroc"],
                orientation="h",
                marker=dict(
                    color=reg_df["Score_moyen"],
                    colorscale=[[0,"#002B1A"],[0.35,"#00C566"],[0.65,"#FF8C00"],[1,"#FF2D55"]],
                    cmin=0, cmax=100,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                text=[f"{s:.0f}" for s in reg_df["Score_moyen"]], textposition="outside",
                textfont=dict(color="#E8EDF5", size=11),
                customdata=reg_df[["Nb_total","Nb_rouge","Taux_critique_%"]].values,
                hovertemplate="<b>%{y}</b><br>Score moyen: %{x:.1f}<br>Fournisseurs: %{customdata[0]}<br>🔴 Critiques: %{customdata[1]} (%{customdata[2]:.1f}%)<extra></extra>",
            ))
            apply_theme(fig_reg, "Score de risque moyen par région", 440)
            fig_reg.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=11)))
            st.plotly_chart(fig_reg, use_container_width=True)

        with c_map2:
            # Stacked bar by region
            fig_stack = go.Figure()
            fig_stack.add_trace(go.Bar(name="Action immédiate", x=reg_df["Region_Maroc"], y=reg_df["Nb_rouge"],
                                        marker_color="#FF2D55", hovertemplate="%{x}<br>🔴 Critiques: %{y}<extra></extra>"))
            fig_stack.add_trace(go.Bar(name="Surveillance", x=reg_df["Region_Maroc"], y=reg_df["Nb_orange"],
                                        marker_color="#FF8C00", hovertemplate="%{x}<br>🟠 Surveillance: %{y}<extra></extra>"))
            fig_stack.add_trace(go.Bar(name="Risque faible", x=reg_df["Region_Maroc"], y=reg_df["Nb_vert"],
                                        marker_color="#00C566", hovertemplate="%{x}<br>🟢 Stables: %{y}<extra></extra>"))
            fig_stack.update_layout(barmode="stack")
            apply_theme(fig_stack, "Composition des risques par région", 440)
            fig_stack.update_layout(
                legend=dict(orientation="h", y=-0.2, font=dict(color="#8BA3C7", size=10)),
                xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

        # Tableau récap région
        st.markdown("#### Tableau récapitulatif par région")
        reg_display = reg_df.rename(columns={
            "Region_Maroc":"Région","Score_moyen":"Score moyen","Nb_total":"Total",
            "Nb_rouge":"🔴 Critiques","Nb_orange":"🟠 Surveillance","Nb_vert":"🟢 Stables","Taux_critique_%":"% Critiques"
        })
        st.dataframe(reg_display.sort_values("Score moyen", ascending=False), use_container_width=True, hide_index=True)

        if "Secteur" in df_res.columns:
            st.markdown("---")
            # Treemap secteur / région
            tree_df = df_res.groupby(["Region_Maroc","Secteur"]).agg(
                Score=("Score_Risque","mean"), Count=("Score_Risque","count")
            ).reset_index()
            tree_df["Score"] = tree_df["Score"].round(1)
            fig_tree = px.treemap(
                tree_df, path=["Region_Maroc","Secteur"],
                values="Count", color="Score",
                color_continuous_scale=[[0,"#002B1A"],[0.35,"#00C566"],[0.65,"#FF8C00"],[1,"#FF2D55"]],
                range_color=[0,100],
                hover_data={"Score":True,"Count":True},
                title="Treemap — Taille = Nb fournisseurs · Couleur = Score de risque",
            )
            fig_tree.update_traces(textinfo="label+value", textfont=dict(family="Sora", size=12))
            apply_theme(fig_tree, "Treemap des risques par région et secteur", 460)
            fig_tree.update_layout(coloraxis_colorbar=dict(title="Score",tickfont=dict(color="#8BA3C7")))
            st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("ℹ️ Aucune colonne `Region_Maroc` détectée. Ajoutez cette colonne à votre fichier pour activer la cartographie géographique.")
        # Treemap par secteur uniquement
        if "Secteur" in df_res.columns:
            tree_df = df_res.groupby("Secteur").agg(Score=("Score_Risque","mean"),Count=("Score_Risque","count")).reset_index()
            fig_tree = px.treemap(tree_df, path=["Secteur"], values="Count", color="Score",
                                   color_continuous_scale=[[0,"#002B1A"],[0.35,"#00C566"],[0.65,"#FF8C00"],[1,"#FF2D55"]],
                                   range_color=[0,100])
            apply_theme(fig_tree,"Treemap des risques par secteur",400)
            st.plotly_chart(fig_tree, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 3 — LISTE INTERACTIVE
# ═══════════════════════════════════════════════════════════════
with tab_list:
    st.markdown('<div class="sg-section"><div class="sg-section-icon" style="background:rgba(155,89,182,0.1)">📋</div><div><div class="sg-section-title">Liste des fournisseurs</div><div class="sg-section-sub">Filtrez, triez et exportez — tous les fournisseurs de votre portefeuille</div></div></div>', unsafe_allow_html=True)

    # ── Filtres ──
    fc1, fc2, fc3, fc4 = st.columns(4)
    sel_alerte = fc1.multiselect("Niveau de risque", ["🔴 Action immédiate","🟠 Surveillance","🟢 Risque faible"],
                                   default=["🔴 Action immédiate","🟠 Surveillance"],
                                   key="filter_alerte")
    smin = fc2.slider("Score min", 0, 100, 0, key="smin")
    smax = fc3.slider("Score max", 0, 100, 100, key="smax")

    alerte_map_rev = {"🔴 Action immédiate":"🔴 Rouge","🟠 Surveillance":"🟠 Orange","🟢 Risque faible":"🟢 Vert"}
    sel_alertes_raw = [alerte_map_rev[s] for s in sel_alerte]

    if "Secteur" in df_res.columns:
        sect_list = ["Tous"] + sorted(df_res["Secteur"].dropna().unique().tolist())
        sel_sect = fc4.selectbox("Secteur", sect_list, key="filter_sect")
    else:
        sel_sect = "Tous"

    df_f = df_res[df_res["Niveau_Alerte"].isin(sel_alertes_raw) & (df_res["Score_Risque"] >= smin) & (df_res["Score_Risque"] <= smax)].copy()
    if sel_sect != "Tous" and "Secteur" in df_f.columns:
        df_f = df_f[df_f["Secteur"] == sel_sect]
    df_f = df_f.sort_values("Score_Risque", ascending=False)

    st.markdown(f"<div style='font-size:0.78rem;color:#4A6FA5;margin-bottom:12px'><b style='color:#E8EDF5'>{len(df_f)}</b> fournisseurs affichés sur {len(df_res)}</div>", unsafe_allow_html=True)

    # ── Tableau stylisé ──
    afficher = [c for c in ["ID_Fournisseur","Nom_Fournisseur","Secteur","Region_Maroc",
                              "Niveau_Alerte","Score_Risque","Priorite","_Score_IF","_Score_VAE","_Top_Facteur"]
                if c in df_f.columns]
    display_df = df_f[afficher].copy()
    display_df.columns = [c.replace("_Score_IF","Score IF").replace("_Score_VAE","Score VAE")
                           .replace("_Top_Facteur","Facteur principal").replace("Niveau_Alerte","Alerte") for c in display_df.columns]

    st.dataframe(
        display_df,
        use_container_width=True, height=460,
        column_config={
            "Score_Risque": st.column_config.ProgressColumn("Score risque", min_value=0, max_value=100, format="%.1f"),
            "Score IF": st.column_config.ProgressColumn("Score IF", min_value=0, max_value=100, format="%.1f"),
            "Score VAE": st.column_config.ProgressColumn("Score VAE", min_value=0, max_value=100, format="%.1f"),
        },
        hide_index=True,
    )

    # ── Exports ──
    ec1, ec2 = st.columns(2)
    with ec1:
        csv_data = df_res.drop(columns=[c for c in df_res.columns if c.startswith("_")], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️  Exporter tout le portefeuille (CSV)", data=csv_data,
                            file_name=f"supplygard_portefeuille_{date.today().strftime('%Y%m%d')}.csv",
                            mime="text/csv", use_container_width=True)
    with ec2:
        csv_filter = df_f.drop(columns=[c for c in df_f.columns if c.startswith("_")], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️  Exporter la sélection (CSV)", data=csv_filter,
                            file_name=f"supplygard_selection_{date.today().strftime('%Y%m%d')}.csv",
                            mime="text/csv", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 4 — FICHE FOURNISSEUR
# ═══════════════════════════════════════════════════════════════
with tab_detail:
    id_col = "ID_Fournisseur" if "ID_Fournisseur" in df_res.columns else None
    opts   = df_res[id_col].tolist() if id_col else list(df_res.index)

    # Tri par score décroissant
    df_sorted_for_sel = df_res.sort_values("Score_Risque", ascending=False)
    opts_sorted = df_sorted_for_sel[id_col].tolist() if id_col else list(df_sorted_for_sel.index)

    sel_id = st.selectbox("🔍 Sélectionnez un fournisseur", opts_sorted,
                           format_func=lambda x: f"{'🔴' if df_res[df_res[id_col]==x]['Niveau_Alerte'].values[0]=='🔴 Rouge' else '🟠' if df_res[df_res[id_col]==x]['Niveau_Alerte'].values[0]=='🟠 Orange' else '🟢'} {x} — Score {df_res[df_res[id_col]==x]['Score_Risque'].values[0]:.0f}/100" if id_col else str(x),
                           key="detail_sel")

    idx  = df_res[df_res[id_col]==sel_id].index[0] if id_col else sel_id
    lig  = df_res.loc[idx]
    sc_v = float(lig["Score_Risque"])
    al_v = str(lig["Niveau_Alerte"])
    fcl  = int(lig["_Cluster"])

    al_class = "critical" if "Rouge" in al_v else "warning" if "Orange" in al_v else "safe"
    al_label = "Action immédiate" if "Rouge" in al_v else "Sous surveillance" if "Orange" in al_v else "Risque faible"
    al_ico   = "🔴" if "Rouge" in al_v else "🟠" if "Orange" in al_v else "🟢"

    fid  = str(lig.get("ID_Fournisseur", f"#{idx}"))
    fsec = str(lig.get("Secteur","N/A"))
    freg = str(lig.get("Region_Maroc","N/A"))

    # Header card
    st.markdown(f"""
    <div class="sg-alert {al_class}" style="margin-bottom:20px">
      <div class="sg-alert-ico" style="font-size:2rem">{al_ico}</div>
      <div style="flex:1">
        <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
          <span style="font-size:1.3rem;font-weight:800;color:#E8EDF5">{fid}</span>
          <span class="badge-{'rouge' if 'Rouge' in al_v else 'orange' if 'Orange' in al_v else 'vert'}">{al_label}</span>
        </div>
        <div style="font-size:0.78rem;color:#4A6FA5;margin-top:4px">{fsec} · {freg} · {"Anomalie comportementale" if fcl == -1 else f"Groupe {fcl}"}</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:2.5rem;font-weight:800;font-family:JetBrains Mono,monospace;color:{'#FF2D55' if 'Rouge' in al_v else '#FF8C00' if 'Orange' in al_v else '#00C566'}">{sc_v:.0f}</div>
        <div style="font-size:0.7rem;color:#4A6FA5">/ 100</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 métriques ──
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Score global",       f"{sc_v:.1f}/100")
    d2.metric("Score anomalie IF",  f"{float(lig['_Score_IF']):.1f}/100",
              delta="⚠️ Élevé" if float(lig['_Score_IF'])>50 else "✓ Normal")
    d3.metric("Score anomalie VAE", f"{float(lig['_Score_VAE']):.1f}/100",
              delta="⚠️ Élevé" if float(lig['_Score_VAE'])>50 else "✓ Normal")
    d4.metric("Priorité",           str(lig.get("Priorite","Standard")))

    st.markdown("---")
    left_d, right_d = st.columns([3, 2])

    with left_d:
        # ── Facteurs d'influence ──
        st.markdown("#### 🎯 Facteurs d'influence")
        if shap_values is not None:
            pos   = df_res.index.get_loc(idx)
            sv    = shap_values[pos]
            top5  = np.argsort(np.abs(sv))[::-1][:8]
            tot   = np.abs(sv).sum() + 1e-10
            for rk, j in enumerate(top5, 1):
                ct    = abs(sv[j]) / tot * 100
                dir_  = "↑ Augmente le risque" if sv[j] > 0 else "↓ Réduit le risque"
                col_  = "#FF2D55" if sv[j] > 0 else "#00C566"
                vn    = features[j]
                try:  vb = round(float(df_res.loc[idx, vn]), 2)
                except: vb = "N/A"
                bar_w = int(ct * 4)
                st.markdown(f"""
                <div style='background:#0D1420;border:1px solid #1E2D45;border-radius:10px;padding:10px 14px;margin-bottom:8px'>
                  <div style='display:flex;align-items:center;gap:12px'>
                    <div style='font-size:0.75rem;font-weight:700;color:#4A6FA5;min-width:20px'>{rk}</div>
                    <div style='flex:1'>
                      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px'>
                        <span style='font-size:0.82rem;font-weight:700;color:#E8EDF5'>{vn}</span>
                        <span style='font-family:JetBrains Mono,monospace;font-size:0.78rem;color:{col_};font-weight:700'>{ct:.1f}%</span>
                      </div>
                      <div class='score-bar-wrap' style='height:6px'>
                        <div class='score-bar-fill' style='width:{min(bar_w,100)}%;background:{col_};height:6px'></div>
                      </div>
                      <div style='display:flex;justify-content:space-between;margin-top:4px'>
                        <span style='font-size:0.7rem;color:{col_}'>{dir_}</span>
                        <span style='font-size:0.7rem;color:#4A6FA5;font-family:JetBrains Mono,monospace'>val: {vb}</span>
                      </div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Activez les facteurs d'influence dans les paramètres (barre latérale).")

    with right_d:
        # ── Radar ──
        rv = [v for v in ["OTD_Pct","Altman_ZScore","Score_ESG","Stabilite_Politique",
                           "_Score_IF","Current_Ratio","Dependance_Mono"] if v in df_res.columns]
        rv_clean = [v.replace("_Score_IF","Score IF") for v in rv]
        if len(rv) >= 3:
            st.markdown("#### 📡 Profil de risque")
            vn_r = []
            for v in rv:
                d_col = df_res[v].dropna()
                vn_r.append(float(np.clip(
                    (float(df_res.loc[idx, v]) - d_col.min()) / (d_col.max()-d_col.min()+1e-10), 0, 1)*100))
            fig_rad = go.Figure(go.Scatterpolar(
                r=vn_r+[vn_r[0]], theta=rv_clean+[rv_clean[0]], fill="toself",
                fillcolor=f"rgba({'255,45,85' if 'Rouge' in al_v else '255,140,0' if 'Orange' in al_v else '0,197,102'},0.12)",
                line=dict(color=("#FF2D55" if "Rouge" in al_v else "#FF8C00" if "Orange" in al_v else "#00C566"), width=2),
                marker=dict(color="#E8EDF5", size=6),
            ))
            fig_rad.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0,100], gridcolor="#1E2D45", tickfont=dict(color="#4A6FA5",size=8)),
                    angularaxis=dict(tickfont=dict(color="#8BA3C7",size=10), gridcolor="#1E2D45"),
                ),
                height=320, showlegend=False, margin=dict(l=20,r=20,t=30,b=20),
                font=dict(family="Sora", color="#8BA3C7"),
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        # ── Recommandation ──
        st.markdown("#### ✅ Plan d'action recommandé")
        aggrave_vars = []
        if shap_values is not None:
            pos  = df_res.index.get_loc(idx)
            sv   = shap_values[pos]
            top5 = np.argsort(np.abs(sv))[::-1][:5]
            aggrave_vars = [features[j] for j in top5 if sv[j] > 0]

        var_cible = aggrave_vars[0] if aggrave_vars else "indicateurs clés"

        if "Rouge" in al_v:
            actions = [
                "Réunion de crise avec le fournisseur sous 72h",
                "Activation du plan de contingence",
                "Identification de fournisseurs alternatifs",
                f"Audit ciblé sur : <b>{var_cible}</b>",
                "Réduction des commandes en cours si possible",
            ]
            bg, bdr = "rgba(255,45,85,0.08)", "#FF2D55"
        elif "Orange" in al_v:
            actions = [
                "Audit de performance planifié ce mois",
                "Passage au reporting hebdomadaire",
                "Mise en place d'alertes sur les KPIs critiques",
                f"Surveillance renforcée de : <b>{var_cible}</b>",
                "Préparation du plan de continuité",
            ]
            bg, bdr = "rgba(255,140,0,0.08)", "#FF8C00"
        else:
            actions = [
                "Maintenir le suivi mensuel standard",
                "Revue trimestrielle des indicateurs",
                f"Continuer à monitorer : <b>{var_cible}</b>",
                "Ce fournisseur peut servir de référence",
            ]
            bg, bdr = "rgba(0,197,102,0.06)", "#00C566"

        action_html = "".join([f"<div style='display:flex;gap:10px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05)'><span style='color:{bdr};font-weight:700;flex-shrink:0'>{i+1}.</span><span style='font-size:0.78rem;color:#8BA3C7'>{a}</span></div>" for i,a in enumerate(actions)])
        st.markdown(f"""
        <div style='background:{bg};border:1px solid {bdr}40;border-radius:12px;padding:16px;margin-top:8px'>
          {action_html}
        </div>
        """, unsafe_allow_html=True)

    # ── Comparaison cluster ──
    if fcl != -1 and mask.sum() > 5:
        st.markdown("---")
        cl_scores = score_100[labels == fcl]
        cl_mean   = cl_scores.mean()
        st.markdown(f"#### 📊 Position dans son groupe ({len(cl_scores)} fournisseurs similaires)")
        fig_hist_cl = go.Figure()
        fig_hist_cl.add_trace(go.Histogram(x=cl_scores, nbinsx=20, marker_color="#2A3F5F", name="Groupe", opacity=0.8))
        fig_hist_cl.add_vline(x=cl_mean, line_dash="dash", line_color="#6EA3FF", annotation_text=f"Moy groupe {cl_mean:.1f}")
        fig_hist_cl.add_vline(x=sc_v, line_dash="solid", line_color=("#FF2D55" if "Rouge" in al_v else "#FF8C00" if "Orange" in al_v else "#00C566"),
                               annotation_text=f"{fid}: {sc_v:.1f}", line_width=2)
        apply_theme(fig_hist_cl, f"Position de {fid} dans son groupe de pairs", 260)
        st.plotly_chart(fig_hist_cl, use_container_width=True)

    # ── Export fiche ──
    fiche_csv = df_res.loc[[idx]].drop(columns=[c for c in df_res.columns if c.startswith("_")], errors="ignore").to_csv(index=False).encode("utf-8-sig")
    st.download_button(f"⬇️ Télécharger la fiche de {fid}", data=fiche_csv,
                        file_name=f"fiche_{fid}_{date.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 5 — TENDANCES & ANALYSES AVANCÉES
# ═══════════════════════════════════════════════════════════════
with tab_trends:
    st.markdown('<div class="sg-section"><div class="sg-section-icon" style="background:rgba(155,89,182,0.1)">📈</div><div><div class="sg-section-title">Analyses approfondies</div><div class="sg-section-sub">Corrélations, distributions et analyses multidimensionnelles</div></div></div>', unsafe_allow_html=True)

    t_a, t_b, t_c = st.tabs(["📊 Corrélations","🔄 Comparaisons","📉 Distributions"])

    with t_a:
        # Matrice corrélation
        st.markdown("#### Matrice de corrélation des indicateurs clés")
        num_feats = [f for f in features if df_res[f].notna().sum() > len(df_res)*0.5][:20]
        if len(num_feats) >= 3:
            corr = df_res[num_feats].corr().round(2)
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale=[[0,"#FF2D55"],[0.5,"#131E30"],[1,"#1E5EFF"]],
                zmid=0, zmin=-1, zmax=1,
                text=corr.values.round(2), texttemplate="%{text}", textfont=dict(size=8),
                hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.2f}<extra></extra>",
                colorbar=dict(title="r", tickfont=dict(color="#8BA3C7")),
            ))
            apply_theme(fig_corr, "Corrélations entre indicateurs (r de Pearson)", 520)
            fig_corr.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=9)), xaxis=dict(tickfont=dict(size=9), tickangle=-40))
            st.plotly_chart(fig_corr, use_container_width=True)

    with t_b:
        # Comparaison multi-indicateurs par niveau d'alerte
        st.markdown("#### Profils moyens par niveau de risque")
        num_feats_cmp = [f for f in features[:12] if df_res[f].notna().sum() > 10]
        if num_feats_cmp:
            # Normaliser pour comparaison
            df_norm = df_res[num_feats_cmp + ["Niveau_Alerte"]].copy()
            for f in num_feats_cmp:
                mn, mx = df_norm[f].min(), df_norm[f].max()
                if mx > mn: df_norm[f] = (df_norm[f]-mn)/(mx-mn)*100

            fig_cmp = go.Figure()
            colors_ = {"🟢 Vert":"#00C566","🟠 Orange":"#FF8C00","🔴 Rouge":"#FF2D55"}
            names_  = {"🟢 Vert":"Risque faible","🟠 Orange":"Surveillance","🔴 Rouge":"Action immédiate"}
            for al, col in colors_.items():
                sub = df_norm[df_norm["Niveau_Alerte"]==al][num_feats_cmp]
                if len(sub) > 0:
                    means = sub.mean().values
                    fig_cmp.add_trace(go.Scatterpolar(
                        r=means.tolist()+[means[0]], theta=num_feats_cmp+[num_feats_cmp[0]],
                        fill="toself", name=names_[al],
                        fillcolor=col.replace("#","rgba(").replace(",","") + ",0.08)" if col else col,
                        line=dict(color=col, width=2),
                    ))
            fig_cmp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0,100], gridcolor="#1E2D45", tickfont=dict(color="#4A6FA5",size=8)),
                    angularaxis=dict(tickfont=dict(color="#8BA3C7",size=9), gridcolor="#1E2D45"),
                ),
                height=460, legend=dict(orientation="h", y=-0.15, font=dict(color="#8BA3C7")),
                font=dict(family="Sora", color="#8BA3C7"),
                title=dict(text="Profils comparatifs — Variables normalisées 0–100", font=dict(size=13,color="#E8EDF5")),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        # Box plots par alerte
        if "Secteur" in df_res.columns and num_feats_cmp:
            st.markdown("#### Distribution des scores par secteur et niveau")
            fig_box = px.box(
                df_res, x="Secteur", y="Score_Risque", color="Niveau_Alerte",
                color_discrete_map={"🟢 Vert":"#00C566","🟠 Orange":"#FF8C00","🔴 Rouge":"#FF2D55"},
                points="outliers",
                labels={"Niveau_Alerte":"Niveau","Score_Risque":"Score de risque"},
            )
            apply_theme(fig_box, "Distribution des scores par secteur", 400)
            fig_box.update_layout(
                legend=dict(orientation="h", y=-0.2, font=dict(color="#8BA3C7")),
                xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
            )
            fig_box.add_hline(y=p_seuil_vert,   line_dash="dash", line_color="#00C566", line_width=1)
            fig_box.add_hline(y=p_seuil_orange, line_dash="dash", line_color="#FF8C00", line_width=1)
            st.plotly_chart(fig_box, use_container_width=True)

    with t_c:
        # Violin plots
        fig_vio = go.Figure()
        for al, col, name in [("🟢 Vert","#00C566","Risque faible"),("🟠 Orange","#FF8C00","Surveillance"),("🔴 Rouge","#FF2D55","Action immédiate")]:
            sub = score_100[alertes == al]
            if len(sub) > 2:
                fig_vio.add_trace(go.Violin(y=sub, name=name, fillcolor=col.replace("#","rgba(")+"19)",
                                             line_color=col, box_visible=True, meanline_visible=True,
                                             hovertemplate=f"{name}<br>Score: %{{y:.1f}}<extra></extra>"))
        apply_theme(fig_vio, "Distribution détaillée des scores par niveau de risque", 400)
        fig_vio.update_layout(violingap=0.3, violinmode="overlay", showlegend=True,
                               legend=dict(orientation="h", y=-0.15, font=dict(color="#8BA3C7")))
        st.plotly_chart(fig_vio, use_container_width=True)

        # Percentile chart
        pct_vals = np.percentile(score_100, [10,25,50,75,90])
        fig_pct = go.Figure(go.Bar(
            x=["P10","P25","Médiane","P75","P90"], y=pct_vals,
            marker=dict(color=pct_vals, colorscale=[[0,"#00C566"],[0.5,"#FF8C00"],[1,"#FF2D55"]], cmin=0, cmax=100),
            text=[f"{v:.1f}" for v in pct_vals], textposition="outside", textfont=dict(color="#E8EDF5"),
        ))
        fig_pct.add_hline(y=p_seuil_vert,   line_dash="dash", line_color="#00C566", line_width=1, annotation_text="Seuil vert")
        fig_pct.add_hline(y=p_seuil_orange, line_dash="dash", line_color="#FF8C00", line_width=1, annotation_text="Seuil orange")
        apply_theme(fig_pct, "Percentiles du score de risque", 300)
        st.plotly_chart(fig_pct, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 6 — RAPPORT EXÉCUTIF
# ═══════════════════════════════════════════════════════════════
with tab_report:
    TODAY = date.today().strftime("%d %B %Y")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0D1830,#0A1525);border:1px solid rgba(30,94,255,0.25);
    border-radius:16px;padding:32px 36px;margin-bottom:24px'>
      <div style='display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px'>
        <div>
          <div style='font-size:0.7rem;color:#4A6FA5;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:8px'>Rapport Exécutif Confidentiel</div>
          <h1 style='font-size:1.8rem;font-weight:800;color:#E8EDF5;margin:0 0 8px;letter-spacing:-0.04em'>Analyse du Portefeuille Fournisseurs</h1>
          <div style='font-size:0.85rem;color:#4A6FA5'>{TODAY} · {len(df_res)} fournisseurs analysés · Intelligence Artificielle</div>
        </div>
        <div style='text-align:right'>
          <div style='font-size:0.7rem;color:#4A6FA5;margin-bottom:4px'>Score global du portefeuille</div>
          <div style='font-size:3rem;font-weight:800;font-family:JetBrains Mono,monospace;color:{"#FF2D55" if score_moy>p_seuil_orange else "#FF8C00" if score_moy>p_seuil_vert else "#00C566"}'>{score_moy:.0f}</div>
          <div style='font-size:0.8rem;color:#4A6FA5'>/ 100</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Synthèse exécutive ──
    if pct_rouge > 20:
        synthese = f"Le portefeuille présente une <b>situation critique</b> : {n_rouge} fournisseurs ({pct_rouge}%) requièrent une action immédiate. Un plan de contingence doit être activé dans les plus brefs délais pour sécuriser la chaîne d'approvisionnement."
        niveau_global = "CRITIQUE"
        ico_g = "🚨"
    elif pct_rouge > 10 or pct_orange > 30:
        synthese = f"Le portefeuille est dans une <b>situation préoccupante</b> : {n_rouge} fournisseurs critiques et {n_orange} sous surveillance. Des mesures correctives ciblées sont nécessaires pour réduire l'exposition au risque."
        niveau_global = "PRÉOCCUPANT"
        ico_g = "⚠️"
    else:
        synthese = f"Le portefeuille est globalement <b>sous contrôle</b> avec seulement {pct_rouge}% de fournisseurs critiques. Un monitoring régulier des {n_orange} fournisseurs en surveillance permet de maintenir ce niveau de performance."
        niveau_global = "MAÎTRISÉ"
        ico_g = "✅"

    st.markdown(f"""
    <div class="exec-section">
      <h4>{ico_g} Synthèse exécutive — Niveau {niveau_global}</h4>
      <p>{synthese}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Chiffres clés ──
    st.markdown("#### 1. Chiffres clés")
    r1c, r2c, r3c, r4c = st.columns(4)
    r1c.metric("Fournisseurs analysés", len(df_res))
    r2c.metric("Score moyen portefeuille", f"{score_moy:.1f}/100")
    r3c.metric("Taux de risque critique", f"{pct_rouge}%")
    r4c.metric("Fournisseurs sous contrôle", f"{n_vert} ({pct_vert}%)")

    # ── Alertes prioritaires ──
    st.markdown("#### 2. Alertes prioritaires")
    top_rouge = df_res[df_res["Niveau_Alerte"]=="🔴 Rouge"].nlargest(5,"Score_Risque")
    if len(top_rouge) > 0:
        id_c = "ID_Fournisseur" if "ID_Fournisseur" in top_rouge.columns else None
        for _, row in top_rouge.iterrows():
            fid_r  = str(row[id_c]) if id_c else f"#{row.name}"
            sec_r  = str(row.get("Secteur","N/A"))
            reg_r  = str(row.get("Region_Maroc","N/A"))
            sc_r   = float(row["Score_Risque"])
            top_f  = str(row.get("_Top_Facteur","N/A"))
            st.markdown(f"""
            <div style='background:rgba(255,45,85,0.06);border:1px solid rgba(255,45,85,0.25);border-radius:10px;padding:12px 16px;margin:6px 0;display:flex;align-items:center;gap:16px'>
              <div style='font-size:1.5rem;font-weight:800;font-family:JetBrains Mono,monospace;color:#FF2D55;min-width:48px'>{sc_r:.0f}</div>
              <div style='flex:1'>
                <div style='font-size:0.88rem;font-weight:700;color:#E8EDF5'>{fid_r}</div>
                <div style='font-size:0.72rem;color:#4A6FA5'>{sec_r} · {reg_r} · Facteur : {top_f}</div>
              </div>
              <div style='font-size:0.72rem;color:#FF2D55;font-weight:700;border:1px solid rgba(255,45,85,0.4);border-radius:20px;padding:3px 10px'>ACTION IMMÉDIATE</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Aucun fournisseur en alerte critique dans le portefeuille actuel.")

    # ── Analyse par secteur (si disponible) ──
    if "Secteur" in df_res.columns:
        st.markdown("#### 3. Analyse par secteur d'activité")
        sect_rpt = df_res.groupby("Secteur").agg(
            Nb=("Score_Risque","count"),
            Score_moy=("Score_Risque","mean"),
            Critiques=("Niveau_Alerte", lambda x:(x=="🔴 Rouge").sum()),
            Stables=("Niveau_Alerte", lambda x:(x=="🟢 Vert").sum()),
        ).round(1).reset_index()
        sect_rpt["% Critique"] = (sect_rpt["Critiques"]/sect_rpt["Nb"]*100).round(0).astype(int)
        sect_rpt["Statut"] = sect_rpt["Score_moy"].apply(lambda s: "🔴 Critique" if s>p_seuil_orange else "🟠 Vigilance" if s>p_seuil_vert else "🟢 Stable")
        st.dataframe(sect_rpt.sort_values("Score_moy",ascending=False).rename(columns={"Secteur":"Secteur","Nb":"Nb fournisseurs","Score_moy":"Score moyen","Critiques":"🔴 Critiques","Stables":"🟢 Stables","% Critique":"% Critique","Statut":"Statut"}),
                     use_container_width=True, hide_index=True)

    # ── Recommandations stratégiques ──
    st.markdown("#### 4. Recommandations stratégiques")
    recs = [
        ("🔴", f"Actions immédiates ({n_rouge} fournisseurs)",
         f"Convoquer des réunions de crise, activer les plans de contingence et identifier des alternatives pour les {n_rouge} fournisseurs en alerte critique. Priorité maximale dans les 2 semaines."),
        ("🟠", f"Surveillance renforcée ({n_orange} fournisseurs)",
         f"Passer au reporting hebdomadaire et mettre en place des KPIs de suivi pour les {n_orange} fournisseurs en surveillance. Planifier des audits dans le mois."),
        ("🟢", f"Optimisation du portefeuille ({n_vert} fournisseurs stables)",
         f"Capitaliser sur les {n_vert} fournisseurs performants. Envisager d'augmenter les volumes avec les meilleurs pour réduire la dépendance aux fournisseurs à risque."),
        ("📊", "Gouvernance des données",
         "Améliorer la collecte et la qualité des données fournisseurs pour affiner la précision des scores. Mettre à jour le modèle trimestriellement."),
    ]
    for ico, title, body in recs:
        al_c = "critical" if ico == "🔴" else "warning" if ico == "🟠" else "safe"
        st.markdown(f"""
        <div class="sg-alert {al_c}">
          <div class="sg-alert-ico">{ico}</div>
          <div>
            <div class="sg-alert-title">{title}</div>
            <div class="sg-alert-body">{body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown("""
    <div style='background:#0D1420;border:1px solid #1E2D45;border-radius:12px;padding:16px 20px;margin-top:24px'>
      <div style='font-size:0.7rem;color:#2A3F5F;font-style:italic;line-height:1.7'>
        <b style='color:#4A6FA5'>Note méthodologique :</b> Les scores présentés sont issus d'une analyse statistique multivariée basée sur les données historiques fournies. Ils reflètent des déviances statistiques et ne constituent pas une décision automatisée. Chaque alerte doit être validée par les équipes métier avant toute action. Université Mohammed V — FSJES Agdal · Master M.I.E.L · 2025–2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Export rapport ──
    st.markdown("---")
    ec1, ec2 = st.columns(2)
    with ec1:
        full_csv = df_res.drop(columns=[c for c in df_res.columns if c.startswith("_")], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Télécharger le rapport complet (CSV)",
                            data=full_csv,
                            file_name=f"SupplyGuard_Rapport_{date.today().strftime('%Y%m%d')}.csv",
                            mime="text/csv", use_container_width=True)
    with ec2:
        # Summary JSON
        summary = {
            "date": TODAY, "total_fournisseurs": len(df_res),
            "score_moyen": round(float(score_moy),1),
            "niveau_global": niveau_global,
            "critiques": n_rouge, "surveillance": n_orange, "stables": n_vert,
            "pct_critiques": pct_rouge, "pct_surveillance": pct_orange, "pct_stables": pct_vert,
            "seuil_vert": p_seuil_vert, "seuil_orange": p_seuil_orange,
        }
        json_data = json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button("⬇️ Synthèse exécutive (JSON)",
                            data=json_data,
                            file_name=f"SupplyGuard_Synthese_{date.today().strftime('%Y%m%d')}.json",
                            mime="application/json", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style='border-top:1px solid #1E2D45;margin-top:40px;padding-top:24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px'>
  <div style='display:flex;align-items:center;gap:10px'>
    <div style='width:28px;height:28px;background:linear-gradient(135deg,#1E5EFF,#0A3CBF);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px'>🛡️</div>
    <span style='font-size:0.78rem;font-weight:700;color:#E8EDF5'>SupplyGuard</span>
  </div>
  <div style='font-size:0.72rem;color:#2A3F5F'>Université Mohammed V · FSJES Agdal · Master M.I.E.L · 2025–2026</div>
  <div style='font-size:0.72rem;color:#2A3F5F'>Moteur IA — Analyse multivariée avancée</div>
</div>
""", unsafe_allow_html=True)
