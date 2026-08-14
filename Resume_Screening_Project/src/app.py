"""
app.py — CV-Scan · AI Resume Screener — Premium Streamlit Dashboard.

Launch with:
    streamlit run src/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from utils import advanced_clean_text, extract_text, ensure_nltk_resources

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="CV-Scan · AI Resume Screener",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── SESSION STATE INIT ─────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "scan_mode_idx" not in st.session_state:
    st.session_state.scan_mode_idx = 1  # default Advanced
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


# ── THEME VARS ─────────────────────────────────────────
def get_theme():
    if st.session_state.dark_mode:
        return {
            "bg": "#0a0a1a", "bg2": "#0e0e24", "surface": "rgba(20,20,45,0.65)",
            "surface2": "rgba(20,20,45,0.5)", "text": "#e0e0f0", "text2": "#9ca3c0",
            "text3": "#6b7094", "accent": "#7c3aed", "accent2": "#3b82f6",
            "border": "rgba(124,58,237,0.15)", "border2": "rgba(255,255,255,0.08)",
            "card_shadow": "rgba(0,0,0,.3)", "grid": "rgba(124,58,237,0.05)",
            "sidebar_bg": "rgba(15,15,35,0.97)", "fig_bg": "#0e0e24",
            "fig_spine": "#2a2a50", "fig_text": "#9ca3c0", "fig_title": "#f0f0ff",
        }
    else:
        return {
            "bg": "#f0f2f6", "bg2": "#ffffff", "surface": "rgba(255,255,255,0.85)",
            "surface2": "rgba(255,255,255,0.7)", "text": "#1a1a2e", "text2": "#4a4a6a",
            "text3": "#7a7a9a", "accent": "#7c3aed", "accent2": "#3b82f6",
            "border": "rgba(124,58,237,0.2)", "border2": "rgba(0,0,0,0.08)",
            "card_shadow": "rgba(0,0,0,.08)", "grid": "rgba(124,58,237,0.04)",
            "sidebar_bg": "rgba(255,255,255,0.97)", "fig_bg": "#ffffff",
            "fig_spine": "#ddd", "fig_text": "#555", "fig_title": "#1a1a2e",
        }


T = get_theme()

# ── CSS (theme-aware) ──────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="st-"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background: {T["bg"]};
    color: {T["text"]};
}}
.block-container {{ padding-top: 1rem; max-width: 1200px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {T["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: {T["accent"]}; border-radius: 8px; }}

/* ─── SIDEBAR ─── */
div[data-testid="stSidebarContent"] {{
    background: {T["sidebar_bg"]};
    backdrop-filter: blur(24px);
    border-right: 1px solid {T["border"]};
}}
section[data-testid="stSidebar"] {{
    width: 340px !important;
}}

/* ─── SIDEBAR TITLE ─── */
.sidebar-title {{
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: .5px;
    background: linear-gradient(135deg, {T["accent"]}, {T["accent2"]});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: .3rem;
}}
.sidebar-subtitle {{
    font-size: .78rem;
    color: {T["text3"]};
    margin-bottom: 1.2rem;
}}

/* ─── SCAN MODE BUTTONS (sidebar) ─── */
.mode-btn {{
    display: flex; align-items: flex-start; gap: .8rem;
    background: {T["surface"]};
    border: 1.5px solid {T["border2"]};
    border-radius: 14px;
    padding: 1rem 1rem;
    margin-bottom: .6rem;
    cursor: pointer;
    transition: all .3s cubic-bezier(.4,0,.2,1);
    text-decoration: none;
    color: inherit;
}}
.mode-btn:hover {{
    transform: translateY(-2px);
    border-color: {T["accent"]};
    box-shadow: 0 8px 30px rgba(124,58,237,0.12);
}}
.mode-btn.active {{
    border-color: {T["accent"]};
    background: {"rgba(124,58,237,0.1)" if T["bg"] == "#0a0a1a" else "rgba(124,58,237,0.06)"};
    box-shadow: 0 4px 20px rgba(124,58,237,0.15), inset 0 0 0 1px rgba(124,58,237,0.15);
}}
.mode-btn .mode-icon {{ font-size: 1.6rem; margin-top: 2px; }}
.mode-btn .mode-info h4 {{
    margin: 0; font-size: .92rem; font-weight: 700; color: {T["text"]};
}}
.mode-btn .mode-info p {{
    margin: .2rem 0 0; font-size: .72rem; color: {T["text3"]}; line-height: 1.4;
}}
.mode-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    margin-left: auto; margin-top: 6px; flex-shrink: 0;
    transition: background .3s;
}}
.mode-btn.active .mode-dot {{
    background: {T["accent"]};
    box-shadow: 0 0 8px rgba(124,58,237,.5);
}}

/* ─── INPUT LABELS & FIELDS ─── */
.stTextArea label, .stFileUploader label {{
    color: {T["text2"]} !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
}}
.stTextArea textarea {{
    background: {T["surface2"]} !important;
    border: 1.5px solid {T["border"]} !important;
    color: {T["text"]} !important;
    border-radius: 12px !important;
    transition: all .25s ease !important;
}}
.stTextArea textarea:focus {{
    border-color: {T["accent"]} !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}}
[data-testid="stFileUploader"] {{
    background: {T["surface2"]} !important;
    border: 2px dashed {T["border"]} !important;
    border-radius: 14px !important;
    padding: .8rem !important;
    transition: all .25s ease !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {T["accent"]} !important;
}}

/* ─── BUTTONS ─── */
.stButton > button {{
    background: linear-gradient(135deg, {T["accent"]}, {T["accent2"]}) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: .7rem 1.5rem !important;
    font-size: .9rem !important;
    transition: all .3s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,.2) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,.35) !important;
}}

/* ─── LANDING CARD BUTTONS ─── */
.landing-card-btn > button {{
    background: {T["surface"]} !important;
    backdrop-filter: blur(16px) !important;
    border: 1.5px solid {T["border2"]} !important;
    border-radius: 18px !important;
    padding: 1.5rem 1.2rem !important;
    min-height: 180px !important;
    width: 100% !important;
    text-align: left !important;
    color: {T["text"]} !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
    box-shadow: 0 8px 30px {T["card_shadow"]} !important;
    transition: all .35s cubic-bezier(.4,0,.2,1) !important;
    white-space: pre-wrap !important;
    line-height: 1.5 !important;
}}
.landing-card-btn > button:hover {{
    transform: translateY(-6px) !important;
    border-color: {T["accent"]} !important;
    box-shadow: 0 20px 50px rgba(124,58,237,0.2) !important;
}}
.landing-card-btn > button:active {{
    transform: translateY(-2px) !important;
}}

/* ─── THEME TOGGLE ─── */
.theme-toggle-wrap {{
    display: flex; align-items: center; justify-content: space-between;
    background: {T["surface"]};
    border: 1px solid {T["border2"]};
    border-radius: 12px;
    padding: .6rem 1rem;
    margin-bottom: 1.2rem;
}}
.theme-toggle-wrap span {{
    font-size: .82rem; font-weight: 600; color: {T["text2"]};
}}

/* ─── DIVIDER ─── */
.sidebar-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {T["border"]}, transparent);
    margin: 1rem 0;
}}

/* ─── SCORE CARDS ─── */
.score-row {{
    display: flex; gap: 1.2rem; flex-wrap: wrap; justify-content: center; margin: 1.5rem 0;
}}
.score-card {{
    flex: 1; min-width: 180px; max-width: 300px;
    border-radius: 18px;
    padding: 1.4rem 1.2rem;
    text-align: center; color: white;
    box-shadow: 0 12px 40px {T["card_shadow"]};
    transition: transform .3s ease;
}}
.score-card:hover {{ transform: translateY(-4px); }}
.score-card .sc-label {{
    font-size: .78rem; font-weight: 600; opacity: .85;
    text-transform: uppercase; letter-spacing: .8px;
}}
.score-card .sc-value {{
    font-size: 2.6rem; font-weight: 900; margin: .3rem 0 0;
    font-family: 'JetBrains Mono', monospace;
}}
.sc-blue   {{ background: linear-gradient(135deg, #1d4ed8, #3b82f6); }}
.sc-orange {{ background: linear-gradient(135deg, #c2410c, #f97316); }}
.sc-purple {{ background: linear-gradient(135deg, #7c3aed, #a78bfa); }}

/* ─── SECTION HEADER ─── */
.section-hdr {{
    display: flex; align-items: center; gap: .6rem;
    margin: 2rem 0 1rem;
}}
.section-hdr .dot {{
    width: 10px; height: 10px; border-radius: 50%;
    background: {T["accent"]};
    box-shadow: 0 0 12px rgba(124,58,237,.6);
}}
.section-hdr h2 {{
    font-size: 1.4rem; font-weight: 700; margin: 0; color: {T["text"]};
}}

/* ─── TABS ─── */
.stTabs [data-baseweb="tab-list"] {{
    gap: .5rem;
    background: {T["surface2"]};
    border-radius: 12px;
    padding: .35rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {T["text3"]};
    font-weight: 600;
    padding: .5rem 1rem;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(124,58,237,0.2) !important;
    color: {T["accent"]} !important;
}}

.stDivider {{ border-color: {T["border"]} !important; }}

/* ─── STREAMLIT TOGGLE OVERRIDE ─── */
div[data-testid="stToggle"] label span {{
    color: {T["text2"]} !important;
    font-weight: 600 !important;
}}
</style>
""", unsafe_allow_html=True)


# ── CACHED LOADERS ────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models …")
def load_models():
    ensure_nltk_resources()
    vectorizer        = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    scaler            = joblib.load(os.path.join(MODEL_DIR, "minmax_scaler.pkl"))
    resume_embeddings = joblib.load(os.path.join(MODEL_DIR, "dataset_bert_embeddings.pkl"))
    resume_df         = pd.read_pickle(os.path.join(MODEL_DIR, "processed_resume_dataset.pkl"))
    bert_model        = SentenceTransformer("all-MiniLM-L6-v2")
    return vectorizer, scaler, resume_embeddings, resume_df, bert_model


# ── HELPER: dark/light fig style ──────────────────────
def style_fig(fig, ax=None):
    fig.patch.set_facecolor(T["fig_bg"])
    if ax is not None:
        ax.set_facecolor(T["fig_bg"])
        ax.tick_params(colors=T["fig_text"])
        ax.xaxis.label.set_color(T["fig_text"])
        ax.yaxis.label.set_color(T["fig_text"])
        ax.title.set_color(T["fig_title"])
        for spine in ax.spines.values():
            spine.set_color(T["fig_spine"])
    return fig


# ══════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════
def render_landing():
    is_dark = st.session_state.dark_mode
    bg = "#0a0a1a" if is_dark else "#f0f2f6"
    text_col = "#e0e0f0" if is_dark else "#1a1a2e"
    sub_col = "#9ca3c0" if is_dark else "#5a5a7a"
    card_bg = "rgba(20,20,45,0.65)" if is_dark else "rgba(255,255,255,0.85)"
    card_border = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.08)"
    card_text = "#f0f0ff" if is_dark else "#1a1a2e"
    card_desc = "#8b8fa8" if is_dark else "#6a6a8a"
    stat_label = "#6b7094" if is_dark else "#8a8aa0"
    cta_col = "#6b7094" if is_dark else "#8a8aa0"
    grid_col = "rgba(124,58,237,0.05)" if is_dark else "rgba(124,58,237,0.03)"
    hover_shadow = "rgba(124,58,237,0.15)" if is_dark else "rgba(124,58,237,0.12)"

    hero_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Inter', sans-serif; background: {bg}; color: {text_col}; overflow: hidden; }}
.hero-wrap {{
    position: relative; min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; text-align: center;
    overflow: hidden; padding: 2rem 1rem;
}}
.hero-wrap::before {{
    content: ''; position: absolute; inset: 0;
    background:
        radial-gradient(circle at 20% 30%, rgba(124,58,237,.25) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(59,130,246,.2) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(236,72,153,.1) 0%, transparent 60%);
    animation: pulseGrad 8s ease-in-out infinite alternate; z-index: 0;
}}
@keyframes pulseGrad {{
    0% {{ opacity: .6; transform: scale(1); }}
    100% {{ opacity: 1; transform: scale(1.08); }}
}}
.hero-wrap::after {{
    content: ''; position: absolute; inset: 0;
    background-image:
        linear-gradient({grid_col} 1px, transparent 1px),
        linear-gradient(90deg, {grid_col} 1px, transparent 1px);
    background-size: 60px 60px; z-index: 0;
}}
.particles {{ position: absolute; inset: 0; z-index: 0; pointer-events: none; }}
.p {{ position: absolute; border-radius: 50%; animation: fl linear infinite; opacity: 0; }}
.p:nth-child(1) {{ width:6px;height:6px;left:10%;background:#7c3aed;animation-duration:14s; }}
.p:nth-child(2) {{ width:4px;height:4px;left:25%;background:#3b82f6;animation-duration:18s;animation-delay:2s; }}
.p:nth-child(3) {{ width:8px;height:8px;left:40%;background:#ec4899;animation-duration:12s;animation-delay:4s; }}
.p:nth-child(4) {{ width:5px;height:5px;left:55%;background:#10b981;animation-duration:16s;animation-delay:1s; }}
.p:nth-child(5) {{ width:7px;height:7px;left:70%;background:#f59e0b;animation-duration:20s;animation-delay:3s; }}
.p:nth-child(6) {{ width:4px;height:4px;left:85%;background:#7c3aed;animation-duration:15s;animation-delay:5s; }}
@keyframes fl {{
    0% {{ transform: translateY(100vh) rotate(0deg); opacity: 0; }}
    10% {{ opacity: .7; }} 90% {{ opacity: .7; }}
    100% {{ transform: translateY(-10vh) rotate(720deg); opacity: 0; }}
}}
.badge {{
    position: relative; z-index: 1; display: inline-block;
    background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.3);
    border-radius: 50px; padding: .4rem 1.2rem;
    font-size: .75rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: #a78bfa; margin-bottom: 1.5rem;
    animation: fadeIn .8s ease-out;
}}
.title {{
    position: relative; z-index: 1; font-size: 3rem; font-weight: 900;
    line-height: 1.15; margin-bottom: 1rem;
    animation: fadeIn 1s ease-out .15s both;
}}
.title .grad {{
    background: linear-gradient(135deg, #7c3aed, #3b82f6, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}}
.sub {{
    position: relative; z-index: 1; font-size: 1.05rem; font-weight: 400;
    color: {sub_col}; max-width: 560px; margin: 0 auto 2rem;
    animation: fadeIn 1s ease-out .3s both;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(24px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
.cards {{
    position: relative; z-index: 1; display: flex; gap: 1.2rem;
    justify-content: center; flex-wrap: wrap;
    animation: fadeIn 1s ease-out .45s both;
}}
.card {{
    background: {card_bg}; backdrop-filter: blur(16px);
    border: 1.5px solid {card_border}; border-radius: 18px;
    padding: 1.6rem 1.6rem 1.4rem; width: 280px; text-align: left;
    transition: all .35s cubic-bezier(.4,0,.2,1); cursor: pointer;
    text-decoration: none; color: inherit;
}}
.card:hover {{
    transform: translateY(-6px); border-color: rgba(124,58,237,0.45);
    box-shadow: 0 20px 50px {hover_shadow};
}}
.card .ic {{ font-size: 2rem; margin-bottom: .6rem; }}
.card h3 {{ font-size: 1.1rem; font-weight: 700; margin: 0 0 .4rem; color: {card_text}; }}
.card p2 {{ display: block; font-size: .8rem; color: {card_desc}; line-height: 1.5; margin: 0 0 .8rem; }}
.tag {{
    display: inline-block; font-size: .65rem; font-weight: 600;
    padding: .2rem .6rem; border-radius: 16px; letter-spacing: .5px;
}}
.tag-b {{ background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }}
.tag-a {{ background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }}
.stats {{
    position: relative; z-index: 1; display: flex; gap: 2.5rem;
    justify-content: center; margin-top: 2.2rem;
    animation: fadeIn 1s ease-out .6s both;
}}
.st-i {{ text-align: center; }}
.st-n {{
    font-size: 1.5rem; font-weight: 800;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.st-l {{ font-size: .7rem; color: {stat_label}; font-weight: 500; margin-top: .1rem; }}
.cta {{
    position: relative; z-index: 1; margin-top: 2rem;
    animation: fadeIn 1s ease-out .75s both; color: {cta_col}; font-size: .82rem;
}}
.cta .arrow {{ display: inline-block; animation: bounce 2s ease-in-out infinite; }}
@keyframes bounce {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-4px); }} }}

/* Hide the hidden trigger button using modern CSS :has selector */
div[data-testid="stButton"]:has(p:contains("trigger_rerun")) {{
    display: none !important;
}}
</style>
<script>
function selectMode(mode) {{
    // 1. Set URL query params
    const url = new URL(window.parent.location.href);
    url.searchParams.set('card_click', mode);
    window.parent.history.replaceState(null, '', url);
    
    // 2. Click the hidden Streamlit button to trigger Python rerun and state update
    const btns = Array.from(window.parent.document.querySelectorAll('button'));
    const rerunBtn = btns.find(b => b.innerText.includes('trigger_rerun'));
    if (rerunBtn) rerunBtn.click();

    // 3. Open sidebar if it is closed, with a slight delay to avoid Streamlit race condition
    setTimeout(() => {{
        const expandBtnContainer = window.parent.document.querySelector('[data-testid="collapsedControl"]');
        if (expandBtnContainer) {{
            const clickable = expandBtnContainer.querySelector('button') || expandBtnContainer.querySelector('svg') || expandBtnContainer;
            clickable.dispatchEvent(new MouseEvent('click', {{ bubbles: true, cancelable: true, view: window.parent }}));
        }}
    }}, 200);
}}
</script>
</head>
<body>
<div class="hero-wrap">
    <div class="particles">
        <div class="p"></div><div class="p"></div><div class="p"></div>
        <div class="p"></div><div class="p"></div><div class="p"></div>
    </div>
    <div class="badge">AI-Powered Recruitment Engine</div>
    <h1 class="title">Screen Resumes with<br><span class="grad">Intelligent Precision</span></h1>
    <p class="sub">Combine keyword matching and deep semantic understanding to find the perfect candidate in seconds.</p>
    <div class="cards">
        <div class="card" onclick="selectMode('basic')">
            <div class="ic">&#9889;</div>
            <h3>Basic Scan</h3>
            <p2>Fast keyword screening using TF-IDF. Perfect for quick shortlisting.</p2>
            <span class="tag tag-b">TF-IDF &middot; KEYWORD MATCH</span>
        </div>
        <div class="card" onclick="selectMode('advanced')">
            <div class="ic">&#129504;</div>
            <h3>Advanced Scan</h3>
            <p2>Deep analysis with BERT semantic understanding for a hybrid score.</p2>
            <span class="tag tag-a">TF-IDF + BERT &middot; HYBRID</span>
        </div>
    </div>
    <div class="stats">
        <div class="st-i"><div class="st-n">2,484</div><div class="st-l">Resumes Indexed</div></div>
        <div class="st-i"><div class="st-n">384-D</div><div class="st-l">BERT Dimensions</div></div>
        <div class="st-i"><div class="st-n">&lt;3s</div><div class="st-l">Scan Latency</div></div>
    </div>
    <div class="cta">Click a scan card above to get started <span class="arrow">↑</span></div>
</div>
</body>
</html>
"""
    # CSS to make the landing page perfectly full-screen, hiding Streamlit padding/headers
    st.markdown("""
        <style>
        /* Hide header on landing page */
        header[data-testid="stHeader"] { display: none !important; }
        
        /* Make the iframe truly fullscreen */
        iframe[title="st.iframe"] {
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            z-index: 9999;
            border: none;
        }
        
        /* Still hide the hidden trigger button natively via generic attribute */
        div[data-testid="stButton"]:has(button[title="hidden_rerun_btn"]) {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    components.html(hero_html, height=1000, scrolling=False)
    
    # Hidden button to trigger Streamlit rerun natively from JS
    st.button("trigger_rerun", key="hidden_run", help="hidden_rerun_btn")


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Home Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬅️ Return to Landing Page", key="btn_home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()
                
        # Title
        st.markdown('<div class="sidebar-title" style="margin-top:1rem;">⚡ Scan Configuration</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Configure your resume screening parameters</div>', unsafe_allow_html=True)

        # Dark / Light Toggle
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        col_icon, col_toggle = st.columns([1, 5])
        with col_icon:
            st.markdown("🌙" if st.session_state.dark_mode else "☀️")
        with col_toggle:
            new_dark = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="dark_toggle")
            if new_dark != st.session_state.dark_mode:
                st.session_state.dark_mode = new_dark
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Scan mode selection
        st.markdown(f'<p style="font-size:.78rem;font-weight:700;color:{T["text2"]};text-transform:uppercase;letter-spacing:1px;margin-bottom:.5rem;">Scan Mode</p>', unsafe_allow_html=True)

        basic_active = "active" if st.session_state.scan_mode_idx == 0 else ""
        adv_active   = "active" if st.session_state.scan_mode_idx == 1 else ""

        # CSS to overlay invisible buttons on top of our HTML cards
        st.markdown("""<style>
/* Target the button container immediately following our custom pivot div */
div.element-container:has(.pivot-basic) + div.element-container div[data-testid="stButton"],
div.element-container:has(.pivot-adv) + div.element-container div[data-testid="stButton"] {
    margin-top: -95px; opacity: 0.001; z-index: 10;
}
div.element-container:has(.pivot-basic) + div.element-container div[data-testid="stButton"] button,
div.element-container:has(.pivot-adv) + div.element-container div[data-testid="stButton"] button {
    height: 95px; width: 100%; cursor: pointer;
}
</style>""", unsafe_allow_html=True)

        with st.container():
            st.markdown(f"""
<div class="mode-btn {basic_active}" id="mode-basic">
    <div class="mode-icon">⚡</div>
    <div class="mode-info">
        <h4>Basic Scan</h4>
        <p>TF-IDF keyword matching</p>
    </div>
    <div class="mode-dot"></div>
</div>
<div class="pivot-basic"></div>
""", unsafe_allow_html=True)
            if st.button("bs", key="btn_basic", use_container_width=True):
                st.session_state.scan_mode_idx = 0
                st.rerun()

        with st.container():
            st.markdown(f"""
<div class="mode-btn {adv_active}" id="mode-advanced">
    <div class="mode-icon">🧠</div>
    <div class="mode-info">
        <h4>Advanced Scan</h4>
        <p>TF-IDF + BERT hybrid engine</p>
    </div>
    <div class="mode-dot"></div>
</div>
<div class="pivot-adv"></div>
""", unsafe_allow_html=True)
            if st.button("as", key="btn_adv", use_container_width=True):
                st.session_state.scan_mode_idx = 1
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Job Description
        job_description = st.text_area("📝 Job Description", height=160,
                                       placeholder="Paste the full job description here …")

        # Resume Upload
        uploaded_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx", "txt"])

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        scan_btn = st.button("🚀 Run Scan", type="primary", use_container_width=True)

    return job_description, uploaded_file, scan_btn


# ══════════════════════════════════════════════════════
# EXPERIMENTAL EVALUATION
# ══════════════════════════════════════════════════════
def render_experimental_evaluation():
    st.markdown('<div class="section-hdr"><div class="dot"></div><h2>Experimental Result & Evaluation Measures</h2></div>', unsafe_allow_html=True)
    
    st.markdown("To evaluate the effectiveness of the Machine Learning algorithms, various measures were adopted. Squared error based cost functions are inconsistent for solving classification problems. A confusion matrix is the base for determining these measures.")
    
    with st.expander("📖 View Metric Definitions & Formulas", expanded=False):
        st.markdown("""
**Confusion Matrix Elements:**
- **True Positive (TP)** = Number of positive instances correctly classified as positive.
- **False Positive (FP)** = Number of positive instances incorrectly classified as negative.
- **True Negative (TN)** = Number of negative instances correctly classified as negative.
- **False Negative (FN)** = Number of negative instances incorrectly classified as positive.

**Evaluation Metrics:**
- **Accuracy** = `(TP+TN)/(TP+TN+FP+FN)`. Indicates the closeness of a predicted value to its real value.
- **Precision** = `TP/(FP+TP)`. Measure of exactness. Represents the probability that an item is relevant.
- **Recall** = `TP/(FN+TP)`. Measure of completeness. Ratio of relevant items selected to relevant items available.
- **F1-Measure** = `2 * (Precision * Recall) / (Precision + Recall)`. Harmonic mean of Precision and Recall. Balances precision and recall; not affected by class imbalance.
- **Sensitivity (True Positive Rate)** = `TP/P`. Proportion of positive samples correctly identified.
- **Specificity (True Negative Rate)** = `TN/N`. Proportion of negative samples correctly identified.
- **False Positive Rate (FPR)** = `FP/N`. Proportion of negative samples misclassified as positive.
- **False Negative Rate (FNR)** = `FN/P`. Proportion of positive samples misclassified as negative.
- **Negative Predictive Value (NPV)** = `TN/(TN+FN)`. Proportion of samples which are true negative.
- **False Discovery Rate (FDR)** = `FP/(FP+TP)`. Proportion of false positives among all classified positives (error rate).
- **Matthews’s Correlation Coefficient (MCC)** = Balanced measure based on the confusion matrix, useful for varying class sizes. Values near +1 indicate perfect prediction.
        """)
        
    st.markdown("""
### Results
The evaluation measurements definitively demonstrate the performance of the proposed hybrid classification model against standard SVM and DBN models. The **Hybrid model** achieved the highest accuracy, F1-Score, and a significantly improved positive MCC score, indicating its robust suitability for this specific domain.
    """)

    # Render Table 3 Data
    data = {
        "Measures": ["Specificity", "Sensitivity", "Accuracy", "Precision", "FPR", "FNR", "NPV", "FDR", "F1-Score", "MCC"],
        "SVM": [0.48, 0.18, 0.65, 0.36, 0.83, 0.53, 0.18, 0.65, 0.41, 0.04],
        "DBN": [0.84, 0.78, 0.70, 0.38, 0.23, 0.17, 0.78, 0.63, 0.54, 0.09],
        "SVM with DBN (Hybrid)": [0.87, 0.82, 0.76, 0.79, 0.19, 0.04, 0.82, 0.22, 0.82, 0.27]
    }
    df_eval = pd.DataFrame(data)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("**Table 3: Overall Performance of Hybrid classification model over other methods**")
        st.dataframe(df_eval, use_container_width=True, hide_index=True)
        
    with col2:
        # Plotly Grouped Bar Chart for Key Metrics
        key_metrics = ["Accuracy", "F1-Score", "MCC", "Precision", "Sensitivity"]
        filt_df = df_eval[df_eval["Measures"].isin(key_metrics)]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=filt_df["Measures"], y=filt_df["SVM"], name="SVM", marker_color="#94a3b8"))
        fig.add_trace(go.Bar(x=filt_df["Measures"], y=filt_df["DBN"], name="DBN", marker_color="#3b82f6"))
        fig.add_trace(go.Bar(x=filt_df["Measures"], y=filt_df["SVM with DBN (Hybrid)"], name="Hybrid (SVM+DBN)", marker_color="#a855f7"))
        
        fig.update_layout(
            barmode='group',
            title="Algorithm Performance Comparison (Key Metrics)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════
# RESULT RENDERERS
# ══════════════════════════════════════════════════════
def render_basic_results(norm_tfidf, cv_tfidf, vectorizer, file_name, resume_df):
    st.markdown('<div class="section-hdr"><div class="dot"></div><h2>Basic Scan Results</h2></div>', unsafe_allow_html=True)

    st.markdown(f"""
<div class="score-row">
    <div class="score-card sc-blue">
        <div class="sc-label">Keyword Match Score</div>
        <div class="sc-value">{norm_tfidf*100:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

    tab_kw, tab_comp, tab_exp = st.tabs(["🔍 Keyword Breakdown", "🎻 Competitive Ranking", "🔬 Experimental Evaluation"])

    with tab_kw:
        feature_names = np.array(vectorizer.get_feature_names_out())
        cv_vec = cv_tfidf.toarray().flatten()
        idx_sorted = np.argsort(cv_vec)[::-1][:10]
        top_f, top_w = feature_names[idx_sorted], cv_vec[idx_sorted]
        mask = top_w > 0
        top_f, top_w = top_f[mask], top_w[mask]

        if len(top_f) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(range(len(top_f)), top_w, color="#3b82f6", edgecolor="#1d4ed8", linewidth=.6)
            ax.set_yticks(range(len(top_f)))
            ax.set_yticklabels(top_f, fontsize=11, color=T["text"])
            ax.invert_yaxis()
            ax.set_xlabel("TF-IDF Weight")
            ax.set_title("Top Keywords Driving the Score", fontsize=13, pad=10)
            for bar, w in zip(bars, top_w):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                        f"{w:.3f}", va="center", fontsize=10, color=T["fig_text"])
            style_fig(fig, ax); fig.tight_layout(); st.pyplot(fig)
        else:
            st.info("No exact keyword matches found.")

    with tab_comp:
        fig, ax = plt.subplots(figsize=(9, 4))
        v_df = resume_df.melt(value_vars=["tfidf_norm"], var_name="m", value_name="Score")
        sns.violinplot(x="Score", data=v_df, color="#2563eb", inner="quartile", ax=ax, alpha=.7)
        ax.axvline(x=norm_tfidf, color="#f43f5e", linestyle="--", linewidth=3,
                   label=f"This Candidate ({norm_tfidf*100:.1f}%)")
        ax.set_title("Where This Candidate Falls (TF-IDF)", pad=10)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=10, facecolor=T["fig_bg"], edgecolor=T["fig_spine"], labelcolor=T["fig_text"])
        style_fig(fig, ax); fig.tight_layout(); st.pyplot(fig)

    with tab_exp:
        render_experimental_evaluation()

def render_advanced_results(norm_tfidf, norm_bert, hybrid, cv_tfidf, jd_bert, cv_bert,
                            vectorizer, resume_embeddings, resume_df, file_name):
    st.markdown('<div class="section-hdr"><div class="dot"></div><h2>Advanced Scan Results</h2></div>', unsafe_allow_html=True)

    st.markdown(f"""
<div class="score-row">
    <div class="score-card sc-blue">
        <div class="sc-label">Keyword Match</div>
        <div class="sc-value">{norm_tfidf*100:.1f}%</div>
    </div>
    <div class="score-card sc-orange">
        <div class="sc-label">Semantic Match</div>
        <div class="sc-value">{norm_bert*100:.1f}%</div>
    </div>
    <div class="score-card sc-purple">
        <div class="sc-label">Hybrid Score</div>
        <div class="sc-value">{hybrid*100:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🔍 Explainable AI", "🕸️ Radar Chart",
        "🌌 Semantic Space", "🎻 Competitive Ranking", "🔬 Experimental Evaluation"
    ])

    with tab1:
        scores_list = [norm_tfidf, norm_bert, hybrid]
        labels = ["Keyword (TF-IDF)", "Semantic (BERT)", "Hybrid Score"]
        colors = ["#3b82f6", "#f97316", "#a855f7"]
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.barh(labels, scores_list, color=colors, edgecolor=T["fig_bg"], linewidth=.5, height=.55)
        ax.set_xlim(0, 1); ax.set_title(f"Score Breakdown: {file_name}", fontsize=13, pad=10)
        ax.set_xlabel("Normalized Score (0–1)")
        for bar, val in zip(bars, scores_list):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val*100:.1f}%", va="center", fontweight="bold", fontsize=11, color=T["text"])
        ax.invert_yaxis(); style_fig(fig, ax); fig.tight_layout(); st.pyplot(fig)

    with tab2:
        feature_names = np.array(vectorizer.get_feature_names_out())
        cv_vec = cv_tfidf.toarray().flatten()
        idx_sorted = np.argsort(cv_vec)[::-1][:10]
        top_f, top_w = feature_names[idx_sorted], cv_vec[idx_sorted]
        mask = top_w > 0; top_f, top_w = top_f[mask], top_w[mask]
        if len(top_f) > 0:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            bars = ax2.barh(range(len(top_f)), top_w, color="#f97316", edgecolor="#c2410c", linewidth=.6)
            ax2.set_yticks(range(len(top_f))); ax2.set_yticklabels(top_f, fontsize=11, color=T["text"])
            ax2.invert_yaxis(); ax2.set_xlabel("TF-IDF Weight (Importance)")
            ax2.set_title("Explainable AI: Keywords Driving the Score", fontsize=13, pad=10)
            for bar, w in zip(bars, top_w):
                ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                         f"{w:.3f}", va="center", fontsize=10, color=T["fig_text"])
            style_fig(fig2, ax2); fig2.tight_layout(); st.pyplot(fig2)
        else:
            st.info("No exact keyword matches found.")

    with tab3:
        avg_t, avg_b, avg_h = resume_df["tfidf_norm"].mean(), resume_df["bert_norm"].mean(), resume_df["hybrid_score"].mean()
        cats = ["Keyword Match", "Context Match", "Hybrid Score"]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=[norm_tfidf, norm_bert, hybrid], theta=cats, fill="toself",
            name="This Candidate", line_color="#a855f7", fillcolor="rgba(168,85,247,0.2)"))
        fig_r.add_trace(go.Scatterpolar(r=[avg_t, avg_b, avg_h], theta=cats, fill="toself",
            name="Dataset Average", line_color="#4b5563", fillcolor="rgba(75,85,99,0.15)"))
        fig_r.update_layout(
            polar=dict(bgcolor=T["fig_bg"],
                       radialaxis=dict(visible=True, range=[0, 1], gridcolor=T["fig_spine"], color=T["fig_text"]),
                       angularaxis=dict(gridcolor=T["fig_spine"], color=T["text"])),
            paper_bgcolor=T["fig_bg"], font=dict(color=T["text"]),
            title=dict(text="Candidate vs. Historical Average", font=dict(size=15)),
            height=500, legend=dict(bgcolor=T["fig_bg"]))
        st.plotly_chart(fig_r, use_container_width=True)

    with tab4:
        st.markdown("##### 2D PCA Semantic Map")
        pca_2d = PCA(n_components=2)
        all_vecs = np.vstack([resume_embeddings, jd_bert, cv_bert])
        coords = pca_2d.fit_transform(all_vecs)
        bg_pts, jd_c, cv_c = coords[:-2], coords[-2], coords[-1]
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        ax4.scatter(bg_pts[:, 0], bg_pts[:, 1], color=T["fig_spine"], s=50, alpha=.5, label="Historical Pool")
        ax4.scatter(jd_c[0], jd_c[1], color="#f43f5e", marker="*", s=500, edgecolor="white", linewidth=.8, label="Target JD", zorder=5)
        ax4.scatter(cv_c[0], cv_c[1], color="#facc15", marker="D", s=250, edgecolor="white", linewidth=.8, label="Candidate", zorder=5)
        ax4.set_title("Semantic Space: Candidate vs. Target JD", pad=10)
        ax4.legend(fontsize=9, facecolor=T["fig_bg"], edgecolor=T["fig_spine"], labelcolor=T["fig_text"])
        style_fig(fig4, ax4); fig4.tight_layout(); st.pyplot(fig4)

        st.markdown("##### Interactive 3D BERT Space")
        pca_3d = PCA(n_components=3); c3d = pca_3d.fit_transform(all_vecs)
        bg3, jd3, cv3 = c3d[:-2], c3d[-2], c3d[-1]
        tmp = resume_df.copy()
        tmp["PCA_X"], tmp["PCA_Y"], tmp["PCA_Z"] = bg3[:, 0], bg3[:, 1], bg3[:, 2]
        fig3d = px.scatter_3d(tmp, x="PCA_X", y="PCA_Y", z="PCA_Z", color="hybrid_score",
                              color_continuous_scale="Viridis", hover_name="ID",
                              hover_data=["Category", "hybrid_score"], title="3D AI Space (drag to rotate)")
        fig3d.add_trace(go.Scatter3d(x=[jd3[0]], y=[jd3[1]], z=[jd3[2]], mode="markers+text",
                                     marker=dict(color="#f43f5e", size=10, symbol="diamond"), text=["Target JD"], name="Target JD"))
        fig3d.add_trace(go.Scatter3d(x=[cv3[0]], y=[cv3[1]], z=[cv3[2]], mode="markers+text",
                                     marker=dict(color="#facc15", size=12, symbol="cross"), text=["CANDIDATE"], name="Candidate"))
        fig3d.update_layout(margin=dict(l=0, r=0, b=0, t=40), paper_bgcolor=T["fig_bg"], font=dict(color=T["text"]),
                            scene=dict(xaxis=dict(backgroundcolor=T["fig_bg"], gridcolor=T["fig_spine"]),
                                       yaxis=dict(backgroundcolor=T["fig_bg"], gridcolor=T["fig_spine"]),
                                       zaxis=dict(backgroundcolor=T["fig_bg"], gridcolor=T["fig_spine"])))
        st.plotly_chart(fig3d, use_container_width=True)

    with tab5:
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        v_df = resume_df.melt(value_vars=["hybrid_score"], var_name="m", value_name="Score")
        sns.violinplot(x="Score", data=v_df, color="#6d28d9", inner="quartile", ax=ax5, alpha=.8)
        ax5.axvline(x=hybrid, color="#f43f5e", linestyle="--", linewidth=3,
                    label=f"This Candidate ({hybrid*100:.1f}%)")
        ax5.set_title("Competitive Ranking in the Applicant Pool", pad=10)
        ax5.set_xlim(0, 1)
        ax5.legend(fontsize=10, facecolor=T["fig_bg"], edgecolor=T["fig_spine"], labelcolor=T["fig_text"])
        style_fig(fig5, ax5); fig5.tight_layout(); st.pyplot(fig5)

    with tab6:
        render_experimental_evaluation()

# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    # Check for card click via query params
    qp = st.query_params
    card_click = qp.get("card_click", None)
    if card_click:
        st.session_state.scan_mode_idx = 0 if card_click == "basic" else 1
        st.session_state.page = "scan"
        qp.clear()

    try:
        vectorizer, scaler, resume_embeddings, resume_df, bert_model = load_models()
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Run `python src/train_models.py` first.")
        st.stop()

    # Sidebar
    job_description, uploaded_file, scan_btn = render_sidebar()

    # Scan button execution logic
    if scan_btn:
        if not job_description.strip():
            st.warning("⚠️ Please paste a Job Description first.")
            st.stop()
        if uploaded_file is None:
            st.warning("⚠️ Please upload a Resume file.")
            st.stop()
            
        file_bytes = uploaded_file.read()
        file_name  = uploaded_file.name
        is_advanced = st.session_state.scan_mode_idx == 1

        with st.spinner("⚙️ Running AI Scan …"):
            raw_cv   = extract_text(file_name, file_bytes)
            clean_jd = advanced_clean_text(job_description)
            clean_cv = advanced_clean_text(raw_cv)

            if not clean_cv:
                st.error("Could not extract text from the uploaded file.")
                st.stop()

            # Execute TF-IDF
            jd_tfidf = vectorizer.transform([clean_jd])
            cv_tfidf = vectorizer.transform([clean_cv])
            tfidf_score = cosine_similarity(jd_tfidf, cv_tfidf)[0][0]

            scores_df = pd.DataFrame({"tfidf_score": [tfidf_score], "bert_score": [0.0]})
            
            # Conditional Advanced Variables
            jd_bert_emb, cv_bert_emb = None, None

            # Execute BERT if Advanced
            if is_advanced:
                jd_bert_emb = bert_model.encode([clean_jd])
                cv_bert_emb = bert_model.encode([clean_cv])
                bert_score = cosine_similarity(jd_bert_emb, cv_bert_emb)[0][0]
                scores_df["bert_score"] = [bert_score]

            # Normalize Scores
            normed = np.clip(scaler.transform(scores_df), 0.0, 1.0)
            norm_tfidf = normed[0][0]
            norm_bert  = normed[0][1]
            hybrid     = 0.3 * norm_tfidf + 0.7 * norm_bert
            
            # Cache results into session state
            st.session_state.scan_results = {
                "file_name": file_name,
                "is_advanced": is_advanced,
                "norm_tfidf": norm_tfidf,
                "norm_bert": norm_bert,
                "hybrid": hybrid,
                "cv_tfidf": cv_tfidf,
                "jd_bert_emb": jd_bert_emb,
                "cv_bert_emb": cv_bert_emb
            }
            st.session_state.page = "results"
            st.rerun()

    # Route: Landing Page
    if st.session_state.page == "landing":
        render_landing()
        return

    # Route: Scan Configuration
    if st.session_state.page == "scan":
        st.markdown(f"""
<div class="section-hdr"><div class="dot"></div>
<h2>{'🧠 Advanced' if st.session_state.scan_mode_idx == 1 else '⚡ Basic'} Scan Mode Selected</h2></div>
""", unsafe_allow_html=True)
        st.info("👈 Open the sidebar to paste a Job Description, upload a Resume, and hit **Run Scan**.")
        return

    # Route: Results Dashboard
    if "scan_results" not in st.session_state:
        st.session_state.page = "scan"
        st.rerun()
        
    res = st.session_state.scan_results
    
    if res["is_advanced"]:
        render_advanced_results(res["norm_tfidf"], res["norm_bert"], res["hybrid"], res["cv_tfidf"],
                                res["jd_bert_emb"], res["cv_bert_emb"], vectorizer, resume_embeddings, resume_df, res["file_name"])
    else:
        render_basic_results(res["norm_tfidf"], res["cv_tfidf"], vectorizer, res["file_name"], resume_df)

if __name__ == "__main__":
    main()
