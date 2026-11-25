"""
ui.py — Design system for the CV-Scan dashboard: global CSS + HTML components.

All colours are expressed with CSS ``light-dark()`` so custom components follow
Streamlit's light / dark theme switch automatically (Streamlit sets
``color-scheme`` on the app root). Any user-provided text that ends up in HTML
is escaped with ``html.escape``.
"""

from __future__ import annotations

from html import escape

import streamlit as st

# ── Series identity (validated categorical slots 1–3, light / dark steps) ──
SERIES = {
    "keyword": {"light": "#257de0", "dark": "#338ced", "label": "Keyword (TF-IDF)"},
    "semantic": {"light": "#f47230", "dark": "#e76624", "label": "Semantic (BERT)"},
    "hybrid": {"light": "#00b894", "dark": "#00a888", "label": "Hybrid"},
}

CSS = """
<style>
/* ───────────────────────── tokens ───────────────────────── */
.stApp {
  --cv-surface: light-dark(rgba(255,255,255,.84), rgba(20,23,36,.66));
  --cv-surface-solid: light-dark(#ffffff, #141724);
  --cv-surface-sunken: light-dark(#f0f2f8, #0f121c);
  --cv-border: light-dark(rgba(22,25,39,.09), rgba(255,255,255,.075));
  --cv-border-strong: light-dark(rgba(22,25,39,.17), rgba(255,255,255,.15));
  --cv-text: light-dark(#161927, #e7e9f2);
  --cv-text-2: light-dark(#585e73, #a3a9be);
  --cv-text-3: light-dark(#7c8297, #737a92);
  --cv-accent: light-dark(#5b4cdb, #9a8cf8);
  --cv-accent-soft: light-dark(rgba(91,76,219,.08), rgba(139,124,246,.13));
  --cv-accent-ring: light-dark(rgba(91,76,219,.25), rgba(139,124,246,.32));
  --cv-grad-1: light-dark(#5b4cdb, #9a8cf8);
  --cv-grad-2: light-dark(#2563eb, #60a5fa);
  --cv-grad-3: light-dark(#0b8f78, #2dd4bf);
  --cv-btn: linear-gradient(135deg, #6152e6 0%, #3b6fe8 100%);
  --cv-shadow-1: light-dark(rgba(16,24,40,.05), rgba(0,0,0,.35));
  --cv-shadow-2: light-dark(rgba(16,24,40,.10), rgba(0,0,0,.55));
  --cv-grid: light-dark(rgba(22,25,39,.055), rgba(255,255,255,.045));
  --cv-kw: light-dark(#257de0, #338ced);
  --cv-sem: light-dark(#f47230, #e76624);
  --cv-hyb: light-dark(#00b894, #00a888);
  --cv-neutral: light-dark(#a3a8b8, #5d6378);
  --cv-good: #0aac23;  --cv-good-ink: light-dark(#006c10, #53c556);
  --cv-warn: #fbb315;  --cv-warn-ink: light-dark(#8a5a00, #fbc34a);
  --cv-crit: #dc3c3d;  --cv-crit-ink: light-dark(#b42324, #f17474);
}

/* ───────────────────────── canvas ───────────────────────── */
[data-testid="stAppViewContainer"] {
  background-image:
    radial-gradient(58rem 30rem at 8% -6%, light-dark(rgba(124,108,246,.12), rgba(124,108,246,.19)), transparent 64%),
    radial-gradient(46rem 26rem at 94% -4%, light-dark(rgba(59,130,246,.10), rgba(59,130,246,.14)), transparent 62%),
    radial-gradient(40rem 26rem at 50% 108%, light-dark(rgba(20,184,166,.07), rgba(20,184,166,.08)), transparent 60%);
  background-attachment: fixed;
}
header[data-testid="stHeader"] {
  background: light-dark(rgba(246,247,251,.72), rgba(11,13,20,.62));
  backdrop-filter: saturate(170%) blur(16px);
  -webkit-backdrop-filter: saturate(170%) blur(16px);
  border-bottom: 1px solid var(--cv-border);
}
[data-testid="stMainBlockContainer"] { max-width: 1260px; padding-top: 4.6rem; padding-bottom: 3rem; }
[data-testid="stAppDeployButton"] { display: none; }  /* not relevant for this project; keeps the ⋮ menu's theme switcher */
[data-testid="stTopNavLink"] { border-radius: 10px; transition: background .15s ease; }
[data-testid="stTopNavLink"]:hover { background: var(--cv-accent-soft); }
::selection { background: var(--cv-accent-ring); }

/* ───────────────────────── typography ───────────────────────── */
.cv-ms { font-family: "Material Symbols Rounded"; font-weight: 400; font-style: normal; line-height: 1; letter-spacing: normal;
  text-transform: none; display: inline-block; white-space: nowrap; direction: ltr; font-feature-settings: "liga";
  -webkit-font-smoothing: antialiased; user-select: none; }
.cv-eyebrow { font-size: .74rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--cv-accent); margin-bottom: .35rem; }
.cv-h2 { font-size: 1.55rem; font-weight: 750; letter-spacing: -.022em; color: var(--cv-text); margin: 0 0 .3rem; line-height: 1.2; }
.cv-h3 { font-size: 1.08rem; font-weight: 700; letter-spacing: -.012em; color: var(--cv-text); margin: 0 0 .2rem; }
.cv-muted { color: var(--cv-text-2); font-size: .92rem; line-height: 1.55; margin: 0; }
.cv-small { color: var(--cv-text-3); font-size: .8rem; }
.cv-section { margin: 2.2rem 0 1rem; }
.cv-page-head { margin: .4rem 0 1.4rem; }
.cv-page-head h1 { font-size: clamp(1.8rem, 3.2vw, 2.35rem); font-weight: 800; letter-spacing: -.03em; margin: 0 0 .35rem; color: var(--cv-text); line-height: 1.12; }

/* ───────────────────────── surfaces ───────────────────────── */
.cv-card, div[class*="st-key-card"] {
  background: var(--cv-surface);
  border: 1px solid var(--cv-border);
  border-radius: 18px;
  box-shadow: 0 1px 2px var(--cv-shadow-1), 0 18px 40px -26px var(--cv-shadow-2);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
}
.cv-card { padding: 1.15rem 1.25rem; }
div[class*="st-key-card"] { padding: 1.2rem 1.3rem 1.3rem; }

/* ───────────────────────── widgets ───────────────────────── */
[data-testid="stBaseButton-primary"] {
  background: var(--cv-btn) !important; border: none !important; color: #fff !important;
  font-weight: 650 !important; letter-spacing: .005em;
  box-shadow: 0 10px 26px -12px rgba(79,70,229,.85), inset 0 1px 0 rgba(255,255,255,.18) !important;
  transition: transform .15s ease, box-shadow .2s ease, filter .2s ease !important;
}
[data-testid="stBaseButton-primary"]:hover { transform: translateY(-1px); filter: brightness(1.08); box-shadow: 0 14px 30px -12px rgba(79,70,229,.95) !important; }
[data-testid="stBaseButton-primary"]:active { transform: translateY(0); }
[data-testid="stBaseButton-secondary"] { background: var(--cv-surface-solid) !important; border: 1px solid var(--cv-border-strong) !important; font-weight: 600 !important; transition: border-color .15s ease, color .15s ease, transform .15s ease !important; }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--cv-accent) !important; color: var(--cv-accent) !important; transform: translateY(-1px); }
div[class*="st-key-cta"] button { min-height: 3rem; padding: 0 1.4rem !important; font-size: 1rem !important; }
[data-testid="stFileUploaderDropzone"] {
  border: 1.5px dashed var(--cv-border-strong) !important; border-radius: 14px !important;
  background: var(--cv-surface-sunken) !important; transition: border-color .15s ease, background .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--cv-accent) !important; background: var(--cv-accent-soft) !important; }
[data-testid="stTextArea"] textarea { font-size: .92rem; line-height: 1.55; }
[data-testid="stTabs"] [role="tablist"] {
  gap: .2rem; padding: .28rem; border-radius: 13px; width: fit-content; max-width: 100%;
  background: var(--cv-surface-sunken); border: 1px solid var(--cv-border); overflow-x: auto;
}
[data-testid="stTabs"] [role="tab"] { border-radius: 10px; padding: .42rem .9rem !important; height: auto; margin: 0 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: var(--cv-surface-solid); box-shadow: 0 1px 2px var(--cv-shadow-1), 0 4px 12px -6px var(--cv-shadow-2); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
[data-testid="stExpander"] details { border-radius: 14px; background: var(--cv-surface); }

/* ───────────────────────── hero ───────────────────────── */
.cv-hero { position: relative; isolation: isolate; text-align: center; padding: 3.2rem 1rem 1.6rem; }
.cv-hero::before {
  content: ""; position: absolute; inset: -1rem 0 -2rem; z-index: -1;
  background-image: linear-gradient(var(--cv-grid) 1px, transparent 1px), linear-gradient(90deg, var(--cv-grid) 1px, transparent 1px);
  background-size: 46px 46px;
  -webkit-mask-image: radial-gradient(ellipse 62% 72% at 50% 34%, #000 25%, transparent 74%);
  mask-image: radial-gradient(ellipse 62% 72% at 50% 34%, #000 25%, transparent 74%);
}
.cv-badge {
  display: inline-flex; align-items: center; gap: .55rem; padding: .4rem .9rem; border-radius: 999px;
  border: 1px solid var(--cv-accent-ring); background: var(--cv-accent-soft); color: var(--cv-accent);
  font-size: .78rem; font-weight: 650; letter-spacing: .02em; animation: cvRise .7s ease both;
}
.cv-badge .pulse { width: 7px; height: 7px; border-radius: 50%; background: currentColor; animation: cvPulse 2.4s ease-out infinite; }
.cv-hero h1.cv-hero-title {
  font-size: clamp(2.5rem, 5.6vw, 4.3rem); line-height: 1.03; font-weight: 800; letter-spacing: -.04em;
  color: var(--cv-text); margin: 1.2rem auto 1.1rem; max-width: 15ch; padding: 0; animation: cvRise .8s .08s ease both;
}
.cv-grad-text { background: linear-gradient(92deg, var(--cv-grad-1), var(--cv-grad-2) 52%, var(--cv-grad-3)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.cv-hero-sub { font-size: 1.12rem; line-height: 1.6; color: var(--cv-text-2); max-width: 44rem; margin: 0 auto; animation: cvRise .8s .16s ease both; }

/* ───────────────────────── stat tiles & meters ───────────────────────── */
.cv-stats { display: grid; grid-template-columns: repeat(var(--cols, 4), minmax(0, 1fr)); gap: .8rem; animation: cvRise .8s .22s ease both; }
.cv-stat { padding: 1rem 1.15rem; border-radius: 16px; background: var(--cv-surface); border: 1px solid var(--cv-border); }
.cv-stat .v { font-size: 1.65rem; font-weight: 750; letter-spacing: -.025em; color: var(--cv-text); line-height: 1.15; }
.cv-stat .l { font-size: .82rem; color: var(--cv-text-2); margin-top: .15rem; }
.cv-tiles { display: grid; grid-template-columns: 1.25fr repeat(var(--cols, 3), minmax(0, 1fr)); gap: .8rem; margin: .2rem 0 .6rem; }
.cv-tile { position: relative; padding: 1rem 1.15rem 1.05rem; border-radius: 16px; background: var(--cv-surface); border: 1px solid var(--cv-border); overflow: hidden; }
.cv-tile.hero { background:
    radial-gradient(120% 140% at 100% 0%, color-mix(in srgb, var(--series, var(--cv-accent)) 16%, transparent), transparent 60%),
    var(--cv-surface); }
.cv-tile .lab { display: flex; align-items: center; gap: .45rem; font-size: .8rem; font-weight: 600; color: var(--cv-text-2); }
.cv-key { width: 10px; height: 10px; border-radius: 3px; background: var(--series, var(--cv-accent)); flex: none; }
.cv-tile .val { font-size: 1.9rem; font-weight: 750; letter-spacing: -.03em; color: var(--cv-text); margin: .3rem 0 .55rem; line-height: 1; }
.cv-tile.hero .val { font-size: 2.6rem; }
.cv-tile .val small { font-size: .55em; font-weight: 650; color: var(--cv-text-2); margin-left: .1rem; }
.cv-tile .sub { font-size: .78rem; color: var(--cv-text-3); margin-top: .5rem; }
.cv-meter { position: relative; height: 8px; border-radius: 999px; background: color-mix(in srgb, var(--series, var(--cv-accent)) 17%, transparent); overflow: hidden; }
.cv-meter > span { position: absolute; inset: 0 auto 0 0; border-radius: 999px; background: var(--series, var(--cv-accent)); transform-origin: left; animation: cvGrow .9s cubic-bezier(.2,.7,.2,1) both; }

/* ───────────────────────── meter list (benchmarks) ───────────────────────── */
.cv-mlist { display: grid; gap: .85rem; }
.cv-mrow { display: grid; grid-template-columns: minmax(8rem, 12rem) 1fr 3.6rem; align-items: center; gap: .9rem; }
.cv-mrow .name { font-size: .88rem; font-weight: 600; color: var(--cv-text); display: flex; gap: .45rem; align-items: center; }
.cv-mrow .num { font-size: .9rem; font-weight: 650; color: var(--cv-text); text-align: right; font-variant-numeric: tabular-nums; }
.cv-track { position: relative; height: 10px; border-radius: 999px; background: var(--cv-surface-sunken); border: 1px solid var(--cv-border); }
.cv-track > .fill { position: absolute; inset: 0 auto 0 0; border-radius: 999px; background: var(--series, var(--cv-accent)); animation: cvGrow .9s cubic-bezier(.2,.7,.2,1) both; transform-origin: left; }
.cv-track > .tick { position: absolute; top: -5px; bottom: -5px; width: 2px; border-radius: 2px; background: var(--cv-text-2); box-shadow: 0 0 0 2px var(--cv-surface-solid); }
.cv-track > .tick.p90 { background: var(--cv-text); }
.cv-legend { display: flex; flex-wrap: wrap; gap: 1.1rem; font-size: .78rem; color: var(--cv-text-2); margin-top: .9rem; }
.cv-legend span { display: inline-flex; align-items: center; gap: .4rem; }
.cv-legend i { display: inline-block; }
.cv-legend i.bar { width: 16px; height: 8px; border-radius: 999px; background: var(--series, var(--cv-accent)); }
.cv-legend i.tick { width: 2px; height: 12px; border-radius: 2px; background: var(--cv-text-2); }
.cv-legend i.tick.p90 { background: var(--cv-text); }

/* ───────────────────────── badges & chips ───────────────────────── */
.cv-verdict { display: inline-flex; align-items: center; gap: .4rem; padding: .3rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 650; white-space: nowrap; }
.cv-verdict .cv-ms { font-size: 16px; }
.cv-verdict.good { color: var(--cv-good-ink); background: color-mix(in srgb, var(--cv-good) 13%, transparent); }
.cv-verdict.warn { color: var(--cv-warn-ink); background: color-mix(in srgb, var(--cv-warn) 16%, transparent); }
.cv-verdict.crit { color: var(--cv-crit-ink); background: color-mix(in srgb, var(--cv-crit) 13%, transparent); }
.cv-pill { display: inline-flex; align-items: center; gap: .35rem; padding: .28rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 600; color: var(--cv-text-2); background: var(--cv-surface-sunken); border: 1px solid var(--cv-border); white-space: nowrap; }
.cv-chips { display: flex; flex-wrap: wrap; gap: .4rem; }
.cv-chip { display: inline-flex; align-items: center; gap: .35rem; padding: .3rem .62rem; border-radius: 9px; font-size: .82rem; font-weight: 550; color: var(--cv-text); background: var(--cv-surface-solid); border: 1px solid var(--cv-border); }
.cv-chip .cv-ms { font-size: 15px; flex: none; }
.cv-chip.hit .cv-ms { color: var(--cv-good-ink); }
.cv-chip.miss .cv-ms { color: var(--cv-crit-ink); }
.cv-chip.miss { background: transparent; border-style: dashed; color: var(--cv-text-2); }

/* ───────────────────────── candidate header ───────────────────────── */
.cv-cand { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin: .2rem 0 1rem; }
.cv-cand .who { display: flex; align-items: center; gap: .9rem; min-width: 0; }
.cv-avatar { width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center; font-weight: 750; font-size: 1.05rem; color: #fff; background: var(--cv-btn); box-shadow: 0 8px 20px -10px rgba(79,70,229,.9); flex: none; }
.cv-cand .nm { font-size: 1.2rem; font-weight: 750; letter-spacing: -.015em; color: var(--cv-text); overflow-wrap: anywhere; }
.cv-cand .meta { display: flex; gap: .45rem; flex-wrap: wrap; margin-top: .3rem; }

/* ───────────────────────── mode cards & pipeline ───────────────────────── */
.cv-mode { padding: .2rem .1rem .6rem; }
.cv-icon { width: 44px; height: 44px; border-radius: 13px; display: grid; place-items: center; margin-bottom: .9rem;
  color: var(--series, var(--cv-accent)); background: color-mix(in srgb, var(--series, var(--cv-accent)) 14%, transparent);
  border: 1px solid color-mix(in srgb, var(--series, var(--cv-accent)) 28%, transparent); }
.cv-icon .cv-ms { font-size: 24px; }
.cv-mode .tag { font-size: .74rem; font-weight: 650; letter-spacing: .04em; color: var(--cv-text-3); text-transform: uppercase; margin: .15rem 0 .55rem; }
.cv-list { list-style: none; padding: 0; margin: .8rem 0 0; display: grid; gap: .4rem; }
.cv-list li { display: flex; gap: .5rem; align-items: flex-start; font-size: .88rem; color: var(--cv-text-2); margin: 0; }
.cv-list li .cv-ms { font-size: 18px; color: var(--cv-good-ink); flex: none; }
.cv-steps { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .8rem; position: relative; }
.cv-step { position: relative; padding: 1.05rem 1rem 1.1rem; border-radius: 16px; background: var(--cv-surface); border: 1px solid var(--cv-border); transition: transform .2s ease, border-color .2s ease; }
.cv-step:hover { transform: translateY(-3px); border-color: var(--cv-accent-ring); }
.cv-step .n { font-size: .72rem; font-weight: 750; color: var(--cv-accent); letter-spacing: .08em; }
.cv-step .t { font-size: .95rem; font-weight: 700; color: var(--cv-text); margin: .35rem 0 .3rem; }
.cv-step .d { font-size: .82rem; line-height: 1.5; color: var(--cv-text-2); }
.cv-step .cv-icon { width: 36px; height: 36px; border-radius: 11px; margin-bottom: .7rem; }
.cv-step .cv-icon .cv-ms { font-size: 20px; }
.cv-formula { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: .8rem; line-height: 1.6; color: var(--cv-text); background: var(--cv-surface-sunken); border: 1px solid var(--cv-border); border-radius: 10px; padding: .55rem .75rem; margin-top: .6rem; white-space: pre-wrap; overflow-wrap: anywhere; }

/* ───────────────────────── tables ───────────────────────── */
.cv-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .88rem; }
.cv-table th { text-align: right; font-weight: 650; color: var(--cv-text-2); font-size: .78rem; letter-spacing: .02em; padding: .55rem .8rem; border-bottom: 1px solid var(--cv-border-strong); }
.cv-table th:first-child, .cv-table td:first-child { text-align: left; }
.cv-table td { text-align: right; padding: .5rem .8rem; color: var(--cv-text); border-bottom: 1px solid var(--cv-border); font-variant-numeric: tabular-nums; }
.cv-table tr:last-child td { border-bottom: none; }
.cv-table td.best { font-weight: 750; }
.cv-table td.best::after { content: " ●"; color: var(--cv-hyb); font-size: .7em; vertical-align: middle; }
.cv-table .th-key { display: inline-flex; align-items: center; gap: .4rem; }

.cv-note { display: flex; gap: .6rem; align-items: flex-start; padding: .75rem .9rem; border-radius: 12px; background: var(--cv-accent-soft); border: 1px solid var(--cv-accent-ring); color: var(--cv-text); font-size: .86rem; line-height: 1.5; }
.cv-note .cv-ms { font-size: 20px; color: var(--cv-accent); flex: none; }
.cv-footer { margin-top: 3.2rem; padding-top: 1.2rem; border-top: 1px solid var(--cv-border); display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; color: var(--cv-text-3); font-size: .8rem; }

/* ───────────────────────── motion ───────────────────────── */
@keyframes cvRise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes cvGrow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes cvPulse { 0% { box-shadow: 0 0 0 0 color-mix(in srgb, currentColor 55%, transparent); } 70%, 100% { box-shadow: 0 0 0 9px transparent; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }

/* ───────────────────────── responsive ───────────────────────── */
@media (max-width: 1000px) { .cv-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); } .cv-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 760px) { .cv-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } .cv-mrow { grid-template-columns: 1fr 3.2rem; } .cv-mrow .cv-track { grid-column: 1 / -1; grid-row: 2; } }
</style>
"""

# ── Icons ──────────────────────────────────────────────────────────────────
# st.html sanitises inline SVG away, so icons use the Material Symbols font that
# Streamlit already bundles (ligatures: the text "bolt" renders the bolt glyph).
_ICONS = {
    "bolt": "bolt", "sparkles": "auto_awesome", "file": "description", "broom": "cleaning_services",
    "hash": "tag", "layers": "layers", "target": "leaderboard", "check": "check", "x": "close",
    "alert": "warning", "info": "info", "chart": "monitoring",
}


def icon(name: str) -> str:
    return f'<span class="cv-ms" aria-hidden="true">{_ICONS.get(name, name)}</span>'


# ── Helpers ────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.html(CSS)


def html(markup: str) -> None:
    st.html(markup)


def series_var(kind: str) -> str:
    return {"keyword": "var(--cv-kw)", "semantic": "var(--cv-sem)", "hybrid": "var(--cv-hyb)"}.get(kind, "var(--cv-accent)")


def pct(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    html(
        f'<div class="cv-page-head"><div class="cv-eyebrow">{escape(eyebrow)}</div>'
        f"<h1>{escape(title)}</h1><p class='cv-muted'>{escape(subtitle)}</p></div>"
    )


def section(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eb = f'<div class="cv-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    sub = f'<p class="cv-muted">{escape(subtitle)}</p>' if subtitle else ""
    html(f'<div class="cv-section">{eb}<div class="cv-h2">{escape(title)}</div>{sub}</div>')


def card_title(title: str, subtitle: str = "", step: str = "") -> None:
    eb = f'<div class="cv-eyebrow">{escape(step)}</div>' if step else ""
    sub = f'<p class="cv-muted" style="font-size:.86rem">{escape(subtitle)}</p>' if subtitle else ""
    html(f'<div style="margin-bottom:.35rem">{eb}<div class="cv-h3">{escape(title)}</div>{sub}</div>')


def note(text: str, kind: str = "info") -> None:
    html(f'<div class="cv-note">{icon(kind)}<div>{text}</div></div>')


def stat_strip(stats: list[tuple[str, str]], cols: int = 4) -> None:
    items = "".join(f'<div class="cv-stat"><div class="v">{escape(v)}</div><div class="l">{escape(l)}</div></div>' for v, l in stats)
    html(f'<div class="cv-stats" style="--cols:{cols}">{items}</div>')


VERDICT_STYLE = {
    "Excellent match": ("good", "check"),
    "Strong match": ("good", "check"),
    "Partial match": ("warn", "alert"),
    "Weak match": ("crit", "x"),
}


def verdict_badge(verdict: str) -> str:
    cls, ic = VERDICT_STYLE.get(verdict, ("warn", "info"))
    return f'<span class="cv-verdict {cls}">{icon(ic)}{escape(verdict)}</span>'


def score_tiles(tiles: list[dict]) -> None:
    """tiles: dicts with label, value (0–1), series, sub, hero(bool)."""
    out = []
    for t in tiles:
        v = t["value"]
        width = 0 if v is None else max(0.0, min(1.0, v)) * 100
        value = "—" if v is None else f'{v * 100:.1f}<small>%</small>'
        cls = "cv-tile hero" if t.get("hero") else "cv-tile"
        out.append(
            f'<div class="{cls}" style="--series:{series_var(t.get("series", ""))}">'
            f'<div class="lab"><span class="cv-key"></span>{escape(t["label"])}</div>'
            f'<div class="val">{value}</div>'
            f'<div class="cv-meter" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{width:.0f}" '
            f'aria-label="{escape(t["label"])}"><span style="width:{width:.1f}%"></span></div>'
            f'<div class="sub">{escape(t.get("sub", ""))}</div></div>'
        )
    html(f'<div class="cv-tiles" style="--cols:{max(len(tiles) - 1, 1)}">{"".join(out)}</div>')


def meter_list(rows: list[dict], legend: bool = True) -> None:
    """rows: name, value (0–1), series, median (0–1|None), p90 (0–1|None)."""
    out = []
    for r in rows:
        ticks = ""
        if r.get("median") is not None:
            ticks += f'<div class="tick" style="left:calc({r["median"] * 100:.1f}% - 1px)" title="Pool median {r["median"] * 100:.0f}%"></div>'
        if r.get("p90") is not None:
            ticks += f'<div class="tick p90" style="left:calc({r["p90"] * 100:.1f}% - 1px)" title="Top-10% threshold {r["p90"] * 100:.0f}%"></div>'
        out.append(
            f'<div class="cv-mrow" style="--series:{series_var(r.get("series", ""))}">'
            f'<div class="name"><span class="cv-key"></span>{escape(r["name"])}</div>'
            f'<div class="cv-track"><div class="fill" style="width:{max(0, min(1, r["value"])) * 100:.1f}%"></div>{ticks}</div>'
            f'<div class="num">{r["value"] * 100:.0f}%</div></div>'
        )
    leg = ""
    if legend:
        leg = (
            '<div class="cv-legend"><span><i class="bar"></i>This candidate</span>'
            '<span><i class="tick"></i>Pool median</span><span><i class="tick p90"></i>Top-10% threshold</span></div>'
        )
    html(f'<div class="cv-mlist">{"".join(out)}</div>{leg}')


def chips(terms: list[str], kind: str = "hit", limit: int = 30) -> str:
    ic = icon("check") if kind == "hit" else icon("x")
    items = "".join(f'<span class="cv-chip {kind}">{ic}{escape(t)}</span>' for t in terms[:limit])
    return f'<div class="cv-chips">{items}</div>'


def candidate_header(name: str, verdict: str, pills: list[str]) -> None:
    initials = "".join(part[0] for part in name.replace("_", " ").replace("-", " ").split()[:2]).upper() or "CV"
    pill_html = "".join(f'<span class="cv-pill">{escape(p)}</span>' for p in pills)
    html(
        f'<div class="cv-cand"><div class="who"><div class="cv-avatar">{escape(initials[:2])}</div>'
        f'<div><div class="nm">{escape(name)}</div><div class="meta">{pill_html}</div></div></div>'
        f"<div>{verdict_badge(verdict)}</div></div>"
    )


def footer(meta: dict | None) -> None:
    model = escape(meta.get("bert_model", "all-MiniLM-L6-v2").split("/")[-1]) if meta else "all-MiniLM-L6-v2"
    html(
        '<div class="cv-footer"><span>CV-Scan · NLP resume screening — TF-IDF · Sentence-BERT · hybrid ranking</span>'
        f"<span>Models: scikit-learn TF-IDF, {model} · Theme: ⋮ menu → System / Light / Dark</span></div>"
    )
