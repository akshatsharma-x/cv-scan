"""
app.py — CV-Scan · AI resume screener (Streamlit dashboard).

Launch from the repository root (so .streamlit/config.toml is picked up):
    streamlit run Resume_Screening_Project/src/app.py

Pages
  • Home              — overview, live stats and how the pipeline works
  • Screen resumes    — job description + resumes → ranked, explained results
  • Model evaluation  — real, reproducible metrics for TF-IDF / BERT / Hybrid
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import charts  # noqa: E402
import ui  # noqa: E402
from engine import DEFAULT_KEYWORD_WEIGHT, MODEL_DIR, TOP_JD_TERMS, ScreeningEngine  # noqa: E402
from sample_data import SAMPLE_JOBS  # noqa: E402
from utils import ExtractionError, extract_text, word_count  # noqa: E402

st.set_page_config(
    page_title="CV-Scan · AI Resume Screener",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

REQUIRED_ARTIFACTS = (
    "metadata.json",
    "processed_resume_dataset.pkl",
    "dataset_bert_embeddings.pkl",
    "tfidf_vectorizer.pkl",
    "category_classifier.pkl",
)
EVAL_PATH = os.path.join(MODEL_DIR, "evaluation_results.json")
MODE_LABELS = {"basic": "Basic · TF-IDF keywords", "advanced": "Advanced · Hybrid (TF-IDF + BERT)"}
MODE_SHORT = {"basic": ":material/bolt: Basic", "advanced": ":material/auto_awesome: Advanced"}
LOWER_IS_BETTER = {"FPR", "FNR", "FDR"}


# ── Data access (cached) ───────────────────────────────────────────────────
def artifacts_ready() -> bool:
    return all(os.path.exists(os.path.join(MODEL_DIR, name)) for name in REQUIRED_ARTIFACTS)


@st.cache_resource(show_spinner="Loading the screening engine …")
def get_engine() -> ScreeningEngine:
    return ScreeningEngine(MODEL_DIR)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_json(path: str) -> dict | None:
    return _read_json(path, os.path.getmtime(path)) if os.path.exists(path) else None


def metadata() -> dict | None:
    return load_json(os.path.join(MODEL_DIR, "metadata.json"))


def evaluation() -> dict | None:
    return load_json(EVAL_PATH)


@st.cache_data(show_spinner=False)
def demo_resumes() -> pd.DataFrame:
    path = os.path.join(MODEL_DIR, "demo_resumes.pkl")
    return pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame(columns=["ID", "Category", "text"])


def is_dark() -> bool:
    return st.context.theme.type != "light"


# ── Session state ──────────────────────────────────────────────────────────
for key, value in {"mode": "advanced", "kw_weight": int(DEFAULT_KEYWORD_WEIGHT * 100), "jd_text": "",
                   "samples": [], "uploads": {}, "run": None, "run_errors": [], "run_signature": None,
                   "sample_seed": 0, "uploader_nonce": 0, "uploader_names": set()}.items():
    st.session_state.setdefault(key, value)


def _load_sample_job() -> None:
    title = st.session_state.get("sample_job")
    if title:
        st.session_state.jd_text = SAMPLE_JOBS[title]["text"]


def _uploader_key() -> str:
    return f"uploader_{st.session_state.uploader_nonce}"


def _sync_uploads() -> None:
    """Mirror the uploader into session state so files survive page switches."""
    files = st.session_state.get(_uploader_key()) or []
    current = {f.name: f.getvalue() for f in files}
    for name in st.session_state.uploader_names - set(current):  # removed in the widget
        st.session_state.uploads.pop(name, None)
    st.session_state.uploads.update(current)
    st.session_state.uploader_names = set(current)


def _remove_upload(name: str) -> None:
    st.session_state.uploads.pop(name, None)


def _remove_sample(sample_id) -> None:
    st.session_state.samples = [s for s in st.session_state.samples if s["id"] != sample_id]


def _add_samples(n: int = 6) -> None:
    pool = demo_resumes()
    if pool.empty:
        return
    rng = np.random.default_rng(1000 + st.session_state.sample_seed)
    st.session_state.sample_seed += 1
    taken = {s["id"] for s in st.session_state.samples}
    pool = pool[~pool["ID"].isin(taken)]
    target = SAMPLE_JOBS.get(st.session_state.get("sample_job") or "", {}).get("category")
    picks = []
    if target:  # a realistic batch: a few on-target resumes mixed with others
        same = pool[pool["Category"] == target]
        picks += same.sample(n=min(3, len(same)), random_state=int(rng.integers(1e9))).index.tolist()
    rest = pool[~pool.index.isin(picks)]
    if target:  # fillers come from other categories, so exactly three on-target resumes are added
        rest = rest[rest["Category"] != target]
    others = rest.groupby("Category").sample(n=1, random_state=int(rng.integers(1e9)))
    picks += others.sample(n=min(n - len(picks), len(others)), random_state=int(rng.integers(1e9))).index.tolist()
    for _, row in pool.loc[picks].iterrows():
        st.session_state.samples.append({
            "id": row["ID"], "category": row["Category"], "text": row["text"],
            "name": f"{charts._short(row['Category'])} resume #{row['ID']}",
        })


def _clear_candidates() -> None:
    st.session_state.samples = []
    st.session_state.uploads = {}
    st.session_state.uploader_names = set()
    st.session_state.uploader_nonce += 1  # a fresh (empty) uploader widget


def _signature(jd: str, names: list[str], mode: str, weight: int) -> str:
    raw = json.dumps([jd.strip(), sorted(names), mode, weight if mode == "advanced" else None])
    return hashlib.sha1(raw.encode()).hexdigest()


def _current_candidates() -> list[str]:
    return list(st.session_state.uploads) + [s["name"] for s in st.session_state.samples]


# ══════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════
def home_page() -> None:
    meta = metadata() or {}
    ev = evaluation()
    n_pool = meta.get("n_resumes", 0)

    ui.html(
        '<section class="cv-hero">'
        '<span class="cv-badge"><span class="pulse"></span>NLP · Hybrid retrieval · Explainable AI</span>'
        '<h1 class="cv-hero-title">Screen resumes with <span class="cv-grad-text">intelligent precision</span></h1>'
        '<p class="cv-hero-sub">CV-Scan ranks candidates against any job description by fusing TF-IDF keyword '
        "matching with Sentence-BERT semantic understanding — and shows exactly which skills drive every score.</p>"
        "</section>"
    )
    _, b1, b2, _ = st.columns([1, 1, 1, 1], gap="small")
    with b1:
        if st.button("Start screening", type="primary", key="cta_start", icon=":material/arrow_forward:", width="stretch"):
            st.switch_page(SCREEN_PAGE)
    with b2:
        if st.button("Model evaluation", key="cta_eval", icon=":material/monitoring:", width="stretch"):
            st.switch_page(EVAL_PAGE)

    st.space("medium")
    stats = [(f"{n_pool:,}", "historical resumes indexed"), (str(meta.get("n_categories", 24)), "professional categories"),
             (f"{meta.get('model_dim', 384)}-D × 2", "Sentence-BERT views per resume")]
    if ev:
        stats.append((f"{ev['ranking']['summary']['Hybrid']['ROC-AUC']:.2f}", "hybrid ranking ROC-AUC (held-out)"))
    ui.stat_strip(stats, cols=len(stats))

    ui.section("Two ways to scan", "Pick the depth of analysis — you can switch at any time in the workspace.", eyebrow="Scan modes")
    col_a, col_b = st.columns(2, gap="medium")
    modes = {
        "basic": (col_a, "keyword", "bolt", "Basic scan", "TF-IDF · keyword match",
                  "Fast, fully transparent lexical screening. Ideal for shortlisting against hard requirements.",
                  ["Matched and missing JD keywords", "Each term's share of the score", "Rank vs. the historical pool"]),
        "advanced": (col_b, "hybrid", "sparkles", "Advanced scan", "TF-IDF + Sentence-BERT · hybrid",
                     "Adds deep semantic understanding, so synonyms and related experience count too.",
                     ["Hybrid score = 30% keyword + 70% semantic", "Semantic map of the applicant pool", "Predicted professional domain"]),
    }
    for mode, (col, series, ic, title, tag, desc, bullets) in modes.items():
        with col, st.container(key=f"card_mode_{mode}"):
            items = "".join(f"<li>{ui.icon('check')}<span>{b}</span></li>" for b in bullets)
            ui.html(
                f'<div class="cv-mode" style="--series:{ui.series_var(series)}"><div class="cv-icon">{ui.icon(ic)}</div>'
                f'<div class="cv-h3">{title}</div><div class="tag">{tag}</div><p class="cv-muted">{desc}</p>'
                f'<ul class="cv-list">{items}</ul></div>'
            )
            if st.button(f"Start {title.lower()}", key=f"go_{mode}", width="stretch",
                         type="primary" if mode == "advanced" else "secondary"):
                st.session_state.mode = mode
                st.switch_page(SCREEN_PAGE)

    ui.section("How it works", "Every score is reproducible and explainable — no black boxes.", eyebrow="Pipeline")
    steps = [
        ("file", "keyword", "Parse", "Extract text from PDF, DOCX (incl. tables) or TXT files."),
        ("broom", "keyword", "Clean", "Remove URLs, e-mails & noise, drop stop-words, lemmatise with NLTK."),
        ("hash", "keyword", "Keyword engine", "TF-IDF vectors + cosine similarity; each term's contribution is traced."),
        ("layers", "semantic", "Semantic engine", "Sentence-BERT embeds the whole resume (150-word chunks) plus its header."),
        ("target", "hybrid", "Hybrid & rank", f"Scores calibrated against {n_pool:,} resumes for the same job, then fused."),
    ]
    cards = "".join(
        f'<div class="cv-step" style="--series:{ui.series_var(s)}"><div class="cv-icon">{ui.icon(i)}</div>'
        f'<div class="n">STEP {k}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for k, (i, s, t, d) in enumerate(steps, start=1)
    )
    ui.html(f'<div class="cv-steps">{cards}</div>')
    st.space("small")
    f1, f2, f3 = st.columns(3, gap="medium")
    formulas = [
        (f1, "TF-IDF weight", "w(t,d) = (1 + ln tf) · idf(t)\nidf(t) = ln((1+N)/(1+df)) + 1", "Sub-linear term frequency × smoothed inverse document frequency, L2-normalised."),
        (f2, "Cosine similarity", "cos(J, R) = J · R / (‖J‖ ‖R‖)", "Used by both engines: job vector J vs. resume vector R."),
        (f3, "Hybrid score", "H = w · K̂ + (1 − w) · Ŝ,  w = 0.30", "K̂, Ŝ are min-max calibrated against the pool for the same job."),
    ]
    for col, title, formula, desc in formulas:
        with col:
            ui.html(f'<div class="cv-card"><div class="cv-h3">{title}</div><div class="cv-formula">{formula}</div>'
                    f'<p class="cv-small" style="margin:.55rem 0 0">{desc}</p></div>')
    ui.footer(meta)


# ══════════════════════════════════════════════════════════════════════════
# SCREEN
# ══════════════════════════════════════════════════════════════════════════
def screen_page() -> None:
    engine = get_engine()
    meta = engine.meta
    ui.page_header("Screening workspace", "Rank candidates for a job",
                   "Paste a job description, add one or more resumes and run the scan. Results are explained term by term.")

    left, right = st.columns([1.12, 1], gap="medium")
    with left, st.container(key="card_jd"):
        ui.card_title("Job description", "Paste the full posting — duties, skills and qualifications.", step="Step 1")
        st.selectbox("Sample job", list(SAMPLE_JOBS), index=None, key="sample_job", on_change=_load_sample_job,
                     placeholder="Or load one of 24 sample job descriptions…", label_visibility="collapsed",
                     persist_state="session")
        st.text_area("Job description", key="jd_text", height=268, label_visibility="collapsed",
                     placeholder="e.g. We are hiring an HR Generalist to manage recruiting, onboarding, payroll …",
                     persist_state="session")
        words = word_count(st.session_state.jd_text)
        st.caption(f"{words} words" + (" · add more detail for a reliable match" if 0 < words < 25 else ""))

    with right, st.container(key="card_resumes"):
        ui.card_title("Candidate resumes", "PDF, DOCX or TXT — add several files to rank them against each other.", step="Step 2")
        if _uploader_key() not in st.session_state:  # widget is fresh (first visit or after a page switch)
            st.session_state.uploader_names = set()
        st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True, key=_uploader_key(),
                         on_change=_sync_uploads, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        c1.button("Add sample resumes", key="add_samples", on_click=_add_samples, icon=":material/library_add:",
                  width="stretch", help="Adds 6 held-out resumes from the dataset that no model has seen. "
                                        "If a sample job is loaded, three of them match its category.")
        c2.button("Clear all", key="clear_candidates", on_click=_clear_candidates, icon=":material/delete_sweep:",
                  width="stretch", disabled=not (st.session_state.samples or st.session_state.uploads))
        names = _current_candidates()
        if names:
            in_widget = st.session_state.uploader_names
            st.caption(f"**{len(names)} candidate{'s' if len(names) != 1 else ''} in this batch**"
                       + (" · uploaded files are listed in the box above" if in_widget else ""))
            extra = [n for n in st.session_state.uploads if n not in in_widget]
            rows = len(extra) + len(st.session_state.samples)
            if rows:
                with st.container(height=min(40 * rows + 8, 200), border=False):
                    for i, name in enumerate(extra):
                        a, b = st.columns([8, 1], vertical_alignment="center")
                        a.markdown(f":material/description: {name} :gray[· kept from earlier]")
                        b.button(":material/close:", key=f"rm_up_{i}_{name}", on_click=_remove_upload, args=(name,),
                                 type="tertiary", help="Remove from batch")
                    for s in st.session_state.samples:
                        a, b = st.columns([8, 1], vertical_alignment="center")
                        a.markdown(f":material/science: {s['name']}")
                        b.button(":material/close:", key=f"rm_s_{s['id']}", on_click=_remove_sample, args=(s["id"],),
                                 type="tertiary", help="Remove from batch")
        else:
            st.caption("No resumes yet — upload files or add sample resumes.")

    with st.container(key="card_settings"):
        c_mode, c_weight, c_run = st.columns([1.25, 1.5, 0.95], gap="medium", vertical_alignment="bottom")
        with c_mode:
            st.segmented_control("Scan mode", options=list(MODE_SHORT), format_func=MODE_SHORT.get, key="mode",
                                 required=True, persist_state="session", width="stretch",
                                 help="Basic: TF-IDF keyword matching only. Advanced: hybrid of TF-IDF and Sentence-BERT "
                                      "with semantic map and domain prediction.")
        with c_weight:
            if st.session_state.mode == "advanced":
                st.slider("Keyword weight in the hybrid score", 0, 100, step=5, key="kw_weight", format="%d%%",
                          persist_state="session",
                          help="Hybrid = w × keyword + (1 − w) × semantic. 30% is the default and the best "
                               "setting in the ranking benchmark (see Model evaluation).")
            else:
                st.caption("Basic mode scores keyword overlap only (TF-IDF). Switch to Advanced to add "
                           "semantic matching and domain prediction.")
        with c_run:
            run_clicked = st.button("Run scan", type="primary", key="cta_run", icon=":material/rocket_launch:", width="stretch")

    if run_clicked:
        _run_scan(engine)

    run = st.session_state.run
    if run is None:
        ui.html(
            '<div class="cv-card" style="margin-top:1.2rem;text-align:center;padding:2.2rem 1rem">'
            f'<div class="cv-icon" style="margin:0 auto .8rem;--series:var(--cv-accent)">{ui.icon("sparkles")}</div>'
            '<div class="cv-h3">Your ranked shortlist will appear here</div>'
            '<p class="cv-muted" style="max-width:36rem;margin:.35rem auto 0">Tip: load a sample job and add sample '
            "resumes to try CV-Scan in seconds — three of the samples match the job's category, so you can check "
            "that they rise to the top.</p></div>"
        )
    else:
        render_results(engine, run, meta)
    ui.footer(meta)


def _run_scan(engine: ScreeningEngine) -> None:
    jd = (st.session_state.jd_text or "").strip()
    candidates, errors, seen = [], [], set()
    for name, data in st.session_state.uploads.items():
        try:
            candidates.append((name, extract_text(name, data)))
            seen.add(name)
        except ExtractionError as exc:
            errors.append((name, str(exc)))
    for s in st.session_state.samples:
        if s["name"] not in seen:
            candidates.append((s["name"], s["text"]))
            seen.add(s["name"])

    if not jd:
        st.warning("Add a job description first (Step 1).", icon=":material/edit_note:")
        return
    if not candidates:
        for name, message in errors:
            st.error(f"**{name}** — {message}", icon=":material/error:")
        st.warning("Add at least one readable resume (Step 2).", icon=":material/upload_file:")
        return

    mode = st.session_state.mode
    weight = st.session_state.kw_weight
    label = "Running keyword and semantic engines …" if mode == "advanced" else "Running keyword engine …"
    try:
        with st.spinner(f"{label} ({len(candidates)} resume{'s' if len(candidates) > 1 else ''})", show_time=True):
            run = engine.screen(jd, candidates, mode=mode, keyword_weight=weight / 100)
    except ValueError as exc:
        st.warning(str(exc), icon=":material/warning:")
        return
    st.session_state.run = run
    st.session_state.run_errors = errors
    st.session_state.run_signature = _signature(jd, [n for n, _ in candidates] + [n for n, _ in errors], mode, weight)
    st.session_state.selected = run.candidates[0].name


def render_results(engine: ScreeningEngine, run, meta: dict) -> None:
    dark = is_dark()
    adv = run.is_advanced
    n = len(run.candidates)
    n_pool = len(run.pool_score)
    series = "hybrid" if adv else "keyword"
    current = _signature(st.session_state.jd_text or "", _current_candidates(), st.session_state.mode, st.session_state.kw_weight)

    ui.section("Results", f"{n} candidate{'s' if n != 1 else ''} ranked · {MODE_LABELS[run.mode]}"
               + (f" · keyword weight {run.keyword_weight:.0%}" if adv else ""), eyebrow="Step 3")
    if current != st.session_state.run_signature:
        ui.note("The inputs changed since this scan — press <b>Run scan</b> to refresh the results.")
        st.space("small")
    for name, message in st.session_state.run_errors:
        st.error(f"**{name}** was skipped — {message}", icon=":material/error:")

    best = run.candidates[0]
    strong = sum(c.score >= 0.5 for c in run.candidates)
    stats = [(str(n), "candidates screened"), (f"{best.score:.0%}", f"best match · {best.name[:28]}"),
             (f"{strong}", "strong or excellent matches")]
    if adv and run.jd_categories:
        stats.append((charts._short(run.jd_categories[0][0]), "predicted job domain"))
    else:
        stats.append((f"{np.median([c.coverage for c in run.candidates]):.0%}", "median JD-term coverage"))
    ui.stat_strip(stats, cols=4)
    st.space("small")

    # Leaderboard -----------------------------------------------------------
    rows = []
    for c in run.candidates:
        row = {"Rank": c.rank, "Candidate": c.name, "Verdict": c.verdict, "Score": c.score * 100}
        if adv:
            row |= {"Keyword": c.keyword_score * 100, "Semantic": c.semantic_score * 100}
        row |= {"Coverage": c.coverage * 100, "Top": max(100 - c.percentile, 0.1)}
        if adv:
            row["Domain"] = charts._short(c.categories[0][0]) if c.categories else "—"
        rows.append(row)
    board = pd.DataFrame(rows)
    m = "dark" if dark else "light"
    colcfg = {
        "Rank": st.column_config.NumberColumn("#", width=44),
        "Candidate": st.column_config.TextColumn("Candidate", width=220),
        "Verdict": st.column_config.TextColumn("Verdict", width=118),
        "Score": st.column_config.ProgressColumn("Hybrid score" if adv else "Keyword score", min_value=0, max_value=100,
                                                 format="%.0f%%", color=ui.SERIES[series][m], width=128),
        "Keyword": st.column_config.ProgressColumn("Keyword", min_value=0, max_value=100, format="%.0f%%",
                                                   color=ui.SERIES["keyword"][m], width=118),
        "Semantic": st.column_config.ProgressColumn("Semantic", min_value=0, max_value=100, format="%.0f%%",
                                                    color=ui.SERIES["semantic"][m], width=118),
        "Coverage": st.column_config.NumberColumn("JD terms", format="%.0f%%", width=82,
                                                  help=f"Share of the job's {TOP_JD_TERMS} key terms found in the resume"),
        "Top": st.column_config.NumberColumn("Pool rank", format="top %.1f%%", width=94,
                                             help=f"Rank among {n_pool:,} historical resumes scored for this job"),
        "Domain": st.column_config.TextColumn("Predicted domain", width=136),
    }
    with st.container(key="card_board"):
        h1, h2 = st.columns([3, 1], vertical_alignment="center")
        with h1:
            ui.card_title("Leaderboard", "Sorted by score. Pool rank compares each candidate with every historical resume for this job.")
        with h2:
            st.download_button("Export CSV", board.to_csv(index=False).encode(), file_name="cv-scan-results.csv",
                               mime="text/csv", icon=":material/download:", on_click="ignore", width="stretch")
        st.dataframe(board, hide_index=True, width="stretch", column_config=colcfg,
                     height=min(35 * (len(board) + 1) + 3, 423))

    with st.expander("What the job description asks for", icon=":material/fact_check:"):
        ui.html(f'<p class="cv-small" style="margin:0 0 .5rem">Top {len(run.jd_terms)} terms by TF-IDF weight — '
                "these drive the keyword score and the coverage metric.</p>" + ui.chips([t for t, _ in run.jd_terms], "hit"))
        if adv and run.jd_categories:
            ui.html('<div class="cv-h3" style="margin:1rem 0 .6rem">Predicted job domain</div>')
            ui.meter_list([{"name": charts._short(cat), "value": p, "series": "hybrid"} for cat, p in run.jd_categories], legend=False)

    # Deep dive -------------------------------------------------------------
    ui.section("Candidate deep-dive", "Why a candidate scored the way they did.", eyebrow="Explainable AI")
    names = [c.name for c in run.candidates]
    if st.session_state.get("selected") not in names:
        st.session_state.selected = names[0]
    st.selectbox("Candidate", names, key="selected", format_func=lambda x: f"#{names.index(x) + 1} · {x}",
                 label_visibility="collapsed")
    idx = names.index(st.session_state.selected)
    cand = run.candidates[idx]

    pills = [f"Rank {cand.rank} of {n}", f"Top {max(100 - cand.percentile, 0.1):.1f}% of {n_pool:,} resumes",
             f"{cand.word_count:,} words"]
    if adv and cand.categories:
        pills.append(f"Domain: {charts._short(cand.categories[0][0])}")
    ui.candidate_header(cand.name, cand.verdict, pills)

    tiles = []
    if adv:
        tiles.append({"label": "Hybrid score", "value": cand.hybrid_score, "series": "hybrid", "hero": True,
                      "sub": f"{run.keyword_weight:.0%} keyword · {1 - run.keyword_weight:.0%} semantic"})
    tiles.append({"label": "Keyword match", "value": cand.keyword_score, "series": "keyword", "hero": not adv,
                  "sub": f"TF-IDF cosine {cand.keyword_raw:.3f}, calibrated to the pool"})
    if adv:
        tiles.append({"label": "Semantic match", "value": cand.semantic_score, "series": "semantic",
                      "sub": f"BERT cosine {cand.semantic_raw:.3f}, calibrated to the pool"})
    tiles.append({"label": "JD term coverage", "value": cand.coverage, "series": "",
                  "sub": f"{round(cand.coverage * len(run.jd_terms))} of {len(run.jd_terms)} key terms present"})
    ui.score_tiles(tiles)
    st.space("small")

    tab_names = ["Keyword insights", "Pool benchmark"] + (["Semantic map", "Domain fit"] if adv else []) + ["Resume text"]
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    with tabs["Keyword insights"]:
        c1, c2 = st.columns([1.25, 1], gap="large")
        with c1:
            other = f" Word pairs that are not real phrases add another {cand.other_share:.0%}." if cand.other_share >= 0.005 else ""
            ui.card_title("What drives the keyword score", "Each bar is the term's share of the TF-IDF cosine similarity." + other)
            if cand.matched:
                st.plotly_chart(charts.matched_terms(cand.matched, dark), config=charts.PLOT_CONFIG, key="matched_chart")
            else:
                st.info("No overlapping terms between this resume and the job description.")
        with c2:
            jd_terms = [t for t, _ in run.jd_terms]
            missing = [t for t, _ in cand.missing]
            present = [t for t in jd_terms if t not in set(missing)]
            ui.card_title(f"Key requirements found · {len(present)}/{len(jd_terms)}")
            ui.html(ui.chips(present, "hit") if present else '<p class="cv-small">None of the key terms appear.</p>')
            st.space("small")
            ui.card_title(f"Missing from the resume · {len(missing)}", "Ordered by importance in the job description.")
            ui.html(ui.chips(missing, "miss") if missing else '<p class="cv-small">Every key term is covered.</p>')

    with tabs["Pool benchmark"]:
        def stats_of(arr):
            return float(np.median(arr)), float(np.quantile(arr, 0.9))

        rows = []
        if adv:
            med, p90 = stats_of(run.pool_score)
            rows.append({"name": "Hybrid score", "value": cand.hybrid_score, "series": "hybrid", "median": med, "p90": p90})
        med, p90 = stats_of(run.pool_keyword)
        rows.append({"name": "Keyword match", "value": cand.keyword_score, "series": "keyword", "median": med, "p90": p90})
        if adv:
            med, p90 = stats_of(run.pool_semantic)
            rows.append({"name": "Semantic match", "value": cand.semantic_score, "series": "semantic", "median": med, "p90": p90})
        med, p90 = stats_of(run.pool_coverage)
        rows.append({"name": "JD term coverage", "value": cand.coverage, "series": "", "median": med, "p90": p90})
        c1, c2 = st.columns([1, 1.35], gap="large")
        with c1:
            ui.card_title("Candidate vs. historical pool", f"Ticks mark the median and the top-10% threshold of {n_pool:,} resumes scored for this job.")
            ui.meter_list(rows)
            st.space("small")
            ui.card_title("Closest historical profiles", "The pool's best matches for this job — what a strong fit looks like.")
            top = run.pool_top.copy()
            titles = engine.pool.set_index("ID")["title"]
            top.insert(1, "Title", top["ID"].map(titles).fillna(""))
            top["Category"] = top["Category"].map(charts._short)
            top["Score"] = top["Score"] * 100
            st.dataframe(top[["Title", "Category", "Score"]].head(6), hide_index=True, width="stretch",
                         column_config={"Title": st.column_config.TextColumn("Title", width="medium"),
                                        "Category": st.column_config.TextColumn("Category", width="small"),
                                        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f%%",
                                                                                 color=ui.SERIES[series][m], width="small")})
        with c2:
            ui.card_title("Score distribution", f"Where {cand.name} lands among all historical resumes for this job.")
            st.plotly_chart(charts.score_distribution(run.pool_score, [(c.name, c.score) for c in run.candidates], cand.name, series, dark),
                            config=charts.PLOT_CONFIG, key="dist_chart")

    if adv:
        with tabs["Semantic map"]:
            h1, h2 = st.columns([3, 1], vertical_alignment="bottom")
            with h1:
                ui.card_title("Semantic space", "PCA projection of the Sentence-BERT document vectors. Historical resumes are "
                              "coloured by how well they match this job; ◆ marks candidates, ★ the job description.")
            with h2:
                dims = st.segmented_control("Projection", ["2D", "3D"], default="2D", key="map_dims", required=True,
                                            label_visibility="collapsed", width="stretch")
            st.plotly_chart(charts.semantic_map(engine, run, idx, dark, dims=3 if dims == "3D" else 2),
                            config=charts.PLOT_CONFIG if dims == "2D" else {"displaylogo": False}, key="map_chart")

        with tabs["Domain fit"]:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                ui.card_title("Candidate's predicted domain", "Calibrated linear SVM on TF-IDF ⊕ BERT features (24 categories).")
                ui.meter_list([{"name": charts._short(cat), "value": p, "series": "hybrid"} for cat, p in cand.categories], legend=False)
            with c2:
                ui.card_title("Job's predicted domain", "The same classifier applied to the job description.")
                ui.meter_list([{"name": charts._short(cat), "value": p, "series": "hybrid"} for cat, p in run.jd_categories], legend=False)
            if cand.categories and run.jd_categories:
                same = cand.categories[0][0] == run.jd_categories[0][0]
                overlap = {c for c, _ in cand.categories} & {c for c, _ in run.jd_categories}
                st.space("small")
                if same:
                    ui.note(f"Domain alignment: both are most likely <b>{charts._short(cand.categories[0][0])}</b>.", "check")
                elif overlap:
                    ui.note("Partial domain alignment: the job's and the candidate's top-3 domains overlap ("
                            + ", ".join(charts._short(c) for c in overlap) + ").", "info")
                else:
                    ui.note("Different domains: this candidate's background differs from the job's field — "
                            "check transferable experience manually.", "alert")

    with tabs["Resume text"]:
        st.caption(f"Extracted text · {cand.word_count:,} words. This is exactly what the engines analysed.")
        st.code(cand.text, language=None, wrap_lines=True, height=420)


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════
METRIC_DEFINITIONS = """
**Confusion matrix (one-vs-rest, per category)**

| | Predicted negative | Predicted positive |
|---|---|---|
| **Actual negative** | True Negative (TN) | False Positive (FP) |
| **Actual positive** | False Negative (FN) | True Positive (TP) |

- **TP** — positive instances correctly classified as positive. **TN** — negative instances correctly classified as negative.
- **FP** — *negative* instances incorrectly classified as positive. **FN** — *positive* instances incorrectly classified as negative.

| Measure | Formula | Meaning |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Share of correct predictions (multi-class: correct / total) |
| Precision | TP / (TP + FP) | Exactness — how many predicted positives are right |
| Recall / Sensitivity | TP / (TP + FN) = TP / P | Completeness — true positive rate |
| Specificity | TN / (TN + FP) = TN / N | True negative rate |
| F1-score | 2 · Precision · Recall / (Precision + Recall) | Harmonic mean of precision and recall |
| FPR | FP / (FP + TN) = FP / N | Negatives misclassified as positive |
| FNR | FN / (FN + TP) = FN / P | Positives misclassified as negative |
| NPV | TN / (TN + FN) | Share of predicted negatives that are truly negative |
| FDR | FP / (FP + TP) | Share of predicted positives that are wrong |
| MCC | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Balanced correlation in [−1, 1]; multi-class generalisation used |

Per-class measures are macro-averaged over the 24 categories; accuracy and MCC are computed on the full multi-class confusion matrix.
"""


def _metric_table(values: dict[str, dict[str, float]], metrics: list[str]) -> str:
    methods = list(values)
    head = "".join(f'<th><span class="th-key"><span class="cv-key" style="--series:{ui.series_var(charts.METHOD_SERIES[m])}"></span>{m}</span></th>' for m in methods)
    body = []
    for metric in metrics:
        vals = [values[m][metric] for m in methods]
        best = min(vals) if metric in LOWER_IS_BETTER else max(vals)
        cells = "".join(f'<td class="{"best" if np.isclose(v, best) else ""}">{v:.3f}</td>' for v in vals)
        arrow = " ↓" if metric in LOWER_IS_BETTER else ""
        body.append(f"<tr><td>{metric}{arrow}</td>{cells}</tr>")
    return f'<table class="cv-table"><thead><tr><th>Measure</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _run_evaluation_now() -> None:
    from evaluation import run_full_evaluation

    engine = get_engine()
    with st.spinner("Running the evaluation suite (≈2–3 minutes) …", show_time=True):
        results = run_full_evaluation(engine, demo_resumes(), SAMPLE_JOBS, log=lambda *_: None)
        results["created_at"] = engine.meta.get("created_at")
        with open(EVAL_PATH, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    _read_json.clear()
    st.rerun()


def evaluation_page() -> None:
    dark = is_dark()
    meta = metadata() or {}
    ev = evaluation()
    ui.page_header("Experimental evaluation", "How well do TF-IDF, BERT and the hybrid work?",
                   "Every number on this page is computed from the project's own pipeline and data — reproduce it with "
                   "python Resume_Screening_Project/src/train_models.py.")
    if ev is None:
        ui.note("No evaluation results found yet. Run the suite once (it trains and tests the classifiers and "
                "benchmarks the ranking engine) or run <code>python Resume_Screening_Project/src/train_models.py</code>.", "info")
        st.space("small")
        if st.button("Run evaluation now", type="primary", key="cta_eval_run", icon=":material/play_arrow:"):
            _run_evaluation_now()
        ui.footer(meta)
        return

    cls = ev["classification"]
    per_split = cls["per_split"]
    split80 = {m: per_split[m]["80%"] for m in per_split}
    rank = ev["ranking"]
    hyb = split80["Hybrid"]
    ui.stat_strip([
        (f"{hyb['Accuracy']:.1%}", "hybrid accuracy · 24-way classification"),
        (f"{hyb['F1-Score']:.3f}", "hybrid macro F1-score"),
        (f"{hyb['MCC']:.3f}", "hybrid MCC"),
        (f"{rank['summary']['Hybrid']['ROC-AUC']:.3f}", "hybrid ranking ROC-AUC"),
        (f"{rank['summary']['Hybrid']['nDCG@10']:.3f}", "hybrid ranking nDCG@10"),
    ], cols=5)

    # A. Classification --------------------------------------------------------
    ui.section("A · Resume category classification",
               f"Linear SVMs predict a resume's category from TF-IDF, BERT or concatenated hybrid features. "
               f"{ev['n_samples']:,} resumes, {ev['n_classes']} categories, stratified splits; the TF-IDF vocabulary is "
               "fitted on the training portion only.", eyebrow="Experiment A")
    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1, st.container(key="card_table3"):
        ui.card_title("Table 3 · Overall performance (80 % training split)", "● marks the best method per measure; ↓ lower is better.")
        ui.html(_metric_table(split80, [m for m in charts_metric_order()]))
    with c2, st.container(key="card_bars"):
        ui.card_title("Key measures by method", "Hybrid values are labelled; hover for the others.")
        st.plotly_chart(charts.grouped_bars(split80, ["Accuracy", "Precision", "Sensitivity", "F1-Score", "MCC"], dark),
                        config=charts.PLOT_CONFIG, key="eval_bars")

    with st.container(key="card_curve"):
        ui.card_title("Performance across training splits", "Figures 9–11: each measure for 50–90 % training data.")
        metric = st.pills("Measure", charts_metric_order(), default="Accuracy", key="curve_metric",
                          required=True, label_visibility="collapsed")
        st.plotly_chart(charts.learning_curve(per_split, metric, dark), config=charts.PLOT_CONFIG, key="curve_chart")

    with st.container(key="card_confusion"):
        h1, h2 = st.columns([2, 1], vertical_alignment="bottom")
        with h1:
            ui.card_title("Confusion matrix (80 % split)", "Rows are actual categories, columns predictions; cells show the share of each row.")
        with h2:
            cm_method = st.segmented_control("Method", list(cls["confusion_80"]), default="Hybrid", key="cm_method",
                                             required=True, label_visibility="collapsed", width="stretch")
        st.plotly_chart(charts.confusion_heatmap(cls["confusion_80"][cm_method], cls["classes"], dark),
                        config=charts.PLOT_CONFIG, key="cm_chart")

    # B. Hyper-parameters -------------------------------------------------------
    hp = ev["hyperparameters"]
    ui.section("B · Training process & hyper-parameters",
               "Training vs. validation accuracy across the SVM's regularisation strength C (5-fold cross-validation). "
               "A widening gap signals over-fitting; the plateau shows where extra capacity stops helping.", eyebrow="Experiment B")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1, st.container(key="card_csweep"):
        method = st.segmented_control("Method", ["TF-IDF", "BERT", "Hybrid"], default="Hybrid", key="sweep_method",
                                      required=True, label_visibility="collapsed")
        st.plotly_chart(charts.c_sweep(hp, method, dark, chosen_c=cls.get("C", 1.0)), config=charts.PLOT_CONFIG, key="sweep_chart")
    with c2, st.container(key="card_hparams"):
        ui.card_title("Hyper-parameters used")
        params = [
            ("Classifier", f"LinearSVC · C = {cls.get('C', 1.0):g} · squared hinge · max_iter 5000"),
            ("Category model (app)", "LinearSVC + sigmoid calibration (5-fold CV) on TF-IDF ⊕ BERT"),
            ("TF-IDF (classifier)", "unigrams · raw tf · min_df = 2 · L2-normalised · fitted on training folds"),
            ("TF-IDF (matching)", "uni + bi-grams · sub-linear tf · fitted per job on pool + job + candidates"),
            ("Sentence-BERT", f"{meta.get('bert_model', 'all-MiniLM-L6-v2').split('/')[-1]} · {meta.get('model_dim', 384)}-D · "
                              f"[mean of {meta.get('chunk_words', 150)}-word chunks ‖ first chunk] / √2"),
            ("Hybrid score", f"w = {DEFAULT_KEYWORD_WEIGHT:.2f} keyword + {1 - DEFAULT_KEYWORD_WEIGHT:.2f} semantic, pool-calibrated"),
            ("Splits", "stratified 50 / 60 / 70 / 80 / 90 % training · random_state = 42"),
        ]
        ui.html('<table class="cv-table">' + "".join(f"<tr><td style='color:var(--cv-text-2)'>{k}</td>"
                                                       f"<td style='text-align:left'>{v}</td></tr>" for k, v in params) + "</table>")

    # C. Ranking ------------------------------------------------------------------
    ui.section("C · Job → resume ranking benchmark",
               f"What the dashboard actually does: each of the {rank['n_jobs']} sample job descriptions ranks "
               f"{rank['n_candidates']} held-out resumes (never seen in training); a resume is relevant when it belongs "
               "to the job's category.", eyebrow="Experiment C")
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(key="card_rank_bars"):
        ui.card_title("Ranking quality by engine", "Mean over all jobs. Hybrid values are labelled.")
        st.plotly_chart(charts.grouped_bars(rank["summary"], ["ROC-AUC", "P@5", "MRR", "nDCG@10"], dark),
                        config=charts.PLOT_CONFIG, key="rank_bars")
    with c2, st.container(key="card_rank_sweep"):
        ui.card_title("Choosing the hybrid weight", "Ranking quality as the keyword share of the hybrid score varies.")
        st.plotly_chart(charts.weight_sweep(rank["weight_sweep"], dark, DEFAULT_KEYWORD_WEIGHT), config=charts.PLOT_CONFIG, key="weight_chart")
    with st.expander("Per-job results", icon=":material/table:"):
        per_job = pd.DataFrame(rank["per_job"])
        per_job["category"] = per_job["category"].map(charts._short)
        st.dataframe(per_job, hide_index=True, width="stretch", column_config={
            "job": "Sample job", "category": "Category",
            "Hybrid P@5": st.column_config.NumberColumn(format="%.2f"),
            "Hybrid ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Relevant median percentile": st.column_config.NumberColumn("Median pool percentile of relevant resumes", format="%.1f"),
        })

    with st.expander("Metric definitions & formulas", icon=":material/function:"):
        st.markdown(METRIC_DEFINITIONS)
    ui.footer(meta)


def charts_metric_order() -> list[str]:
    from evaluation import METRIC_ORDER

    return METRIC_ORDER


# ══════════════════════════════════════════════════════════════════════════
# SETUP / ROUTING
# ══════════════════════════════════════════════════════════════════════════
def setup_page() -> None:
    ui.page_header("Setup required", "Model artifacts not found",
                   "CV-Scan needs its trained artifacts before it can screen resumes.")
    ui.note("Run the training script once from the repository root "
            "(≈3–4 minutes; downloads all-MiniLM-L6-v2 on first use):", "info")
    st.code("python Resume_Screening_Project/src/train_models.py", language="bash")
    st.caption(f"Expected files in {MODEL_DIR}: " + ", ".join(REQUIRED_ARTIFACTS))


ui.inject_css()
if not artifacts_ready():
    setup_page()
    st.stop()

HOME_PAGE = st.Page(home_page, title="Home", icon=":material/home:", default=True)
SCREEN_PAGE = st.Page(screen_page, title="Screen resumes", icon=":material/person_search:", url_path="screen")
EVAL_PAGE = st.Page(evaluation_page, title="Model evaluation", icon=":material/monitoring:", url_path="evaluation")
st.navigation([HOME_PAGE, SCREEN_PAGE, EVAL_PAGE], position="top").run()
