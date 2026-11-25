"""
engine.py — CV-Scan scoring engine (keyword, semantic and hybrid matching).

One screening run scores N candidate resumes against one job description (JD):

1. Keyword engine — TF-IDF + cosine similarity
   A TF-IDF space is fitted on the historical pool + the JD + the candidates,
   so every JD term is part of the vocabulary and IDF weights reflect the whole
   collection. Because rows are L2-normalised, cosine(JD, CV) = Σ wJD·wCV, which
   lets us attribute the score to individual terms (explainability).

2. Semantic engine — Sentence-BERT (all-MiniLM-L6-v2) + cosine similarity
   The model reads at most 256 word-pieces, so documents are split into
   150-word chunks. Each document is represented by two 384-D views:
     • the length-weighted mean of all chunk embeddings (the *whole* resume)
     • the first chunk (title, summary and most recent role)
   concatenated and scaled by 1/√2. The cosine of two such vectors is the
   average of the whole-document and the header similarity. On held-out data
   this beats a single mean-pooled vector for both ranking and classification.

3. Calibration
   Raw cosines live on different scales (TF-IDF ≈ 0–0.4, BERT ≈ 0.1–0.8). Each
   is min–max normalised against the historical pool scored for the *same* JD:
   100 % means "as close to this JD as the best historical resume".

4. Hybrid score = w · keyword + (1 − w) · semantic   (default w = 0.30)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cached_property

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import chunk_words, clean_text, prepare_semantic_text, word_count

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 150
DEFAULT_KEYWORD_WEIGHT = 0.30
TOP_JD_TERMS = 25

# Text is pre-tokenised by utils.clean_text, so the vectorizer splits on spaces.
TFIDF_PARAMS = dict(
    token_pattern=r"\S+",
    lowercase=False,
    sublinear_tf=True,
    ngram_range=(1, 2),
    dtype=np.float32,
)

# Score bands used for the recommendation badge (applied to the 0–1 score).
VERDICTS = (
    (0.70, "Excellent match"),
    (0.50, "Strong match"),
    (0.30, "Partial match"),
    (0.00, "Weak match"),
)


def verdict_for(score: float) -> str:
    for threshold, label in VERDICTS:
        if score >= threshold:
            return label
    return VERDICTS[-1][1]


# ── Embeddings ─────────────────────────────────────────────────────────────
def load_bert_model(name: str = BERT_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed_documents(model, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Embed long documents as [mean of all chunks ‖ first chunk] / √2 (unit norm)."""
    chunks: list[str] = []
    owners: list[int] = []
    weights: list[float] = []
    first_chunk: list[int] = []
    for i, text in enumerate(texts):
        doc_chunks = chunk_words(prepare_semantic_text(text), CHUNK_WORDS) or [""]
        first_chunk.append(len(chunks))
        for chunk in doc_chunks:
            chunks.append(chunk)
            owners.append(i)
            weights.append(max(word_count(chunk), 1))

    chunk_vectors = model.encode(
        chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    ).astype(np.float32)

    pooled = np.zeros((len(texts), chunk_vectors.shape[1]), dtype=np.float32)
    np.add.at(pooled, np.asarray(owners), chunk_vectors * np.asarray(weights, dtype=np.float32)[:, None])
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pooled /= norms
    return np.hstack([pooled, chunk_vectors[first_chunk]]) / np.float32(np.sqrt(2.0))


def hybrid_features(class_vectorizer, clean_texts, embeddings):
    """Feature matrix used by the category classifier: [TF-IDF | BERT]."""
    return sparse.hstack(
        [class_vectorizer.transform(clean_texts), sparse.csr_matrix(embeddings)], format="csr"
    )


def _minmax(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = hi - lo
    if span <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / span, 0.0, 1.0).astype(np.float32)


def _is_phrase(term: str, vectorizer, df_pool) -> bool:
    """Unigrams always; bigrams only when they behave like a real phrase."""
    if " " not in term:
        return True
    a, b = term.split(" ")
    df = df_pool[vectorizer.vocabulary_[term]]
    rarer = min(df_pool[vectorizer.vocabulary_[a]], df_pool[vectorizer.vocabulary_[b]])
    return a != b and df >= 3 and df / max(rarer, 1) >= 0.12


def select_key_terms(vocab, vectorizer, jd_idx, jd_w, df_pool, specificity, k: int = TOP_JD_TERMS) -> list[int]:
    """Pick the job's key requirements from its TF-IDF vector.

    Raw TF-IDF ranking favours word pairs that occur *only* in the JD
    ("support full"), so terms must (1) appear in ≥ 2 historical resumes,
    (2) for bigrams, be cohesive phrases (≥ 3 resumes and ≥ 12 % of the
    rarer word's occurrences), and (3) be domain vocabulary — concentrated in
    some of the 24 categories rather than spread evenly (normalised category
    entropy). Unigrams already covered by a chosen phrase are skipped.
    """
    chosen: list[int] = []
    covered: set[str] = set()
    for j in np.argsort(-jd_w):
        col = jd_idx[j]
        term = str(vocab[col])
        df = df_pool[col]
        if df < 2 or specificity[j] < 0.12:
            continue
        if " " in term:
            if not _is_phrase(term, vectorizer, df_pool):
                continue
        elif term in covered:
            continue
        chosen.append(j)
        covered.update(term.split(" "))
        if len(chosen) >= k:
            break
    if len(chosen) < 5:  # unusual or very short JD: fall back to attested terms
        for j in np.argsort(-jd_w):
            if j not in chosen and df_pool[jd_idx[j]] >= 1:
                chosen.append(j)
            if len(chosen) >= min(k, len(jd_w)):
                break
    return chosen


# ── Result containers ──────────────────────────────────────────────────────
@dataclass
class CandidateResult:
    name: str
    text: str
    word_count: int
    keyword_raw: float
    keyword_score: float
    semantic_raw: float | None
    semantic_score: float | None
    hybrid_score: float | None
    score: float
    percentile: float
    verdict: str
    coverage: float
    matched: list[tuple[str, float]]
    missing: list[tuple[str, float]]
    categories: list[tuple[str, float]] = field(default_factory=list)
    embedding: np.ndarray | None = None
    rank: int = 0
    other_share: float = 0.0  # keyword-score share from non-phrase word pairs (hidden in the chart)


@dataclass
class ScreeningRun:
    mode: str
    keyword_weight: float
    jd_text: str
    jd_terms: list[tuple[str, float]]
    jd_categories: list[tuple[str, float]]
    jd_embedding: np.ndarray | None
    pool_keyword: np.ndarray
    pool_semantic: np.ndarray | None
    pool_score: np.ndarray
    pool_coverage: np.ndarray
    pool_top: pd.DataFrame
    candidates: list[CandidateResult]

    @property
    def is_advanced(self) -> bool:
        return self.mode == "advanced"


# ── Engine ─────────────────────────────────────────────────────────────────
class ScreeningEngine:
    """Loads the trained artifacts and scores resumes against a job description."""

    def __init__(self, model_dir: str = MODEL_DIR, bert_model=None):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "metadata.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.pool: pd.DataFrame = pd.read_pickle(os.path.join(model_dir, "processed_resume_dataset.pkl"))
        self.pool_embeddings: np.ndarray = joblib.load(os.path.join(model_dir, "dataset_bert_embeddings.pkl"))
        self.class_vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        self.classifier = joblib.load(os.path.join(model_dir, "category_classifier.pkl"))
        self._bert = bert_model
        self._pool_clean = self.pool["clean_text"].tolist()
        categories = self.pool["Category"].to_numpy()
        self._cat_onehot = np.stack([categories == c for c in sorted(set(categories))], axis=1).astype(np.float32)
        self._cat_sizes = self._cat_onehot.sum(axis=0)

    # Lazily loaded so keyword-only use never pays for the transformer.
    @property
    def bert(self):
        if self._bert is None:
            self._bert = load_bert_model(self.meta.get("bert_model", BERT_MODEL_NAME))
        return self._bert

    def embed(self, texts: list[str]) -> np.ndarray:
        return embed_documents(self.bert, texts)

    @cached_property
    def pca(self):
        """3-component PCA fitted on the historical pool (a stable semantic map)."""
        from sklearn.decomposition import PCA

        return PCA(n_components=3, random_state=0).fit(self.pool_embeddings)

    @cached_property
    def pool_projection(self) -> np.ndarray:
        return self.pca.transform(self.pool_embeddings)

    def predict_categories(self, clean_texts: list[str], embeddings: np.ndarray, top_k: int = 3):
        proba = self.classifier.predict_proba(hybrid_features(self.class_vectorizer, clean_texts, embeddings))
        classes = self.classifier.classes_
        out = []
        for row in proba:
            order = np.argsort(row)[::-1][:top_k]
            out.append([(str(classes[i]), float(row[i])) for i in order])
        return out

    # ------------------------------------------------------------------
    def screen(
        self,
        jd_text: str,
        candidates: list[tuple[str, str]],
        mode: str = "advanced",
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        candidate_embeddings: np.ndarray | None = None,
        jd_embedding: np.ndarray | None = None,
    ) -> ScreeningRun:
        """Score ``candidates`` (list of ``(name, raw_text)``) against ``jd_text``."""
        if mode not in ("basic", "advanced"):
            raise ValueError(f"Unknown mode: {mode}")
        jd_clean = clean_text(jd_text)
        if len(jd_clean.split()) < 3:
            raise ValueError("The job description is too short — add the role's duties and required skills.")
        if not candidates:
            raise ValueError("Add at least one resume to screen.")
        cand_clean = [clean_text(text) for _, text in candidates]
        keyword_weight = float(np.clip(keyword_weight, 0.0, 1.0))
        advanced = mode == "advanced"
        n_pool = len(self._pool_clean)

        # 1. Keyword engine ------------------------------------------------
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        matrix = vectorizer.fit_transform(self._pool_clean + [jd_clean] + cand_clean).tocsr()
        pool_m, jd_vec, cand_m = matrix[:n_pool], matrix[n_pool], matrix[n_pool + 1 :]
        pool_kw_raw = np.asarray((pool_m @ jd_vec.T).todense()).ravel()
        cand_kw_raw = np.asarray((cand_m @ jd_vec.T).todense()).ravel()
        kw_lo, kw_hi = float(pool_kw_raw.min()), float(pool_kw_raw.max())
        pool_kw = _minmax(pool_kw_raw, kw_lo, kw_hi)
        cand_kw = _minmax(cand_kw_raw, kw_lo, kw_hi)

        vocab = vectorizer.get_feature_names_out()
        jd_idx, jd_w = jd_vec.indices, jd_vec.data
        df_pool = np.bincount(pool_m.indices, minlength=len(vocab))
        presence = pool_m[:, jd_idx].tocsc()
        presence.data[:] = 1.0
        rates = np.asarray(presence.T @ self._cat_onehot) / self._cat_sizes
        probs = rates / np.maximum(rates.sum(axis=1, keepdims=True), 1e-12)
        entropy = -(np.where(probs > 0, probs * np.log(np.where(probs > 0, probs, 1.0)), 0.0)).sum(axis=1)
        specificity = 1.0 - entropy / np.log(max(self._cat_onehot.shape[1], 2))
        key = select_key_terms(vocab, vectorizer, jd_idx, jd_w, df_pool, specificity)
        # Bigrams are built on stop-word-free text, so some span removed words
        # ("relations and performance" → "relation performance"). Only cohesive
        # phrases are listed individually in the explanation chart.
        displayable = np.array([_is_phrase(str(vocab[c]), vectorizer, df_pool) for c in jd_idx], dtype=bool)
        top_idx = jd_idx[key]
        jd_terms = [(str(vocab[jd_idx[j]]), float(jd_w[j])) for j in key]
        # Coverage of the JD's key terms, also for the pool (for benchmarking).
        pool_cov = np.asarray((pool_m[:, top_idx] > 0).sum(axis=1)).ravel() / max(len(top_idx), 1)

        # 2. Semantic engine -----------------------------------------------
        pool_sem = cand_sem = cand_sem_raw = None
        jd_categories: list[tuple[str, float]] = []
        cand_categories: list[list[tuple[str, float]]] = [[] for _ in candidates]
        cand_emb = None
        if advanced:
            if candidate_embeddings is None:
                to_embed = ([] if jd_embedding is not None else [jd_text]) + [text for _, text in candidates]
                vectors = self.embed(to_embed)
                if jd_embedding is None:
                    jd_embedding, cand_emb = vectors[0], vectors[1:]
                else:
                    cand_emb = vectors
            else:
                cand_emb = np.asarray(candidate_embeddings, dtype=np.float32)
                if jd_embedding is None:
                    jd_embedding = self.embed([jd_text])[0]
            pool_sem_raw = self.pool_embeddings @ jd_embedding
            cand_sem_raw = cand_emb @ jd_embedding
            sem_lo, sem_hi = float(pool_sem_raw.min()), float(pool_sem_raw.max())
            pool_sem = _minmax(pool_sem_raw, sem_lo, sem_hi)
            cand_sem = _minmax(cand_sem_raw, sem_lo, sem_hi)

            pool_score = keyword_weight * pool_kw + (1 - keyword_weight) * pool_sem
            cand_score = keyword_weight * cand_kw + (1 - keyword_weight) * cand_sem

            predictions = self.predict_categories([jd_clean] + cand_clean, np.vstack([jd_embedding, cand_emb]))
            jd_categories, cand_categories = predictions[0], predictions[1:]
        else:
            jd_embedding = None
            pool_score, cand_score = pool_kw, cand_kw

        # 3. Per-candidate explanations -------------------------------------
        results: list[CandidateResult] = []
        for i, (name, text) in enumerate(candidates):
            cv_weights = cand_m[i, jd_idx].toarray().ravel()
            contrib = jd_w * cv_weights
            total = float(contrib.sum())
            matched_order = np.argsort(-contrib)
            matched = [
                (str(vocab[jd_idx[j]]), float(contrib[j] / total))
                for j in matched_order
                if contrib[j] > 0 and displayable[j]
            ]
            other_share = max(0.0, 1.0 - sum(share for _, share in matched)) if total > 0 else 0.0
            present_top = cand_m[i, top_idx].toarray().ravel() > 0
            missing = [pair for pair, present in zip(jd_terms, present_top) if not present]
            score = float(cand_score[i])
            results.append(
                CandidateResult(
                    name=name,
                    text=text,
                    word_count=word_count(text),
                    keyword_raw=float(cand_kw_raw[i]),
                    keyword_score=float(cand_kw[i]),
                    semantic_raw=None if cand_sem_raw is None else float(cand_sem_raw[i]),
                    semantic_score=None if cand_sem is None else float(cand_sem[i]),
                    hybrid_score=score if advanced else None,
                    score=score,
                    percentile=float((pool_score < score).mean() * 100),
                    verdict=verdict_for(round(score, 2)),  # consistent with the displayed percentage
                    coverage=float(present_top.mean()) if len(present_top) else 0.0,
                    matched=matched,
                    missing=missing,
                    categories=cand_categories[i],
                    embedding=None if cand_emb is None else cand_emb[i],
                    other_share=other_share,
                )
            )

        for rank, result in enumerate(sorted(results, key=lambda r: r.score, reverse=True), start=1):
            result.rank = rank

        top = np.argsort(-pool_score)[:8]
        pool_top = pd.DataFrame(
            {
                "ID": self.pool["ID"].to_numpy()[top],
                "Category": self.pool["Category"].to_numpy()[top],
                "Keyword": pool_kw[top],
                "Semantic": pool_sem[top] if pool_sem is not None else np.nan,
                "Score": pool_score[top],
            }
        )

        return ScreeningRun(
            mode=mode,
            keyword_weight=keyword_weight,
            jd_text=jd_text,
            jd_terms=jd_terms,
            jd_categories=jd_categories,
            jd_embedding=jd_embedding,
            pool_keyword=pool_kw,
            pool_semantic=pool_sem,
            pool_score=np.asarray(pool_score, dtype=np.float32),
            pool_coverage=pool_cov.astype(np.float32),
            pool_top=pool_top,
            candidates=sorted(results, key=lambda r: r.rank),
        )
