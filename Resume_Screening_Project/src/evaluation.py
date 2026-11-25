"""
evaluation.py — Experimental evaluation of the three methods.

Two complementary experiments are run on the historical resume pool:

A. Category classification (what the assignment's evaluation measures expect)
   Linear SVM classifiers predict a resume's professional category from
     • TF-IDF features              (lexical method)
     • Sentence-BERT embeddings     (semantic method)
     • [TF-IDF | BERT] concatenated (hybrid method)
   for training splits of 50–90 %. The TF-IDF vocabulary is fitted on the
   training portion only, so no statistics leak from the test set. A
   hyper-parameter sweep over the SVM's C (5-fold CV) visualises the training
   process (train vs. validation accuracy).

B. JD → resume ranking (what the dashboard actually does)
   Each built-in sample job description is scored against held-out resumes
   that were never part of the pool. A resume is relevant when it belongs to
   the job's category. ROC-AUC, Precision@5, MRR and nDCG@10 are reported for
   the keyword, semantic and hybrid scores, plus a sweep of the hybrid weight.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

METRIC_ORDER = [
    "Specificity", "Sensitivity", "Accuracy", "Precision", "FPR",
    "FNR", "NPV", "FDR", "F1-Score", "MCC",
]
METHODS = ["TF-IDF", "BERT", "Hybrid"]
TRAIN_SIZES = (0.5, 0.6, 0.7, 0.8, 0.9)
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
CLASSIFIER_TFIDF_PARAMS = dict(
    token_pattern=r"\S+", lowercase=False, sublinear_tf=False, min_df=2, dtype=np.float32
)
DEFAULT_C = 1.0


def compute_metrics(y_true, y_pred, labels) -> dict[str, float]:
    """Confusion-matrix measures, macro-averaged one-vs-rest over all classes."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    def ratio(a, b):
        return float(np.mean(np.divide(a, b, out=np.zeros_like(a), where=b > 0)))

    return {
        "Specificity": ratio(tn, tn + fp),
        "Sensitivity": ratio(tp, tp + fn),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": ratio(tp, tp + fp),
        "FPR": ratio(fp, fp + tn),
        "FNR": ratio(fn, fn + tp),
        "NPV": ratio(tn, tn + fn),
        "FDR": ratio(fp, fp + tp),
        "F1-Score": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def _features(train_text, test_text, train_emb, test_emb):
    vectorizer = TfidfVectorizer(**CLASSIFIER_TFIDF_PARAMS)
    tf_train = vectorizer.fit_transform(train_text)
    tf_test = vectorizer.transform(test_text)
    bert_train, bert_test = sparse.csr_matrix(train_emb), sparse.csr_matrix(test_emb)
    return {
        "TF-IDF": (tf_train, tf_test),
        "BERT": (train_emb, test_emb),
        "Hybrid": (
            sparse.hstack([tf_train, bert_train], format="csr"),
            sparse.hstack([tf_test, bert_test], format="csr"),
        ),
    }


def _svm(C: float = DEFAULT_C) -> LinearSVC:
    return LinearSVC(C=C, max_iter=5000, random_state=42)


def evaluate_classifiers(clean_texts, embeddings, labels, train_sizes=TRAIN_SIZES, C=DEFAULT_C, seed=42) -> dict:
    texts = np.asarray(clean_texts, dtype=object)
    y = np.asarray(labels)
    classes = sorted(set(y.tolist()))
    idx = np.arange(len(y))
    per_split: dict[str, dict[str, dict[str, float]]] = {m: {} for m in METHODS}
    confusion = {}
    for size in train_sizes:
        tr, te = train_test_split(idx, train_size=size, random_state=seed, stratify=y)
        feats = _features(texts[tr], texts[te], embeddings[tr], embeddings[te])
        for method in METHODS:
            X_train, X_test = feats[method]
            pred = _svm(C).fit(X_train, y[tr]).predict(X_test)
            per_split[method][f"{int(size * 100)}%"] = compute_metrics(y[te], pred, classes)
            if abs(size - 0.8) < 1e-9:
                confusion[method] = confusion_matrix(y[te], pred, labels=classes).tolist()
    return {"per_split": per_split, "confusion_80": confusion, "classes": classes, "C": C}


def hyperparameter_sweep(clean_texts, embeddings, labels, Cs=C_GRID, folds=5, seed=42) -> dict:
    """5-fold CV train/validation accuracy of each method across the SVM's C."""
    texts = np.asarray(clean_texts, dtype=object)
    y = np.asarray(labels)
    scores = {m: {"train": [[] for _ in Cs], "val": [[] for _ in Cs]} for m in METHODS}
    for tr, va in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(texts, y):
        feats = _features(texts[tr], texts[va], embeddings[tr], embeddings[va])
        for method in METHODS:
            X_train, X_val = feats[method]
            for k, C in enumerate(Cs):
                clf = _svm(C).fit(X_train, y[tr])
                scores[method]["train"][k].append(accuracy_score(y[tr], clf.predict(X_train)))
                scores[method]["val"][k].append(accuracy_score(y[va], clf.predict(X_val)))
    out = {"C": list(Cs), "folds": folds}
    for method in METHODS:
        out[method] = {
            split: {"mean": [float(np.mean(v)) for v in vals], "std": [float(np.std(v)) for v in vals]}
            for split, vals in scores[method].items()
        }
    return out


# ── Ranking benchmark ──────────────────────────────────────────────────────
def _ranking_metrics(scores: np.ndarray, relevant: np.ndarray, k: int = 5) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    rel_sorted = relevant[order]
    first_hit = int(np.argmax(rel_sorted)) if rel_sorted.any() else len(rel_sorted)
    gains = rel_sorted[:10] / np.log2(np.arange(2, 12))
    ideal = np.sort(relevant)[::-1][:10] / np.log2(np.arange(2, 12))
    return {
        "ROC-AUC": float(roc_auc_score(relevant, scores)),
        "P@5": float(rel_sorted[:k].mean()),
        "MRR": 1.0 / (first_hit + 1),
        "nDCG@10": float(gains.sum() / ideal.sum()) if ideal.sum() else 0.0,
    }


def ranking_benchmark(engine, jobs: dict, holdout, weights=np.round(np.linspace(0, 1, 11), 2)) -> dict:
    """Score held-out resumes against every sample job with the live engine."""
    texts = holdout["text"].tolist()
    categories = holdout["Category"].to_numpy()
    names = [f"{c}-{i}" for c, i in zip(categories, holdout["ID"])]
    embeddings = engine.embed(texts)
    candidates = list(zip(names, texts))

    per_method = {m: [] for m in ("Keyword", "Semantic", "Hybrid")}
    sweep = {float(w): [] for w in weights}
    per_job = []
    for title, job in jobs.items():
        run = engine.screen(job["text"], candidates, mode="advanced", candidate_embeddings=embeddings)
        by_name = {c.name: c for c in run.candidates}
        kw = np.array([by_name[n].keyword_score for n in names])
        sem = np.array([by_name[n].semantic_score for n in names])
        hyb = np.array([by_name[n].hybrid_score for n in names])
        relevant = (categories == job["category"]).astype(int)
        if relevant.sum() == 0:
            continue
        m_kw, m_sem, m_hyb = (_ranking_metrics(s, relevant) for s in (kw, sem, hyb))
        per_method["Keyword"].append(m_kw)
        per_method["Semantic"].append(m_sem)
        per_method["Hybrid"].append(m_hyb)
        for w in weights:
            sweep[float(w)].append(_ranking_metrics(w * kw + (1 - w) * sem, relevant))
        rel_pct = [by_name[n].percentile for n, r in zip(names, relevant) if r]
        per_job.append(
            {"job": title, "category": job["category"], "Hybrid P@5": m_hyb["P@5"],
             "Hybrid ROC-AUC": m_hyb["ROC-AUC"], "Relevant median percentile": float(np.median(rel_pct))}
        )

    def mean_of(rows):
        return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}

    return {
        "n_jobs": len(per_job),
        "n_candidates": len(names),
        "summary": {m: mean_of(rows) for m, rows in per_method.items()},
        "weight_sweep": {
            "weights": [float(w) for w in weights],
            **{key: [float(np.mean([r[key] for r in sweep[float(w)]])) for w in weights]
               for key in ("ROC-AUC", "P@5", "MRR", "nDCG@10")},
        },
        "per_job": per_job,
    }


def run_full_evaluation(engine, holdout, jobs, log=print) -> dict:
    texts = engine.pool["clean_text"].tolist()
    labels = engine.pool["Category"].tolist()
    t0 = time.time()
    log("⚙️  Classification experiment (TF-IDF / BERT / Hybrid × 5 training splits) …")
    classification = evaluate_classifiers(texts, engine.pool_embeddings, labels)
    log("🎛️  Hyper-parameter sweep (C, 5-fold CV) …")
    sweep = hyperparameter_sweep(texts, engine.pool_embeddings, labels)
    log("🏁 JD → resume ranking benchmark …")
    ranking = ranking_benchmark(engine, jobs, holdout)
    return {
        "classification": classification,
        "hyperparameters": sweep,
        "ranking": ranking,
        "n_samples": len(labels),
        "n_classes": len(set(labels)),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
