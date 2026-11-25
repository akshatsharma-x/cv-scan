"""
train_models.py — Build and persist every artifact the dashboard needs.

Run once from the repository root (paths are resolved relative to this file):
    python Resume_Screening_Project/src/train_models.py              # train + evaluate
    python Resume_Screening_Project/src/train_models.py --skip-eval

Steps
  1. Load ../dataset/Resume/Resume.csv (2,484 resumes, 24 categories).
  2. Hold out 5 resumes per category; they are never seen by any model and are
     used as demo candidates and for the JD → resume ranking benchmark.
  3. Clean the pool text for TF-IDF (utils.clean_text).
  4. Embed the pool with Sentence-BERT using chunked mean pooling.
  5. Train the category classifier (calibrated linear SVM on [TF-IDF | BERT]).
  6. Save artifacts + metadata to ../models and run the evaluation suite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(__file__))
from engine import (  # noqa: E402
    BERT_MODEL_NAME, CHUNK_WORDS, MODEL_DIR, TFIDF_PARAMS, ScreeningEngine,
    embed_documents, hybrid_features, load_bert_model,
)
from evaluation import CLASSIFIER_TFIDF_PARAMS, DEFAULT_C, run_full_evaluation  # noqa: E402
from sample_data import SAMPLE_JOBS  # noqa: E402
from utils import clean_text, ensure_nltk_resources, resume_title, word_count  # noqa: E402

DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "Resume", "Resume.csv")
)
HOLDOUT_PER_CATEGORY = 5
SEED = 42


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "Resume_str" if "Resume_str" in df.columns else "resume_text"
    df = df.rename(columns={text_col: "text"})[["ID", "text", "Category"]]
    df["text"] = df["text"].fillna("").astype(str)
    before = len(df)
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"   {before:,} rows → {len(df):,} after removing empty / duplicate resumes")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-eval", action="store_true", help="skip the (≈1–2 min) evaluation suite")
    args = parser.parse_args()

    t_start = time.time()
    ensure_nltk_resources()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"📂 Loading dataset from {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)

    print(f"🧪 Holding out {HOLDOUT_PER_CATEGORY} resumes per category …")
    holdout = (
        df.groupby("Category", group_keys=False)
        .sample(n=HOLDOUT_PER_CATEGORY, random_state=SEED)
        .reset_index(drop=True)
    )
    pool = df[~df["ID"].isin(holdout["ID"])].reset_index(drop=True)
    print(f"   pool: {len(pool):,} resumes · hold-out: {len(holdout)} resumes")

    print("🧹 Cleaning text (URLs, e-mails, punctuation, stop-words, lemmatisation) …")
    pool["clean_text"] = pool["text"].map(clean_text)
    pool["word_count"] = pool["text"].map(word_count)
    pool["title"] = pool["text"].map(resume_title)

    print(f"🤖 Embedding the pool with {BERT_MODEL_NAME} ({CHUNK_WORDS}-word chunks → [mean ‖ first chunk]) …")
    bert = load_bert_model()
    t0 = time.time()
    embeddings = embed_documents(bert, pool["text"].tolist(), show_progress=True)
    print(f"   {embeddings.shape} in {time.time() - t0:.0f}s")

    print("🏷️  Training the category classifier (calibrated linear SVM on TF-IDF ⊕ BERT) …")
    class_vectorizer = TfidfVectorizer(**CLASSIFIER_TFIDF_PARAMS).fit(pool["clean_text"])
    classifier = CalibratedClassifierCV(LinearSVC(C=DEFAULT_C, max_iter=5000, random_state=SEED), cv=5, ensemble=False)
    classifier.fit(hybrid_features(class_vectorizer, pool["clean_text"], embeddings), pool["Category"])

    print("💾 Saving artifacts …")
    pool[["ID", "Category", "title", "word_count", "clean_text"]].to_pickle(
        os.path.join(MODEL_DIR, "processed_resume_dataset.pkl")
    )
    joblib.dump(embeddings, os.path.join(MODEL_DIR, "dataset_bert_embeddings.pkl"))
    joblib.dump(class_vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(classifier, os.path.join(MODEL_DIR, "category_classifier.pkl"))
    holdout[["ID", "Category", "text"]].to_pickle(os.path.join(MODEL_DIR, "demo_resumes.pkl"))

    import sentence_transformers
    import sklearn

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "Resume.csv (Kaggle resume dataset, 24 categories)",
        "n_resumes": int(len(pool)),
        "n_holdout": int(len(holdout)),
        "n_categories": int(pool["Category"].nunique()),
        "categories": sorted(pool["Category"].unique().tolist()),
        "category_counts": pool["Category"].value_counts().sort_index().to_dict(),
        "bert_model": BERT_MODEL_NAME,
        "embedding_dim": int(embeddings.shape[1]),
        "model_dim": int(embeddings.shape[1] // 2),
        "embedding_views": ["length-weighted mean of 150-word chunks", "first chunk (title & summary)"],
        "chunk_words": CHUNK_WORDS,
        "tfidf_match_params": {k: str(v) for k, v in TFIDF_PARAMS.items()},
        "tfidf_classifier_vocabulary": int(len(class_vectorizer.vocabulary_)),
        "classifier": f"CalibratedClassifierCV(LinearSVC(C={DEFAULT_C}), method='sigmoid', cv=5, ensemble=False)",
        "versions": {"scikit-learn": sklearn.__version__, "sentence-transformers": sentence_transformers.__version__},
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    # Obsolete artifacts from the previous pipeline.
    for stale in ("minmax_scaler.pkl", "dataset_tfidf_matrix.pkl"):
        path = os.path.join(MODEL_DIR, stale)
        if os.path.exists(path):
            os.remove(path)

    if not args.skip_eval:
        print("📊 Running the evaluation suite …")
        engine = ScreeningEngine(MODEL_DIR, bert_model=bert)
        results = run_full_evaluation(engine, holdout, SAMPLE_JOBS)
        results["created_at"] = metadata["created_at"]
        with open(os.path.join(MODEL_DIR, "evaluation_results.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        hybrid80 = results["classification"]["per_split"]["Hybrid"]["80%"]
        ranking = results["ranking"]["summary"]["Hybrid"]
        print(f"   Hybrid classifier @80% split: accuracy {hybrid80['Accuracy']:.3f}, "
              f"F1 {hybrid80['F1-Score']:.3f}, MCC {hybrid80['MCC']:.3f}")
        print(f"   Hybrid ranking: ROC-AUC {ranking['ROC-AUC']:.3f}, P@5 {ranking['P@5']:.3f}")

    print(f"\n✅ Done in {time.time() - t_start:.0f}s — artifacts in {MODEL_DIR}")
    print("   Launch the dashboard with:  streamlit run Resume_Screening_Project/src/app.py")


if __name__ == "__main__":
    main()
