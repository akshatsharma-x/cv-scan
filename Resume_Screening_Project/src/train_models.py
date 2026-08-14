"""
train_models.py — One-time script to build and persist all ML artifacts.

Run this once:
    python src/train_models.py

It will:
  1. Load the CSV dataset.
  2. Clean resume text.
  3. Fit TF-IDF vectorizer + MinMaxScaler.
  4. Generate BERT embeddings for the historical dataset.
  5. Save everything to the  models/  directory via joblib.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Add parent for utils import
sys.path.insert(0, os.path.dirname(__file__))
from utils import advanced_clean_text, ensure_nltk_resources

# ── CONFIG ──────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "Resume", "Resume.csv")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
SUBSET_SIZE  = 100  # Use first N resumes (same as notebook demo)

# ── SAMPLE JD (used to fit the scaler during training) ──
SAMPLE_JD = """
We are looking for a Human Resources Specialist with experience in recruitment,
benefits administration, and employee relations. Candidates should have strong
communication skills, experience with HR software, and a background in payroll management.
"""


def main():
    ensure_nltk_resources()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load dataset
    print(f"📂 Loading dataset from {os.path.abspath(DATASET_PATH)} ...")
    df = pd.read_csv(DATASET_PATH)
    print(f"   Shape: {df.shape}")

    if "Resume_str" in df.columns:
        df["resume_text"] = df["Resume_str"]

    # Grab 45 resumes equally from every professional Category
    # Guaranteeing a perfectly uniform distribution of classes for the SVM evaluation
    print("⚖️ Balancing class distribution...")
    resume_subset = df.groupby('Category').head(45).copy()

    # 2. Clean text
    print("🧹 Cleaning text …")
    cleaned_jd = advanced_clean_text(SAMPLE_JD)
    resume_subset["cleaned_resume"] = resume_subset["resume_text"].apply(advanced_clean_text)

    # 3. TF-IDF
    print("📊 Fitting TF-IDF vectorizer …")
    all_texts = [cleaned_jd] + resume_subset["cleaned_resume"].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    resume_subset["tfidf_score"] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # 4. BERT embeddings
    print("🤖 Encoding with BERT (all-MiniLM-L6-v2) …")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_embedding = bert_model.encode([cleaned_jd])
    resume_embeddings = bert_model.encode(resume_subset["cleaned_resume"].tolist(), show_progress_bar=True)
    resume_subset["bert_score"] = cosine_similarity(jd_embedding, resume_embeddings).flatten()

    # 5. Normalize & Hybrid
    print("⚙️  Normalizing scores …")
    scaler = MinMaxScaler()
    resume_subset[["tfidf_norm", "bert_norm"]] = scaler.fit_transform(
        resume_subset[["tfidf_score", "bert_score"]]
    )
    resume_subset["hybrid_score"] = (0.3 * resume_subset["tfidf_norm"]) + (0.7 * resume_subset["bert_norm"])

    # Ensure ID column
    if "ID" not in resume_subset.columns:
        resume_subset["ID"] = resume_subset.index.astype(str)

    # 6. Save artifacts
    print("💾 Saving model artifacts …")
    joblib.dump(vectorizer,        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(scaler,            os.path.join(MODEL_DIR, "minmax_scaler.pkl"))
    joblib.dump(tfidf_matrix,      os.path.join(MODEL_DIR, "dataset_tfidf_matrix.pkl"))
    joblib.dump(resume_embeddings, os.path.join(MODEL_DIR, "dataset_bert_embeddings.pkl"))
    resume_subset.to_pickle(       os.path.join(MODEL_DIR, "processed_resume_dataset.pkl"))

    print("\n✅ All models saved to:", os.path.abspath(MODEL_DIR))
    print("   You can now run:  streamlit run src/app.py")


if __name__ == "__main__":
    main()
