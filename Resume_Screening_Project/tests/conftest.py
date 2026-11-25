import os
import sys
import warnings

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
warnings.filterwarnings("ignore")

from engine import MODEL_DIR  # noqa: E402

ARTIFACTS = ("metadata.json", "processed_resume_dataset.pkl", "dataset_bert_embeddings.pkl",
             "tfidf_vectorizer.pkl", "category_classifier.pkl", "demo_resumes.pkl")
HAVE_ARTIFACTS = all(os.path.exists(os.path.join(MODEL_DIR, a)) for a in ARTIFACTS)
needs_artifacts = pytest.mark.skipif(not HAVE_ARTIFACTS, reason="run Resume_Screening_Project/src/train_models.py first")


@pytest.fixture(scope="session")
def engine():
    if not HAVE_ARTIFACTS:
        pytest.skip("run Resume_Screening_Project/src/train_models.py first")
    from engine import ScreeningEngine

    return ScreeningEngine(MODEL_DIR)


@pytest.fixture(scope="session")
def holdout():
    import pandas as pd

    return pd.read_pickle(os.path.join(MODEL_DIR, "demo_resumes.pkl"))
