import numpy as np
import pytest

from conftest import needs_artifacts
from sample_data import SAMPLE_JOBS

pytestmark = needs_artifacts


@pytest.fixture(scope="module")
def batch(holdout):
    """Three HR resumes and three from other categories (all held out)."""
    hr = holdout[holdout["Category"] == "HR"].head(3)
    others = holdout[holdout["Category"].isin(["CHEF", "AVIATION", "TEACHER"])].groupby("Category").head(1)
    rows = list(hr.itertuples()) + list(others.itertuples())
    return [(f"{r.Category}-{r.ID}", r.text) for r in rows]


@pytest.fixture(scope="module")
def advanced_run(engine, batch):
    return engine.screen(SAMPLE_JOBS["HR Generalist"]["text"], batch, mode="advanced")


def test_on_target_resumes_rank_first(advanced_run):
    top3 = {c.name.split("-")[0] for c in advanced_run.candidates[:3]}
    assert top3 == {"HR"}


def test_scores_are_calibrated_and_consistent(advanced_run):
    run = advanced_run
    w = run.keyword_weight
    for c in run.candidates:
        for value in (c.keyword_score, c.semantic_score, c.hybrid_score, c.coverage):
            assert 0.0 <= value <= 1.0
        assert c.hybrid_score == pytest.approx(w * c.keyword_score + (1 - w) * c.semantic_score, abs=1e-6)
        assert 0.0 <= c.percentile <= 100.0
    # Pool calibration: min-max against the pool scored for the same JD.
    assert run.pool_keyword.min() == pytest.approx(0.0) and run.pool_keyword.max() == pytest.approx(1.0)
    assert run.pool_semantic.min() == pytest.approx(0.0) and run.pool_semantic.max() == pytest.approx(1.0)
    assert [c.rank for c in run.candidates] == list(range(1, len(run.candidates) + 1))
    assert all(a.score >= b.score for a, b in zip(run.candidates, run.candidates[1:]))


def test_percentile_matches_pool(advanced_run):
    c = advanced_run.candidates[0]
    assert c.percentile == pytest.approx((advanced_run.pool_score < c.score).mean() * 100)


def test_keyword_explanation_adds_up(advanced_run):
    run = advanced_run
    best = run.candidates[0]
    assert sum(share for _, share in best.matched) + best.other_share == pytest.approx(1.0, abs=1e-4)
    key_terms = [t for t, _ in run.jd_terms]
    missing = [t for t, _ in best.missing]
    assert set(missing) <= set(key_terms)
    assert best.coverage == pytest.approx(1 - len(missing) / len(key_terms))


def test_key_terms_are_real_vocabulary(advanced_run):
    terms = {t for t, _ in advanced_run.jd_terms}
    assert {"payroll", "onboarding", "recruiting"} <= terms
    assert not {"support full", "full employee", "lifecycle full"} & terms  # JD-only word pairs


def test_category_prediction(advanced_run):
    assert advanced_run.jd_categories[0][0] == "HR"
    best = advanced_run.candidates[0]
    assert best.categories and abs(sum(p for _, p in best.categories)) <= 1.0 + 1e-6


def test_basic_mode_is_keyword_only(engine, batch):
    run = engine.screen(SAMPLE_JOBS["HR Generalist"]["text"], batch, mode="basic")
    assert run.mode == "basic" and run.pool_semantic is None and run.jd_embedding is None
    for c in run.candidates:
        assert c.semantic_score is None and c.hybrid_score is None and c.score == c.keyword_score
    assert run.candidates[0].name.startswith("HR")


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_weight_extremes(engine, batch, weight):
    run = engine.screen(SAMPLE_JOBS["HR Generalist"]["text"], batch, mode="advanced", keyword_weight=weight)
    for c in run.candidates:
        expected = c.keyword_score if weight == 1.0 else c.semantic_score
        assert c.hybrid_score == pytest.approx(expected, abs=1e-6)


def test_invalid_inputs(engine, batch):
    with pytest.raises(ValueError, match="too short"):
        engine.screen("Chef", batch)
    with pytest.raises(ValueError, match="at least one"):
        engine.screen(SAMPLE_JOBS["Sous Chef"]["text"], [])
    with pytest.raises(ValueError, match="Unknown mode"):
        engine.screen(SAMPLE_JOBS["Sous Chef"]["text"], batch, mode="turbo")


def test_embeddings_cover_whole_document(engine):
    head = "Experienced pastry chef specialising in French desserts and bakery operations. " * 12
    tail = "Later moved into software engineering, building Python microservices on Kubernetes and AWS. " * 25
    e_head, e_full = engine.embed([head, head + tail])
    assert e_head.shape == (engine.pool_embeddings.shape[1],)
    assert np.linalg.norm(e_full) == pytest.approx(1.0, abs=1e-5)
    # With truncation at 256 word-pieces both would be identical; chunk pooling sees the tail.
    assert float(e_head @ e_full) < 0.99
