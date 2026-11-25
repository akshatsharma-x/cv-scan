"""Headless UI tests with Streamlit's AppTest."""

import os

import pytest
from streamlit.testing.v1 import AppTest

from conftest import needs_artifacts

pytestmark = needs_artifacts
APP = os.path.join(os.path.dirname(__file__), "..", "src", "app.py")


def _app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _no_errors(at):
    assert not at.exception, [e.value for e in at.exception]


def test_home_renders_with_navigation_buttons():
    at = _app()
    assert {"cta_start", "cta_eval", "go_basic", "go_advanced"} <= {b.key for b in at.button}


def test_full_screening_flow_both_modes():
    at = _app()
    at.button(key="go_basic").click().run()
    _no_errors(at)
    assert at.session_state["mode"] == "basic"

    at.selectbox(key="sample_job").select("Sous Chef").run()
    assert "Sous Chef" in at.text_area(key="jd_text").value
    at.button(key="add_samples").click().run()
    assert len(at.session_state["samples"]) == 6

    at.button(key="cta_run").click().run()
    _no_errors(at)
    run = at.session_state["run"]
    assert run.mode == "basic" and len(run.candidates) == 6
    assert run.candidates[0].name.startswith("Chef")
    assert [t.label for t in at.tabs] == ["Keyword insights", "Pool benchmark", "Resume text"]

    at.segmented_control(key="mode").set_value("advanced").run()
    at.button(key="cta_run").click().run()
    _no_errors(at)
    run = at.session_state["run"]
    assert run.mode == "advanced"
    assert "Semantic map" in [t.label for t in at.tabs]

    at.segmented_control(key="map_dims").set_value("3D").run()
    _no_errors(at)
    at.selectbox(key="selected").select(run.candidates[-1].name).run()
    _no_errors(at)


def test_validation_messages():
    at = _app()
    at.button(key="cta_start").click().run()
    at.button(key="cta_run").click().run()
    _no_errors(at)
    assert any("job description" in w.value.lower() for w in at.warning)


def test_evaluation_page_controls():
    at = _app()
    at.button(key="cta_eval").click().run()
    _no_errors(at)
    if not at.segmented_control:
        pytest.skip("evaluation results not generated")
    at.pills(key="curve_metric").set_value("MCC").run()
    _no_errors(at)
    for key, value in (("cm_method", "BERT"), ("sweep_method", "TF-IDF")):
        at.segmented_control(key=key).set_value(value).run()
        _no_errors(at)


def test_batch_item_removal_and_clear_all():
    at = _app()
    at.button(key="cta_start").click().run()
    at.button(key="add_samples").click().run()
    samples = at.session_state["samples"]
    assert len(samples) == 6
    at.button(key=f"rm_s_{samples[0]['id']}").click().run()
    _no_errors(at)
    assert len(at.session_state["samples"]) == 5
    nonce = at.session_state["uploader_nonce"]
    at.button(key="clear_candidates").click().run()
    _no_errors(at)
    assert at.session_state["samples"] == [] and at.session_state["uploads"] == {}
    assert at.session_state["uploader_nonce"] == nonce + 1  # fresh, empty uploader widget
