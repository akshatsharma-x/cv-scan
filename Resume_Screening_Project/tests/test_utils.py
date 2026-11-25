import glob
import io
import os

import pytest

from utils import ExtractionError, chunk_words, clean_text, extract_text, prepare_semantic_text, tokenize

DATASET_PDFS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "data", "data", "*", "*.pdf")))


def test_punctuation_splits_instead_of_merging_words():
    # The previous cleaner deleted punctuation, producing tokens like "pythonjavasql".
    assert clean_text("Skills:Python,Java,SQL; HR/Payroll") == "python java sql hr payroll"


def test_urls_emails_numbers_and_accents_are_removed():
    text = "Contact jane.doe@mail.com or https://linkedin.com/in/jane — résumé 2019, café"
    tokens = tokenize(text)
    assert "resume" in tokens and "cafe" in tokens
    assert not any(t.isdigit() for t in tokens)
    assert not any("linkedin" in t or "mail" in t for t in tokens)


def test_programming_language_tokens_survive():
    assert {"c++", "c#"} <= set(tokenize("Expert in C++ and C# development"))


def test_stopwords_and_boilerplate_removed_and_lemmatised():
    tokens = tokenize("The candidates must have strong skills in managing budgets")
    assert "budget" in tokens
    assert not {"the", "candidate", "must", "strong", "skill"} & set(tokens)


def test_non_string_input():
    assert clean_text(None) == "" and prepare_semantic_text(None) == ""


def test_semantic_text_keeps_natural_language():
    out = prepare_semantic_text("I did   not\nwork at www.example.com on weekends")
    assert out == "I did not work at on weekends"


def test_chunking():
    sizes = [len(c.split()) for c in chunk_words(" ".join(["w"] * 400), size=150)]
    assert sizes == [150, 150, 100]
    assert chunk_words("") == []


def test_txt_encodings():
    assert extract_text("cv.txt", "Café – résumé".encode("utf-8")) == "Café – résumé"
    assert extract_text("cv.txt", "Café – résumé".encode("cp1252")) == "Café – résumé"


def test_docx_paragraphs_and_tables():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Jane Doe — Staff Accountant")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Skills"
    table.cell(0, 1).text = "QuickBooks, GAAP, reconciliations"
    buffer = io.BytesIO()
    document.save(buffer)
    text = extract_text("resume.docx", buffer.getvalue())
    assert "Staff Accountant" in text and "QuickBooks" in text


@pytest.mark.skipif(not DATASET_PDFS, reason="dataset PDFs not available")
def test_real_pdf_extraction():
    path = DATASET_PDFS[0]
    with open(path, "rb") as fh:
        text = extract_text(os.path.basename(path), fh.read())
    assert len(text.split()) > 100


@pytest.mark.parametrize(
    "name,data,message",
    [("cv.doc", b"abc", "Unsupported"), ("cv.pdf", b"not a pdf", "corrupt"), ("cv.txt", b"   ", "No text")],
)
def test_extraction_errors(name, data, message):
    with pytest.raises(ExtractionError, match=message):
        extract_text(name, data)


def test_resume_title_extraction():
    from utils import resume_title

    assert resume_title("HR GENERALIST Professional Summary Dependable") == "HR Generalist"
    assert resume_title("   PUBLIC RELATIONS SPECIALIST AND BRAND CONSULTANT\nSummary") == "Public Relations Specialist and Brand Consultant"
    assert resume_title("A" * 5000) == "A" + "a" * 59  # linear time, length-capped
