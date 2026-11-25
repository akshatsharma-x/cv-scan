"""
utils.py — Text extraction, cleaning and chunking utilities.

Shared by the training script, the evaluation script and the Streamlit app so
that every component processes text in exactly the same way.

Two cleaning paths are provided because the two engines need different input:

* ``clean_text``             → keyword engine (TF-IDF). Aggressive normalisation:
                               lowercase, URL / e-mail / non-ASCII removal,
                               punctuation → space, stop-word removal and
                               WordNet lemmatisation.
* ``prepare_semantic_text``  → semantic engine (BERT). Light normalisation only:
                               transformers rely on natural word order and on
                               function words ("not", "without", ...), so these
                               are preserved.
"""

from __future__ import annotations

import io
import os
import re
import ssl
import unicodedata
from functools import lru_cache

import nltk

# ── NLTK resources ─────────────────────────────────────────────────────────
_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}


def _download_nltk(resource: str) -> None:
    """Download an NLTK resource, retrying once without certificate checks.

    Some macOS Python installs ship without root certificates. Instead of
    disabling TLS verification for the whole process (which would also affect
    the Hugging Face model download), the fallback is scoped to this call.
    """
    try:
        nltk.download(resource, quiet=True, raise_on_error=True)
        return
    except Exception:
        pass
    original_context = ssl._create_default_https_context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        nltk.download(resource, quiet=True, raise_on_error=True)
    finally:
        ssl._create_default_https_context = original_context


def ensure_nltk_resources() -> None:
    """Make sure the NLTK corpora used by the cleaner are available locally."""
    for resource, path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            _download_nltk(resource)


# Generic recruiting boilerplate and placeholders used by the anonymised
# dataset ("Company Name", "City", "State"). They carry no skill information
# and would otherwise show up as "missing keywords" in the explanations.
DOMAIN_STOPWORDS = frozenset(
    """
    company name city state candidate candidates looking seeking ideal required
    require requirement requirements preferred prefer plus ability abilities
    including include includes responsibilities responsibility responsible job
    role position join opportunity opportunities work working worked year years
    month months must will etc within well new using use used also may skill
    skills experience experienced knowledge strong excellent good great highly
    successful successfully various multiple one two three summary highlights
    objective profile current currently present date level ensure ensuring
    duties duty description apply applicant applicants qualification
    qualifications wanted needed hiring hire alongside smoothly running keep busy
    accordance seek seeks sought
    """.split()
)


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    ensure_nltk_resources()
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english")) | DOMAIN_STOPWORDS


@lru_cache(maxsize=1)
def _lemmatizer():
    ensure_nltk_resources()
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def _lemmatize(token: str) -> str:
    return _lemmatizer().lemmatize(token)


# ── Cleaning ───────────────────────────────────────────────────────────────
URL_RE = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# Alphanumeric tokens; keeps programming-language names such as c++, c#, f#.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\+\+|#)?")
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z]+")


def _to_ascii(text: str) -> str:
    """Fold accents (résumé → resume) and drop remaining non-ASCII symbols."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def tokenize(text: str) -> list[str]:
    """Return normalised keyword tokens (stop-words removed, lemmatised)."""
    if not isinstance(text, str) or not text:
        return []
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = _to_ascii(text).lower()

    stop = _stopwords()
    tokens = []
    for token in TOKEN_RE.findall(text):
        if token.isdigit() or len(token) < 2 or token in stop:
            continue
        lemma = _lemmatize(token)
        if len(lemma) < 2 or lemma in stop:
            continue
        tokens.append(lemma)
    return tokens


def clean_text(text: str) -> str:
    """Keyword-level normalisation used by the TF-IDF engine."""
    return " ".join(tokenize(text))


# Backwards-compatible name used by earlier versions of the project.
advanced_clean_text = clean_text


def prepare_semantic_text(text: str) -> str:
    """Light normalisation for the transformer (keeps natural language)."""
    if not isinstance(text, str) or not text:
        return ""
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def chunk_words(text: str, size: int = 150) -> list[str]:
    """Split text into consecutive windows of ``size`` words.

    all-MiniLM-L6-v2 truncates its input at 256 word-pieces (~190 words), so
    long resumes are embedded chunk-by-chunk and the chunk vectors are pooled.
    150 words keeps ~99% of chunks inside the model's window.
    """
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0


_MINOR_WORDS = frozenset({"and", "of", "for", "the", "to", "in", "at", "or", "a", "an", "with"})


def _smart_case(word: str) -> str:
    # Lower-case connectives, keep short acronyms (HR, IT, BPO) and roman numerals.
    if word.lower() in _MINOR_WORDS:
        return word.lower()
    return word if len(word) <= 3 and word.isupper() else word.capitalize()


def resume_title(text: str, max_len: int = 60) -> str:
    """Best-effort job title: the leading upper-case words of the first line.

    Resumes in the dataset start with the title in capitals, often followed by
    the summary on the same line ("HR GENERALIST Professional Summary …").
    Implemented as a linear token scan (no backtracking regex).
    """
    first = next((line for line in str(text).splitlines() if line.strip()), "")
    tokens = first.split()
    title_tokens = []
    for token in tokens:
        if not any(ch.isalpha() for ch in token) or token != token.upper():
            break
        title_tokens.append(token)
    title = " ".join(title_tokens) if title_tokens else " ".join(tokens)
    title = WORD_RE.sub(lambda m: _smart_case(m.group(0)), title.strip(" ,/-")[:max_len].strip())
    return title[:1].upper() + title[1:]


# ── File extraction ────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")


class ExtractionError(ValueError):
    """Raised when no usable text can be extracted from an uploaded file."""


def _extract_pdf(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - legacy fallback
        from PyPDF2 import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ExtractionError("The PDF is password-protected.") from exc
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(file_bytes))
    parts = [p.text for p in document.paragraphs]
    # Many resume templates lay out content in tables.
    for table in document.tables:
        for row in table.rows:
            seen = set()
            for cell in row.cells:
                # Merged cells are returned once per spanned column.
                if cell._tc in seen:
                    continue
                seen.add(cell._tc)
                parts.append(cell.text)
    return "\n".join(p for p in parts if p and p.strip())


def _decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("latin-1", errors="replace")


def extract_text(file_name: str, file_bytes: bytes) -> str:
    """Extract plain text from a PDF, DOCX or TXT file.

    Raises ``ExtractionError`` with a user-friendly message when the file type
    is unsupported, the file is corrupt, or it contains no extractable text
    (e.g. a scanned, image-only PDF).
    """
    extension = os.path.splitext(file_name.lower())[1]
    if extension not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(
            f"Unsupported file type '{extension or file_name}'. Use PDF, DOCX or TXT."
        )
    try:
        if extension == ".pdf":
            text = _extract_pdf(file_bytes)
        elif extension == ".docx":
            text = _extract_docx(file_bytes)
        else:
            text = _decode_text(file_bytes)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Could not read '{file_name}': the file appears to be corrupt.") from exc

    text = text.replace("\x00", " ").strip()
    if not text:
        hint = " It may be a scanned image without a text layer." if extension == ".pdf" else ""
        raise ExtractionError(f"No text could be extracted from '{file_name}'.{hint}")
    return text
