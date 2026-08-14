"""
utils.py — Shared text processing and file extraction utilities.
"""

import string
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def ensure_nltk_resources():
    """Download required NLTK data files if they are not already present."""
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# Run once on import
ensure_nltk_resources()

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def advanced_clean_text(text: str) -> str:
    """Lowercase, strip punctuation, remove stop-words, and lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    cleaned = [_lemmatizer.lemmatize(w) for w in tokens if w not in _stop_words]
    return " ".join(cleaned)


def extract_text(file_name: str, file_bytes: bytes) -> str:
    """Extract plain text from a PDF, DOCX, or TXT file given its bytes."""
    text = ""
    try:
        if file_name.lower().endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "

        elif file_name.lower().endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            text = " ".join(p.text for p in doc.paragraphs)

        elif file_name.lower().endswith(".txt"):
            text = file_bytes.decode("utf-8")
    except Exception as e:
        print(f"Error reading file '{file_name}': {e}")

    return text.strip()
