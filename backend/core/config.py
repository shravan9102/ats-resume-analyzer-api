"""Central, environment-based configuration. No secrets are stored in code."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # Project-root .env is the supported location; backend/.env is accepted
    # as a fallback so older setups keep working. Real env vars always win.
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(_PROJECT_ROOT / "backend" / ".env")
except ImportError:  # python-dotenv is in requirements.txt; this is just a guard
    pass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# API metadata
APP_TITLE = "ATS RESUME ANALYZER API"
APP_VERSION = "1.1.0"
APP_DESCRIPTION = "Analyse resumes against job descriptions using NLP and ML."

# CORS (comma-separated list in ALLOWED_ORIGINS overrides the defaults)
_DEFAULT_ORIGINS = [
    "https://appapppy-ktwxupi73vqhjzweksze9d.streamlit.app",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()
] or _DEFAULT_ORIGINS

# File handling
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_PDF_PAGES = 15

SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "text/plain": "txt",
}
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

# Timeouts (seconds). Keep: GROQ < ANALYSIS < frontend read timeout.
PARSE_TIMEOUT_SECONDS = _int_env("PARSE_TIMEOUT_SECONDS", 30)
ANALYSIS_TIMEOUT_SECONDS = _int_env("ANALYSIS_TIMEOUT_SECONDS", 120)
HISTORY_SAVE_TIMEOUT_SECONDS = _int_env("HISTORY_SAVE_TIMEOUT_SECONDS", 10)

# spaCy / embeddings
SPACY_MODEL_PRIMARY = os.getenv("SPACY_MODEL", "en_core_web_md")
SPACY_MODEL_SECONDARY = "en_core_web_sm"
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
# "sentence-transformers" (default, needs torch) or "onnx" (light build, no torch)
EMBEDDER_BACKEND = os.getenv("EMBEDDER_BACKEND", "sentence-transformers").strip().lower()

# Score weights
SCORE_WEIGHTS = {
    "formatting": 20,
    "keywords": 25,
    "content": 25,
    "skill_validation": 15,
    "ats_compatibility": 15,
}
JD_KEYWORD_WEIGHT = 0.6
JD_SEMANTIC_WEIGHT = 0.4

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")            # service_role key (backend only)
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TIMEOUT_SECONDS = _float_env("GROQ_TIMEOUT_SECONDS", 40.0)
