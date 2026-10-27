"""FastAPI application entry point: uvicorn backend.main:app"""

import logging
import os
import sys
from contextlib import asynccontextmanager

# Import config FIRST: it loads .env before anything reads os.environ.
from backend.core.config import (
    ALLOWED_ORIGINS,
    EMBEDDER_BACKEND,
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    SENTENCE_TRANSFORMER_MODEL,
    SPACY_MODEL_PRIMARY,
    SPACY_MODEL_SECONDARY,
)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

from backend.api.routes import router  # noqa: E402

logger = logging.getLogger("ats_resume_scorer")


# ============================================================
# LOAD SPACY NLP MODEL
# ============================================================

def _load_nlp():
    import spacy

    for name in (SPACY_MODEL_PRIMARY, SPACY_MODEL_SECONDARY):
        try:
            nlp = spacy.load(name)
            logger.info("Loaded spaCy model: %s", name)
            return nlp

        except OSError:
            logger.warning(
                "spaCy model '%s' not installed.",
                name,
            )

    raise RuntimeError(
        f"No spaCy model found. "
        f"Run: python -m spacy download {SPACY_MODEL_PRIMARY}"
    )


# ============================================================
# APPLICATION STARTUP / SHUTDOWN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting ATS Resume Analyzer API...")

    logger.info(
        "Groq API key configured: %s",
        bool(os.getenv("GROQ_API_KEY")),
    )

    logger.info(
        "Supabase configured: %s",
        bool(os.getenv("SUPABASE_URL")),
    )

    # Import heavy modules during startup.
    import backend.services.resume_analyzer  # noqa: F401
    import backend.services.resume_parser  # noqa: F401

    # Load spaCy.
    app.state.nlp = _load_nlp()

    # Load embedding model.
    logger.info(
        "Loading embedder (%s): %s",
        EMBEDDER_BACKEND,
        SENTENCE_TRANSFORMER_MODEL,
    )

    if EMBEDDER_BACKEND == "onnx":

        from backend.services.onnx_embedder import OnnxEmbedder

        app.state.embedder = OnnxEmbedder(
            SENTENCE_TRANSFORMER_MODEL,
            cache_dir=os.getenv("FASTEMBED_CACHE_PATH") or None,
        )

    else:

        from sentence_transformers import SentenceTransformer

        app.state.embedder = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL
        )

    # Warm up embedding model.
    app.state.embedder.encode(
        ["warm-up"],
        show_progress_bar=False,
    )

    logger.info("All models loaded. API is ready.")

    try:
        yield

    finally:
        logger.info("Shutting down ATS Resume Analyzer API.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/", tags=["General"])
async def root():

    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/api/v1/health", tags=["General"])
async def health_check(request: Request):

    nlp_ok = (
        getattr(request.app.state, "nlp", None)
        is not None
    )

    emb_ok = (
        getattr(request.app.state, "embedder", None)
        is not None
    )

    return {
        "status": (
            "healthy"
            if (nlp_ok and emb_ok)
            else "not_ready"
        ),
        "nlp_model_loaded": nlp_ok,
        "embedding_model_loaded": emb_ok,
        "groq_api_key_configured": bool(
            os.getenv("GROQ_API_KEY")
        ),
        "supabase_configured": bool(
            os.getenv("SUPABASE_URL")
        ),
    }


# ============================================================
# REGISTER API ROUTES
# ============================================================

app.include_router(router)


# ============================================================
# SWAGGER BEARER AUTHENTICATION CONFIGURATION
# ============================================================

def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Define Bearer authentication scheme.
    openapi_schema.setdefault(
        "components", {}
    ).setdefault(
        "securitySchemes", {}
    )["HTTPBearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Enter your Supabase access token.",
    }

    # Protected API endpoints.
    protected_paths = [
        "/api/v1/analyze-resume",
        "/api/v1/history",
        "/api/v1/history/{analysis_id}",
        "/api/v1/generate-pdf",
        "/api/v1/history/{analysis_id}/pdf",
    ]

    # Apply security to protected endpoints.
    for path in protected_paths:

        if path not in openapi_schema["paths"]:
            continue

        for method, operation in openapi_schema["paths"][path].items():

            if method.lower() not in {
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "options",
                "head",
                "trace",
            }:
                continue

            operation["security"] = [
                {"HTTPBearer": []}
            ]

    app.openapi_schema = openapi_schema

    return app.openapi_schema


app.openapi = custom_openapi


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )