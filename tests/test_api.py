import numpy as np
import pytest
import spacy
from fastapi.testclient import TestClient

from backend.api.auth import get_current_user
from backend.main import app
from tests.test_parser import make_docx, make_pdf

PARSED = {
    "name": "Jane", "email": "j@x.com", "phone": "123", "location": "", "summary": "Dev",
    "professional_summary": "Python developer", "linkedin": "", "github": "", "portfolio": "",
    "skills": ["Python", "FastAPI", "Docker"], "keywords": ["python", "api"],
    "action_verbs": ["built", "designed"],
    "education": [{"degree": "BSc"}],
    "experience": [{"job_title": "Dev", "company": "X", "description": "Built Python APIs", "duration_months": 24}],
    "projects": [{"title": "API", "description": "FastAPI service"}],
    "certifications": [],
}


class FakeEmbedder:
    def encode(self, texts, **kw):
        def vec(t):
            rng = np.random.default_rng(abs(hash(t)) % (2**32)); return rng.random(8)
        if isinstance(texts, str):
            return vec(texts)
        return np.array([vec(t) for t in texts])


@pytest.fixture()
def client(monkeypatch):
    app.state.nlp = spacy.load("en_core_web_sm")
    app.state.embedder = FakeEmbedder()
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    import backend.services.resume_analyzer as ra
    monkeypatch.setattr(ra, "parse_resume", lambda text: dict(PARSED))
    monkeypatch.setattr(ra, "parse_job_description", lambda t: {
        "required_skills": ["python", "kubernetes"], "preferred_skills": [], "keywords": ["api"]})
    saved = []
    async def fake_save(uid, fn, res): saved.append(fn); return "1"
    import backend.database.supabase_db as db
    monkeypatch.setattr(db, "save_analysis", fake_save)
    c = TestClient(app); c.saved = saved
    yield c
    app.dependency_overrides.clear()


def test_health_single_route_and_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"
    paths = [r.path for r in app.routes if getattr(r, "path", "") == "/api/v1/health"]
    assert len(paths) == 1                      # duplicate route removed


def test_openapi_has_unique_operation_ids(client):
    ids = [op["operationId"] for p in client.get("/openapi.json").json()["paths"].values() for op in p.values()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name,data,mime", [
    ("r.pdf", make_pdf(), "application/pdf"),
    ("r.docx", make_docx(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("r.txt", b"Jane Doe\nPython FastAPI developer", "text/plain"),
], ids=["pdf", "docx", "txt"])
def test_analyze_all_formats(client, name, data, mime):
    r = client.post("/api/v1/analyze-resume", files={"resume": (name, data, mime)}, data={"job_description": ""})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["ATS_score"] <= 100 and body["component_scores"]
    assert name in client.saved                   # history saved in background


def test_analyze_with_jd(client):
    r = client.post("/api/v1/analyze-resume", files={"resume": ("r.txt", b"Python dev", "text/plain")},
                    data={"job_description": "Need Python and Kubernetes"})
    assert r.status_code == 200, r.text
    assert r.json()["jd_comparison"] is not None


def test_empty_file_400(client):
    r = client.post("/api/v1/analyze-resume", files={"resume": ("r.pdf", b"", "application/pdf")})
    assert r.status_code == 400 and "empty" in r.json()["detail"].lower()


def test_garbage_400(client):
    r = client.post("/api/v1/analyze-resume", files={"resume": ("r.pdf", b"\x00\x01zzz", "application/pdf")})
    assert r.status_code == 400


def test_supabase_failure_does_not_break_analysis(client, monkeypatch):
    import backend.database.supabase_db as db
    async def boom(*a): raise RuntimeError("db down")
    monkeypatch.setattr(db, "save_analysis", boom)
    r = client.post("/api/v1/analyze-resume", files={"resume": ("r.txt", b"Python dev", "text/plain")})
    assert r.status_code == 200


def test_requires_auth():
    app.dependency_overrides.clear()
    r = TestClient(app).post("/api/v1/analyze-resume", files={"resume": ("r.txt", b"x", "text/plain")})
    assert r.status_code == 401


def test_generate_pdf_roundtrip(client, tmp_path):
    """Analyse a resume, send the JSON to /generate-pdf, and check the PDF really opens."""
    r = client.post("/api/v1/analyze-resume",
                    files={"resume": ("r.txt", b"Jane Doe\nPython FastAPI developer", "text/plain")},
                    data={"job_description": "Need Python and Kubernetes"})
    assert r.status_code == 200, r.text
    pdf = client.post("/api/v1/generate-pdf", json=r.json())
    assert pdf.status_code == 200, pdf.text
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF-")
    out = tmp_path / "report.pdf"; out.write_bytes(pdf.content)
    from PyPDF2 import PdfReader
    reader = PdfReader(str(out))
    assert len(reader.pages) >= 2
    assert "ATS" in "".join(p.extract_text() or "" for p in reader.pages)


def test_onnx_embedder_adapter(monkeypatch):
    """The light embedder must behave like SentenceTransformer.encode for our two call patterns."""
    import sys, types
    class FakeTE:
        def __init__(self, **kw): self.kw = kw
        def embed(self, docs):
            for d in docs: yield np.arange(384, dtype=np.float32) + len(d)
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTE))
    from backend.services.onnx_embedder import OnnxEmbedder
    e = OnnxEmbedder("all-MiniLM-L6-v2")
    assert e._model.kw["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert e.encode("one text", convert_to_tensor=False).shape == (384,)
    assert e.encode(["a", "bb"], convert_to_tensor=False, show_progress_bar=False).shape == (2, 384)


def test_analysis_works_with_onnx_style_embedder(client):
    """Scoring code must work with an embedder that only offers encode() returning numpy arrays."""
    class OnnxLike:
        def encode(self, texts, convert_to_tensor=False, show_progress_bar=False, **kw):
            rng = np.random.default_rng(1)
            return rng.random(384) if isinstance(texts, str) else rng.random((len(texts), 384))
    app.state.embedder = OnnxLike()
    r = client.post("/api/v1/analyze-resume",
                    files={"resume": ("r.txt", b"Python dev", "text/plain")},
                    data={"job_description": "Need Python"})
    assert r.status_code == 200, r.text
