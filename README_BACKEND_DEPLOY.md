# Light backend (no PyTorch) - for free ~512 MB hosts such as Render free

Differences from the normal backend:
* `fastembed` (ONNX Runtime) runs the same `all-MiniLM-L6-v2` model instead of torch + sentence-transformers.
* spaCy `en_core_web_sm` instead of `en_core_web_md`.
Both are chosen by environment variables set in the Dockerfile (`EMBEDDER_BACKEND=onnx`, `SPACY_MODEL=en_core_web_sm`).

**Be aware:** scores can differ slightly from the full build (smaller spaCy model, ONNX numeric differences),
and this has NOT been measured on a real 512 MB host. If it still runs out of memory, use the full build on a bigger host.
PDF export loads WeasyPrint on first use, so it is the step most likely to hit the memory limit.

## Deploy on Render (free)
1. Push this folder to a GitHub repo.
2. Render -> **New -> Web Service** -> connect the repo -> **Runtime: Docker** -> **Instance type: Free**.
3. **Environment** variables: `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY` (service_role), and `SUPABASE_JWT_SECRET` only if you use legacy HS256 tokens.
4. **Health Check Path:** `/api/v1/health`
5. Deploy. Watch **Logs**: you should see `ONNX embedder ready` and `All models loaded. API is ready.`
6. Open `https://<your-service>.onrender.com/api/v1/health` -> `"status":"healthy"`.
7. Streamlit Cloud -> Settings -> Secrets:
   ```toml
   [backend]
   url = "https://<your-service>.onrender.com"
   ```

Render free services sleep after ~15 minutes idle; the first request afterwards can take 30-60 s. Open the health URL once to wake it before using the app.

## Test locally
```powershell
docker build -t ats-light .
docker run --rm -p 8000:7860 --env-file .env ats-light
curl.exe http://127.0.0.1:8000/api/v1/health
```
Check memory while idle and during an analysis: `docker stats`. It should stay under 512 MB to fit Render free.

## Unit tests (no network)
```powershell
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
python -m pytest -q tests
```
