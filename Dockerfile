# LIGHT build: aims to fit hosts with ~512 MB RAM (e.g. Render free).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 \
        libgdk-pixbuf-2.0-0 shared-mime-info fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDER_BACKEND=onnx \
    SPACY_MODEL=en_core_web_sm \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt \
 && python -m spacy download en_core_web_sm

# Bake the ONNX embedding model into the image (no download at start-up).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/home/user/.cache/fastembed')"

COPY --chown=user backend ./backend
RUN mkdir -p backend/logs

EXPOSE 7860
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
