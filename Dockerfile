FROM python:3.11-slim

# System deps: Tesseract + Poppler (pdf2image) + curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU-only FIRST from PyTorch's own index.
# This must happen before requirements.txt is processed — if sentence-transformers
# pulls torch as a transitive dep from the default PyPI index it gets the full
# CUDA+ROCm wheel (~2.5 GB with nvidia libraries). The CPU wheel is ~180 MB.
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch is already satisfied, pip will not reinstall)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY config/ ./config/

# Create directories for bind mounts
RUN mkdir -p /app/pdfs /app/chroma_db /app/.model_cache

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Streamlit config via environment — no ~/.streamlit/config.toml needed
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["streamlit", "run", "app/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]