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

# Install Python dependencies before copying source
# (layer caching — only reinstalls when requirements change)
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
