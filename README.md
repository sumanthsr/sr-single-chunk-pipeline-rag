# Single Chunk Strategy RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for querying biomedical research papers via a Streamlit chat interface. Built to be minimal, reproducible, and easy to run on any machine with Docker.

---

## What This Does

With PDFs stored in a directory, run one command, and get a local chat interface where you can ask questions across all documents simultaneously. Answers are grounded in the actual content with cited sources showing which paper, page, and section each claim came from.

---

## Stack

| Component | Technology | Why |
|---|---|---|
| PDF parsing | pdfplumber → PyMuPDF → Tesseract | Cascading fallback handles digital, hybrid, and scanned PDFs |
| Chunking | Custom RecursiveTextSplitter | 350 token target, 35 token overlap, never crosses section boundaries |
| Embedding | `pritamdeka/S-PubMedBert-MS-MARCO` | Pretrained on PubMed, fine-tuned for retrieval — understands biomedical terminology |
| Vector store | ChromaDB (embedded) | No separate container, persists to disk, cross-PDF search implicit |
| LLM | Ollama (`phi3`) | Fully local, no API key, CPU-only inference |
| UI | Streamlit | Python-native, fastest to iterate |
| Containers | 2 (streamlit + ollama) | Minimal by design |
| Scaling | Nginx (optional profile) | Add replicas with one flag when needed |

---

## Requirements

- Docker Desktop with Compose v2
- **Windows users:** WSL2 with at least 8 GB allocated to Docker Desktop (Settings → Resources → Memory)
- 10 GB free disk space (model weights + image layers)
- No GPU required — CPU inference works

---

## Project Structure

```
sr-single-chunk-pipeline-rag/
├── app/
│   ├── ui.py                    # Streamlit chat interface + ingestion trigger
│   ├── pipeline/
│   │   ├── parser.py            # Stage 1: PDF → structured elements
│   │   ├── detector.py          # Stage 2: IMRaD section labelling
│   │   ├── chunker.py           # Stage 3: Recursive text splitting
│   │   ├── enricher.py          # Stage 4: Quality filter + MinHash dedup
│   │   ├── embedder.py          # Stage 5: PubMedBERT embedding (shared instance)
│   │   ├── vector_store.py      # ChromaDB read/write
│   │   └── ingestor.py          # Pipeline orchestrator
│   ├── retrieval/
│   │   └── retriever.py         # Query embed + ChromaDB search
│   ├── llm/
│   │   └── ollama_client.py     # Prompt building + streaming
│   ├── observability/
│   │   └── tracer.py            # No-op stubs, ready to wire a provider
│   └── utils/
│       ├── config_loader.py     # config.yml loader with env overrides
│       └── logger.py            # Standard Python logging
├── config/
│   └── config.yml               # Single source of truth — all parameters here
├── infra/
│   └── nginx/
│       └── nginx.conf           # Load balancer (scaled profile only)
├── pdfs/                        # Drop your PDFs here (bind mount :ro)
├── chroma_db/                   # ChromaDB persistence (bind mount)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/sumanthsr/sr-single-chunk-pipeline-rag.git
cd sr-single-chunk-pipeline-rag.git
```

### 2. Drop PDFs into the `./pdfs` folder

```bash
cp /path/to/your/papers/*.pdf ./pdfs/
```

### 3. Start the stack

```bash
docker compose up --build
```

What happens on first run:
- Ollama starts and waits 15 seconds before pulling `tinyllama` (~600 MB)
- Streamlit builds and starts
- PubMedBERT model weights download (~440 MB, cached to a Docker volume)
- Ingestion pipeline runs automatically: parse → chunk → enrich → embed → store
- Ingestion time: approximately 2–5 minutes for 6 papers on CPU

### 4. Open the chat interface

```
http://localhost:8501
```

Type any question. Answers stream token by token with source citations below.

---

## Execution Flow

```
Startup
  │
  ├── ChromaDB already populated?
  │     YES → skip ingestion, go straight to chat UI
  │     NO  → run ingestion pipeline:
  │
  │   For each PDF in ./pdfs:
  │     Stage 1: Parse        pdfplumber → PyMuPDF → Tesseract OCR
  │     Stage 2: Detect       Label elements with IMRaD section (abstract,
  │                           introduction, methods, results, discussion,
  │                           conclusion) — references and supplementary dropped
  │     Stage 3: Chunk        RecursiveTextSplitter, 350 tok target, 35 overlap
  │                           Self-contained prefix: [title | section]: text
  │     Stage 4: Enrich       Quality score filter, MinHash dedup (Jaccard 0.85)
  │     Stage 5: Embed        PubMedBERT in-process, batch_size=32, normalised
  │     Stage 6: Store        ChromaDB upsert — all PDFs in one collection
  │
  └── Chat UI ready
        │
        User types query
        │
        A: Embed query        Same PubMedBERT instance — no drift possible
        B: Retrieve           ChromaDB cosine search, top_k=5, cross-PDF implicit
        C: Build prompt       System prompt + ranked context chunks + question
        D: Stream answer      Ollama /api/generate, tokens rendered as they arrive
        E: Show sources       Expandable panel: PDF name, page, section, score
```

---

## Configuration

Everything is in `config/config.yml`. 

This file is bind-mounted read-only into the container — **changes take effect with a restart, no rebuild needed.**

### Key parameters

| Section | Key | Default | Notes |
|---|---|---|---|
| `chunking` | `target_tokens` | `350` | Sized to fit PubMedBERT's 512 token context |
| `chunking` | `overlap_tokens` | `35` | ~10% overlap between adjacent chunks |
| `retrieval` | `top_k` | `5` | Chunks retrieved per query |
| `retrieval` | `score_threshold` | `0.0` | Raise to filter low-confidence results |
| `llm` | `model` | `phi3` | Ollama model name |
| `llm` | `temperature` | `0.1` | Low = factual, high = creative |
| `llm` | `timeout_seconds` | `300` | Increase if generation times out |
| `ui` | `show_sources` | `true` | Show retrieved chunks below answer |
| `ui` | `show_scores` | `true` | Show cosine similarity scores |

### Environment variable overrides

Any config.yml value can be overridden without editing the file:

```bash
RAG__LLM__MODEL=phi3 docker compose up
RAG__RETRIEVAL__TOP_K=10 docker compose up
RAG__CHUNKING__TARGET_TOKENS=300 docker compose up
```

Pattern: `RAG__<SECTION>__<KEY>=value`

---

## Changing the LLM Model

### Step 1 — Edit `config/config.yml`

```yaml
llm:
  model: "phi3"   # change this
```

### Step 2 — Edit `docker-compose.yml` ollama-init section

```yaml
ollama pull phi3   # must match config.yml
```

### Step 3 — Wipe old model volume and restart

```bash
docker compose down -v        # removes ollama_models volume with old model
docker compose up
```

No rebuild needed. The `-v` flag is important — without it Ollama keeps the old model cached and does not pull the new one.

### Available models (CPU-compatible)

| Model | RAM needed | Speed | Quality |
|---|---|---|---|
| `tinyllama` | ~800 MB | Fast (~10–30s) | Basic |
| `qwen2:1.5b` | ~1 GB | Fast (~20–40s) | Decent |
| `gemma:2b` | ~1.5 GB | Moderate | Decent |
| `phi3` | ~2.3 GB | Slow (~2–4 min) | Good |
| `mistral:7b-instruct-q4_0` | ~4 GB | Very slow | Better |

---

## Re-indexing

When you add new PDFs or change chunking parameters, the existing ChromaDB index must be wiped and rebuilt:

```bash
# Wipe index
rm -rf chroma_db/*
touch chroma_db/.gitkeep

# Restart to trigger re-ingestion
docker compose restart streamlit
```

## Scaling

Default deployment runs one Streamlit container. For high traffic:

```bash
docker compose --profile scaled up --scale streamlit=3
```

This starts Nginx on port 80 as a `least_conn` load balancer in front of 3 Streamlit replicas. Ollama and ChromaDB are shared. Each replica loads PubMedBERT (~440 MB) independently — plan ~1 GB RAM per additional replica.

---

## When to Rebuild vs Restart

| Change made | Command needed |
|---|---|
| `config/config.yml` | `docker compose restart streamlit` |
| `docker-compose.yml` env vars | `docker compose restart streamlit` |
| Any `.py` file in `app/` | `docker compose up --build` |
| `requirements.txt` | `docker compose up --build` |
| `Dockerfile` | `docker compose up --build` |
| Changed LLM model | `docker compose down -v && docker compose up` |
| Added new PDFs | `bash scripts/reindex.sh && docker compose restart streamlit` |

---

## Full Reset

To wipe everything and start completely fresh:

```bash
docker compose down -v
docker system prune -af
rm -rf chroma_db/*
touch chroma_db/.gitkeep
docker compose up --build
```

---

## Observability (Future)

Hooks are wired at retrieval and generation time in `app/observability/tracer.py`. Currently no-ops. To activate a provider:

1. Set in `config/config.yml`:
```yaml
observability:
  enabled: true
  provider: "langfuse"
  langfuse:
    public_key: "pk-..."
    secret_key: "sk-..."
    host: "https://cloud.langfuse.com"
```

2. Uncomment the provider package in `requirements.txt`

3. Rebuild: `docker compose up --build`

Supported providers: `langfuse`, `langsmith`, `otel`. Adding a new one requires only implementing `log_retrieval()` and `log_generation()` in `tracer.py`.

---

## Troubleshooting

**Port 8501 connection refused on Windows**
WSL2 port forwarding sometimes breaks. Try:
```powershell
# Get WSL2 IP
wsl hostname -I
# Open http://<that-ip>:8501 directly
# Or restart Docker Desktop from the system tray
```

**Ollama model not loading / timeout**
```bash
docker logs <ollama-container-name> --tail 30
# If model not pulled:
bash scripts/pull_model.sh
```

**Ingestion failed**
```bash
docker logs <streamlit-container-name> --tail 50
# Wipe and retry:
bash scripts/reindex.sh
docker compose restart streamlit
```

**Out of memory**
- Reduce Streamlit replicas if using scaled profile
- Switch to a smaller model (`tinyllama` needs only ~800 MB)
- On Windows: increase Docker Desktop memory in Settings → Resources