#!/usr/bin/env bash
# scripts/reindex.sh
# Wipe ChromaDB and force a full re-ingestion on next startup.
# Use this when you add new PDFs or change chunking config.

set -euo pipefail

echo "Wiping ChromaDB..."
rm -rf chroma_db/*
touch chroma_db/.gitkeep
echo "ChromaDB cleared. Restart the stack to re-index:"
echo "  docker compose restart streamlit"
