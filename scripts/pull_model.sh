#!/usr/bin/env bash
# scripts/pull_model.sh
# Manually pull the Ollama model configured in config/config.yml.
# Run this if ollama-init failed or you want to pre-pull before first startup.

set -euo pipefail

MODEL=$(python3 -c "
import yaml, sys
with open('config/config.yml') as f:
    cfg = yaml.safe_load(f)
print(cfg['llm']['model'])
")

echo "Pulling model: $MODEL"
docker compose exec ollama ollama pull "$MODEL"
echo "Done."
