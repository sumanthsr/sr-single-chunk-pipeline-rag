from __future__ import annotations

import json
from typing import Generator

import httpx

from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
cfg = get_config()


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = (
            f"[{i}] PDF: {chunk.get('pdf_filename', 'unknown')}, "
            f"Page: {chunk.get('page', '?')}, "
            f"Section: {chunk.get('section_label', '?')}"
        )
        context_parts.append(f"{source}\n{chunk.get('text', '')}")
    context = "\n\n---\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def stream_answer(query: str, chunks: list[dict]) -> Generator[str, None, None]:
    lcfg = cfg.llm
    prompt = build_prompt(query, chunks)
    payload = {
        "model": lcfg.model,
        "prompt": prompt,
        "system": lcfg.system_prompt,
        "stream": True,
        "options": {
            "temperature": lcfg.temperature,
            "num_predict": lcfg.max_tokens,
        },
    }
    url = f"{lcfg.base_url}/api/generate"
    try:
        with httpx.Client(timeout=lcfg.timeout_seconds) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except httpx.ConnectError:
        yield "\n\n[Error: Cannot connect to Ollama. Ensure the ollama service is running.]"
    except Exception as exc:
        logger.error(f"Ollama stream failed: {exc}")
        yield f"\n\n[Error: {exc}]"


def check_ollama_health() -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{cfg.llm.base_url}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return any(cfg.llm.model in m for m in models)
    except Exception:
        return False
