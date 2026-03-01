from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.yml"


class _DotDict:
    """
    Wraps a nested dict and provides attribute-style access.
    d["llm"]["model"]  becomes  d.llm.model
    """

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _DotDict(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"_DotDict({self.__dict__})"


def _apply_env_overrides(data: dict, prefix: str = "RAG") -> dict:
    """
    Walk environment variables prefixed with RAG__ and override
    matching keys in the config dict.
    Delimiter is double underscore: RAG__LLM__MODEL -> data["llm"]["model"]
    """
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(f"{prefix}__"):
            continue
        parts = env_key[len(prefix) + 2:].lower().split("__")
        node = data
        for part in parts[:-1]:
            if part not in node:
                break
            node = node[part]
        else:
            leaf = parts[-1]
            if leaf in node:
                # Preserve original type
                original = node[leaf]
                if isinstance(original, bool):
                    node[leaf] = env_val.lower() in ("true", "1", "yes")
                elif isinstance(original, int):
                    node[leaf] = int(env_val)
                elif isinstance(original, float):
                    node[leaf] = float(env_val)
                else:
                    node[leaf] = env_val
    return data


@lru_cache(maxsize=1)
def get_config() -> _DotDict:
    """
    Load and cache the configuration. Call this from anywhere:
        from app.utils.config_loader import get_config
        cfg = get_config()
        model_name = cfg.embedding.model_name
    """
    with open(_CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)
    raw = _apply_env_overrides(raw)
    return _DotDict(raw)
