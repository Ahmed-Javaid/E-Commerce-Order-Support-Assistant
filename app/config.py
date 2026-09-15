"""Central configuration.

Every knob the system needs is read from the environment once, at import time,
so that the rest of the codebase never touches ``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # --- LLM engine -----------------------------------------------------
    # "ollama" (default), "llamacpp", or "mock" (used by the test-suite).
    engine: str = field(default_factory=lambda: _env_str("NIMBUS_ENGINE", "ollama"))
    ollama_host: str = field(
        default_factory=lambda: _env_str("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    model: str = field(
        default_factory=lambda: _env_str("NIMBUS_MODEL", "qwen2.5:3b-instruct-q4_K_M")
    )
    # Path to a .gguf file, only used when engine == "llamacpp".
    gguf_path: str = field(default_factory=lambda: _env_str("NIMBUS_GGUF_PATH", ""))

    # --- Decoding -------------------------------------------------------
    temperature: float = field(
        default_factory=lambda: _env_float("NIMBUS_TEMPERATURE", 0.3)
    )
    top_p: float = field(default_factory=lambda: _env_float("NIMBUS_TOP_P", 0.9))
    max_output_tokens: int = field(
        default_factory=lambda: _env_int("NIMBUS_MAX_OUTPUT_TOKENS", 320)
    )
    # Hard ceiling passed to the runtime as num_ctx. Kept small on purpose:
    # prompt-eval on CPU is O(prompt tokens), so a 4k window keeps TTFT low.
    context_window: int = field(
        default_factory=lambda: _env_int("NIMBUS_CONTEXT_WINDOW", 4096)
    )
    # Number of CPU threads handed to the runtime. 0 => let the runtime decide.
    num_threads: int = field(default_factory=lambda: _env_int("NIMBUS_NUM_THREADS", 0))
    # The assignment requires CPU-only inference. Ollama offloads to a discrete
    # GPU automatically when it finds one, so we pin num_gpu=0 on every request
    # rather than relying on how the daemon happened to be started. Setting
    # NIMBUS_FORCE_CPU=0 lifts the pin (useful only for A/B measurement).
    force_cpu: bool = field(default_factory=lambda: _env_bool("NIMBUS_FORCE_CPU", True))

    # How the deterministic guard answers an off-domain or persona-override
    # message. The assignment requires that every response come from prompt
    # orchestration and conversational memory alone, so the default is "steer":
    # the guard still *detects* deterministically, but the refusal is written by
    # the model from an injected instruction. "reply" returns a canned string
    # without calling the model at all -- faster and perfectly consistent, but
    # the text is not model-generated. "off" disables the guard entirely and
    # leaves refusals to the system prompt.
    guard_mode: str = field(default_factory=lambda: _env_str("NIMBUS_GUARD_MODE", "steer"))

    # --- Context-memory management -------------------------------------
    # Token budget available for *dialogue history* after the system prompt
    # and the reserved generation space have been subtracted.
    history_token_budget: int = field(
        default_factory=lambda: _env_int("NIMBUS_HISTORY_TOKEN_BUDGET", 1400)
    )
    # Turns that are always kept verbatim, even if the budget is blown.
    min_verbatim_turns: int = field(
        default_factory=lambda: _env_int("NIMBUS_MIN_VERBATIM_TURNS", 2)
    )
    # Roll evicted turns into a running summary (costs one extra short
    # generation whenever eviction happens).
    enable_rolling_summary: bool = field(
        default_factory=lambda: _env_bool("NIMBUS_ENABLE_ROLLING_SUMMARY", True)
    )
    summary_max_tokens: int = field(
        default_factory=lambda: _env_int("NIMBUS_SUMMARY_MAX_TOKENS", 160)
    )

    # --- Sessions -------------------------------------------------------
    session_ttl_seconds: int = field(
        default_factory=lambda: _env_int("NIMBUS_SESSION_TTL_SECONDS", 3600)
    )
    max_sessions: int = field(
        default_factory=lambda: _env_int("NIMBUS_MAX_SESSIONS", 500)
    )
    max_message_chars: int = field(
        default_factory=lambda: _env_int("NIMBUS_MAX_MESSAGE_CHARS", 2000)
    )
    # One in-flight generation per session; a second concurrent send on the
    # same session is rejected rather than queued.
    max_concurrent_generations: int = field(
        default_factory=lambda: _env_int("NIMBUS_MAX_CONCURRENT_GENERATIONS", 4)
    )
    generation_timeout_seconds: float = field(
        default_factory=lambda: _env_float("NIMBUS_GENERATION_TIMEOUT", 180.0)
    )

    # --- Server ---------------------------------------------------------
    host: str = field(default_factory=lambda: _env_str("NIMBUS_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("NIMBUS_PORT", 8000))
    cors_origins: str = field(default_factory=lambda: _env_str("NIMBUS_CORS", "*"))


settings = Settings()
"""Process-wide settings singleton."""
