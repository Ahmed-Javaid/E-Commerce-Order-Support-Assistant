"""Engine selection.

One place that turns ``NIMBUS_ENGINE`` into a concrete backend, so no other
module needs to know which runtimes exist.
"""

from __future__ import annotations

from app.config import settings
from app.llm.base import LLMEngine, LLMError


def build_engine(name: str | None = None) -> LLMEngine:
    """Instantiate the configured backend.

    Imports are local so that an unused optional backend (llama-cpp-python)
    never has to be installed.
    """
    choice = (name or settings.engine).strip().lower()

    if choice == "ollama":
        from app.llm.ollama_engine import OllamaEngine

        return OllamaEngine()

    if choice == "llamacpp":
        from app.llm.llamacpp_engine import LlamaCppEngine

        return LlamaCppEngine()

    if choice == "mock":
        from app.llm.mock_engine import MockEngine

        return MockEngine()

    raise LLMError(
        f"Unknown engine {choice!r}. Expected one of: ollama, llamacpp, mock.",
        code="engine_misconfigured",
    )
