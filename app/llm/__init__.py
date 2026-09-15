"""LLM engine layer: a narrow async streaming interface over local runtimes."""

from app.llm.base import ChatMessage, GenerationStats, LLMEngine, LLMError, StreamChunk
from app.llm.factory import build_engine

__all__ = [
    "ChatMessage",
    "GenerationStats",
    "LLMEngine",
    "LLMError",
    "StreamChunk",
    "build_engine",
]
