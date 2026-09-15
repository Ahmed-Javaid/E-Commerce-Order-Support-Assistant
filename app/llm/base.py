"""The engine contract every local backend implements.

Deliberately tiny: one streaming method, one health check. Everything above
this layer (conversation manager, API) is written against this interface, which
is why swapping Ollama for llama.cpp or for the deterministic test double is a
one-line config change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One message in a chat-formatted prompt."""

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationStats:
    """Per-generation timing and token counters.

    ``ttft_ms`` is measured from just before the HTTP request is issued to the
    arrival of the first non-empty token, so it includes queueing and prompt
    evaluation -- which is what a user actually feels.
    """

    ttft_ms: float = 0.0
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Decode throughput: completion tokens / (total - ttft). Excludes prompt
    #: evaluation, so it reflects steady-state generation speed.
    decode_tps: float = 0.0
    #: End-to-end throughput including prompt evaluation.
    overall_tps: float = 0.0
    model: str = ""
    engine: str = ""
    extra: dict[str, float | int | str] = field(default_factory=dict)

    def finalise(self) -> "GenerationStats":
        decode_ms = max(self.total_ms - self.ttft_ms, 1e-6)
        if self.completion_tokens:
            self.decode_tps = self.completion_tokens / (decode_ms / 1000.0)
            self.overall_tps = self.completion_tokens / (max(self.total_ms, 1e-6) / 1000.0)
        return self


@dataclass(frozen=True)
class StreamChunk:
    """A single streamed delta.

    The final chunk of a stream has ``done=True`` and carries the stats; its
    ``text`` is empty.
    """

    text: str
    done: bool = False
    stats: GenerationStats | None = None


class LLMError(RuntimeError):
    """Raised for any backend failure that the API layer should surface.

    ``code`` is a short machine-readable slug the WebSocket layer forwards to
    the client so the UI can distinguish "model is not running" from "you were
    rate-limited".
    """

    def __init__(self, message: str, code: str = "llm_error") -> None:
        super().__init__(message)
        self.code = code


@runtime_checkable
class LLMEngine(Protocol):
    """Minimal async streaming chat interface."""

    name: str
    model: str

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield :class:`StreamChunk` deltas, then one final ``done`` chunk."""
        ...

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming convenience wrapper (used for rolling summaries)."""
        ...

    async def health(self) -> dict[str, object]:
        """Report reachability and the loaded model."""
        ...

    async def warmup(self, messages: list[ChatMessage] | None = None) -> None:
        """Load the model, and pre-evaluate ``messages`` if the runtime caches prefixes."""
        ...

    async def aclose(self) -> None:
        """Release any held connections."""
        ...
