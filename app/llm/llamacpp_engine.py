"""llama.cpp backend (optional alternative to Ollama).

Kept because the assignment allows llama.cpp as a runtime and because it proves
the engine interface is not shaped around one vendor. It is only imported when
``NIMBUS_ENGINE=llamacpp``, so ``llama-cpp-python`` stays an optional dependency.

``llama_cpp.Llama`` is synchronous and holds the GIL during decoding, so every
call is pushed onto a worker thread and the resulting token stream is handed
back to the event loop through an ``asyncio.Queue``. A single model handle is
also not safe to use from two threads at once, hence the lock: concurrent users
are serialised at the model, but the event loop itself never blocks.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, AsyncIterator

from app.config import settings
from app.llm.base import ChatMessage, GenerationStats, LLMError, StreamChunk

_SENTINEL = object()


class LlamaCppEngine:
    """Streaming chat against an in-process GGUF model."""

    name = "llamacpp"

    def __init__(self, gguf_path: str | None = None, **kwargs: Any) -> None:
        path = gguf_path or settings.gguf_path
        if not path:
            raise LLMError(
                "NIMBUS_GGUF_PATH is not set; it must point at a .gguf model file.",
                code="engine_misconfigured",
            )
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMError(
                "llama-cpp-python is not installed. Install it or use NIMBUS_ENGINE=ollama.",
                code="engine_missing",
            ) from exc

        self.model = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        self._lock = threading.Lock()
        self._llm = Llama(
            model_path=path,
            n_ctx=settings.context_window,
            n_threads=settings.num_threads or None,
            verbose=False,
            **kwargs,
        )

    def _generate_into(
        self,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> None:
        """Runs on a worker thread; pushes deltas back onto the event loop."""

        def put(item: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        try:
            with self._lock:
                stream = self._llm.create_chat_completion(
                    messages=[m.as_dict() for m in messages],
                    temperature=temperature,
                    top_p=settings.top_p,
                    max_tokens=max_tokens,
                    stop=stop or [],
                    stream=True,
                )
                for event in stream:
                    delta = event["choices"][0].get("delta", {})
                    piece = delta.get("content") or ""
                    if piece:
                        put(piece)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
            put(LLMError(f"llama.cpp generation failed: {exc}", code="upstream_error"))
        finally:
            put(_SENTINEL)

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        started = time.perf_counter()

        worker = loop.run_in_executor(
            None,
            self._generate_into,
            queue,
            loop,
            messages,
            settings.temperature if temperature is None else temperature,
            settings.max_output_tokens if max_tokens is None else max_tokens,
            stop,
        )

        ttft_ms = 0.0
        emitted = 0
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, LLMError):
                    raise item
                if emitted == 0:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                emitted += 1
                yield StreamChunk(text=item)
        finally:
            await worker

        prompt_chars = sum(len(m.content) for m in messages)
        stats = GenerationStats(
            ttft_ms=ttft_ms,
            total_ms=(time.perf_counter() - started) * 1000.0,
            prompt_tokens=max(prompt_chars // 4, 1),
            completion_tokens=emitted,
            model=self.model,
            engine=self.name,
        )
        yield StreamChunk(text="", done=True, stats=stats.finalise())

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        parts: list[str] = []
        async for chunk in self.stream_chat(
            messages, temperature=temperature, max_tokens=max_tokens
        ):
            if not chunk.done:
                parts.append(chunk.text)
        return "".join(parts).strip()

    async def health(self) -> dict[str, object]:
        return {
            "engine": self.name,
            "model": self.model,
            "reachable": True,
            "model_present": True,
        }

    async def warmup(self, messages: list[ChatMessage] | None = None) -> None:
        probe = messages or [ChatMessage("user", "hi")]
        async for _ in self.stream_chat(probe, max_tokens=1):
            pass

    async def aclose(self) -> None:
        return None
